"""Worker execution service: WorkerTask → TenantContext → AgentApp → protocol events.

Stage 6B2 adds one centralized governance seam (``trpc_service.worker.
governance``) around execution:

- the input is inspected AFTER the receipt claim and BEFORE any model/tool
  call; a blocked input fails the receipt with the fixed
  ``content_input_blocked`` code and ZERO model invocations (the replay of
  the same message replays that fixed terminal);
- when the output action blocks, the stream is fully buffered inside the
  Worker, inspected, and at most one safe delta plus ``done`` is emitted —
  sensitive text never leaves this process as a delta, IM reply or receipt
  body; non-enforcing tenants keep live incremental streaming;
- every execution terminal carries its audit facts (``content_decision``,
  ``agent_result``, ``tool_decision``) through ``complete()/fail()`` in the
  SAME transaction; pause/finalize carry theirs through the approval
  transactions.  Sync ``chat`` and SSE ``stream`` share the same decisions.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import aclosing
from collections.abc import AsyncIterator

from trpc_agent_sdk.events import LongRunningEvent

from trpc_service.agent.app import AgentApp
from trpc_service.agent.errors import TenantAgentConfigurationError
from trpc_service.agent.execution_coordinator import SessionBusyError, SessionExecutionLostError
from trpc_service.config import ModelConfigurationError
from trpc_service.config.tenant import TenantConfig
from trpc_service.config.tenant_repository import (
    TenantConfigRepository,
    TenantRepositoryDataError,
    TenantRepositoryUnavailableError,
)
from trpc_service.governance.approval import pending_reply_for
from trpc_service.storage.approval_repository import ToolApprovalRepository
from trpc_service.worker.approval_service import make_pause_handler
from trpc_service.worker.governance import ContentGovernance, ExecutionRecorder
from trpc_service.storage.message_repository import (
    MessageReceiptRepository,
    MessageReceiptRepositoryDataError,
    MessageReceiptRepositoryUnavailableError,
    ReceiptAction,
)
from trpc_service.tenant.context import TenantContext
from trpc_service.transport.models import (
    WorkerApprovalData,
    WorkerChatResult,
    WorkerErrorCode,
    WorkerEvent,
    WorkerTask,
    WorkerToolCallData,
    WorkerToolResultData,
)

logger = logging.getLogger(__name__)

_APPROVAL_REQUIRED_RESPONSE = {"status": "approval_required"}


def _error_events(recorder: ExecutionRecorder, code: WorkerErrorCode):
    """Snapshot plus the terminal agent_result/error fact (not appended —
    the failing transaction owns it)."""
    return recorder.snapshot() + (recorder.derive("agent_result", "error", error_code=code), )


def _review_pending_events(recorder: ExecutionRecorder, tool_name: str):
    return recorder.snapshot() + (recorder.derive("tool_decision", "review_pending", tool_name=tool_name), )


class WorkerService:
    """Executes one WorkerTask through the AgentApp, producing protocol results."""

    def __init__(
        self,
        tenant_repository: TenantConfigRepository,
        agent_app: AgentApp,
        receipt_repository: MessageReceiptRepository | None = None,
        approval_repository: ToolApprovalRepository | None = None,
        usage_repository=None,
        pricing=None,
        telemetry=None,
    ) -> None:
        self._tenant_repository = tenant_repository
        self._agent_app = agent_app
        self._receipt_repository = receipt_repository
        self._approval_repository = approval_repository
        # Stage 6C: budget pre-check + post-turn usage accumulation.
        self._usage_repository = usage_repository
        self._pricing = pricing
        self._telemetry = telemetry

    def _metrics(self):
        from trpc_service.telemetry.metrics import NoopMetricsRecorder

        if self._telemetry is None:
            return NoopMetricsRecorder()
        try:
            return self._telemetry.metrics_recorder()
        except Exception:
            return NoopMetricsRecorder()

    def _count_request(self, operation: str, code: WorkerErrorCode | None) -> None:
        # Fixed low-cardinality labels only (service/operation/result/
        # error_code) — enforced structurally by MetricsRecorder.
        metrics = self._metrics()
        if code is None:
            metrics.record_counter("trpc.requests", operation=operation, result="ok")
        else:
            metrics.record_counter("trpc.requests", operation=operation, result="error", error_code=code.value)

    async def _budget_block_reason(self, task: WorkerTask, config: TenantConfig):
        from trpc_service.worker.governance import budget_block_reason

        return await budget_block_reason(task, config, self._usage_repository, self._pricing)

    async def _record_usage(self, task: WorkerTask, config: TenantConfig, usage_acc) -> None:
        from trpc_service.worker.governance import record_usage

        await record_usage(
            task,
            config,
            usage_acc,
            self._usage_repository,
            self._pricing,
            self._metrics() if self._usage_repository is not None else None,
        )

    async def chat(self, task: WorkerTask) -> WorkerChatResult:
        rid = task.request_id
        try:
            config, context = await self._resolve(task)
        except _WorkerProtocolError as exc:
            return WorkerChatResult(protocol_version=1, request_id=rid, response="", error_code=exc.code)
        except (TenantRepositoryUnavailableError, TenantRepositoryDataError):
            logger.warning("WorkerService.chat tenant repository unavailable")
            return WorkerChatResult(
                protocol_version=1,
                request_id=rid,
                response="",
                error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
            )
        except Exception:
            logger.warning("WorkerService.chat resolve failed")
            return WorkerChatResult(
                protocol_version=1,
                request_id=rid,
                response="",
                error_code=WorkerErrorCode.MODEL_RUNTIME,
            )

        governance = ContentGovernance(config.governance.content_policy)

        if self._receipt_repository is not None:
            try:
                claim = await self._receipt_repository.claim(task, task.message)
            except MessageReceiptRepositoryUnavailableError:
                logger.warning("WorkerService.chat receipt repository unavailable")
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                )
            except MessageReceiptRepositoryDataError:
                logger.warning("WorkerService.chat receipt claim data error")
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.MODEL_RUNTIME,
                )

            if claim.action == ReceiptAction.REPLAY:
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response=claim.response_text or "",
                    error_code=claim.error_code,
                )
            if claim.action == ReceiptAction.IN_PROGRESS:
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.MESSAGE_IN_PROGRESS,
                )
            if claim.action == ReceiptAction.CONFLICT:
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.IDEMPOTENCY_CONFLICT,
                )

            start_time = time.monotonic()
            recorder = ExecutionRecorder(task, claim.receipt_id)

            # Input gate: after claim, before ANY model/tool call.
            if recorder.record_input_decision(governance, task.message):
                latency_ms = int((time.monotonic() - start_time) * 1000)
                try:
                    await self._receipt_repository.fail(
                        claim.receipt_id,
                        WorkerErrorCode.CONTENT_INPUT_BLOCKED,
                        latency_ms,
                        execution_events=recorder.snapshot(),
                    )
                except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                    logger.warning(
                        "WorkerService.chat receipt fail failed type=%s",
                        type(exc).__name__,
                    )
                    return WorkerChatResult(
                        protocol_version=1,
                        request_id=rid,
                        response="",
                        error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                    )
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.CONTENT_INPUT_BLOCKED,
                )

            # Budget gate (Stage 6C): after claim, before the model, with
            # the input decision already recorded.  Zero model calls when
            # the accumulated UTC-day budget is (proven or unresolvably)
            # exhausted.
            budget_code = await self._budget_block_reason(task, config)
            if budget_code is not None:
                latency_ms = int((time.monotonic() - start_time) * 1000)
                self._metrics().record_counter(
                    "trpc.budget.rejections",
                    operation="chat",
                    result="rejected",
                    error_code=budget_code.value,
                )
                self._metrics().record_counter(
                    "trpc.requests",
                    operation="chat",
                    result="rejected_budget",
                    error_code=budget_code.value,
                )
                try:
                    await self._receipt_repository.fail(
                        claim.receipt_id,
                        budget_code,
                        latency_ms,
                        execution_events=recorder.snapshot() +
                        (recorder.derive("agent_result", "error", error_code=budget_code), ),
                    )
                except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                    logger.warning(
                        "WorkerService.chat receipt fail failed type=%s",
                        type(exc).__name__,
                    )
                    return WorkerChatResult(
                        protocol_version=1,
                        request_id=rid,
                        response="",
                        error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                    )
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=budget_code,
                )

            from trpc_service.usage.models import UsageAccumulator

            usage_acc = UsageAccumulator()
            final_text = ""
            paused_id: uuid.UUID | None = None
            try:
                # aclosing: the early returns below must close the agent
                # stream now so its trace spans end without waiting for GC
                # (Stage 6B1 cancellation rule).
                events = self._convert_events(
                    task,
                    config,
                    context,
                    pause_handler=(make_pause_handler(
                        self._approval_repository,
                        task,
                        claim.receipt_id,
                        start_time,
                        review_pending_events=lambda name: _review_pending_events(recorder, name),
                    ) if self._approval_repository is not None else None),
                    usage_acc=usage_acc)
                async with aclosing(events):
                    async for event in events:
                        if event.type == "approval":
                            paused_id = event.data.approval_id
                            continue
                        if event.type == "error":
                            latency_ms = int((time.monotonic() - start_time) * 1000)
                            try:
                                await self._receipt_repository.fail(
                                    claim.receipt_id,
                                    event.error_code,
                                    latency_ms,
                                    execution_events=_error_events(recorder, event.error_code),
                                )
                            except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                                logger.warning(
                                    "WorkerService.chat receipt fail failed type=%s",
                                    type(exc).__name__,
                                )
                                return WorkerChatResult(
                                    protocol_version=1,
                                    request_id=rid,
                                    response="",
                                    error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                                )
                            await self._record_usage(task, config, usage_acc)
                            self._count_request("chat", event.error_code)
                            return WorkerChatResult(
                                protocol_version=1,
                                request_id=rid,
                                response="",
                                error_code=event.error_code,
                            )
                        if event.type == "tool":
                            self._record_tool_decision(recorder, config, event)
                        if event.type == "delta" and isinstance(event.data, str):
                            final_text += event.data
            except TenantAgentConfigurationError:
                await self._record_usage(task, config, usage_acc)
                self._count_request("chat", WorkerErrorCode.TENANT_AGENT_CONFIGURATION)
                latency_ms = int((time.monotonic() - start_time) * 1000)
                try:
                    await self._receipt_repository.fail(
                        claim.receipt_id,
                        WorkerErrorCode.TENANT_AGENT_CONFIGURATION,
                        latency_ms,
                        execution_events=_error_events(recorder, WorkerErrorCode.TENANT_AGENT_CONFIGURATION),
                    )
                except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                    logger.warning(
                        "WorkerService.chat receipt fail failed type=%s",
                        type(exc).__name__,
                    )
                    return WorkerChatResult(
                        protocol_version=1,
                        request_id=rid,
                        response="",
                        error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                    )
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.TENANT_AGENT_CONFIGURATION,
                )
            except ModelConfigurationError:
                await self._record_usage(task, config, usage_acc)
                self._count_request("chat", WorkerErrorCode.MODEL_CONFIGURATION)
                latency_ms = int((time.monotonic() - start_time) * 1000)
                try:
                    await self._receipt_repository.fail(
                        claim.receipt_id,
                        WorkerErrorCode.MODEL_CONFIGURATION,
                        latency_ms,
                        execution_events=_error_events(recorder, WorkerErrorCode.MODEL_CONFIGURATION),
                    )
                except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                    logger.warning(
                        "WorkerService.chat receipt fail failed type=%s",
                        type(exc).__name__,
                    )
                    return WorkerChatResult(
                        protocol_version=1,
                        request_id=rid,
                        response="",
                        error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                    )
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.MODEL_CONFIGURATION,
                )
            except SessionBusyError:
                await self._record_usage(task, config, usage_acc)
                self._count_request("chat", WorkerErrorCode.SESSION_BUSY)
                latency_ms = int((time.monotonic() - start_time) * 1000)
                try:
                    await self._receipt_repository.fail(
                        claim.receipt_id,
                        WorkerErrorCode.SESSION_BUSY,
                        latency_ms,
                        execution_events=_error_events(recorder, WorkerErrorCode.SESSION_BUSY),
                    )
                except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                    logger.warning(
                        "WorkerService.chat receipt fail failed type=%s",
                        type(exc).__name__,
                    )
                    return WorkerChatResult(
                        protocol_version=1,
                        request_id=rid,
                        response="",
                        error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                    )
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.SESSION_BUSY,
                )
            except SessionExecutionLostError:
                await self._record_usage(task, config, usage_acc)
                self._count_request("chat", WorkerErrorCode.MODEL_RUNTIME)
                logger.warning("WorkerService.chat session execution lost")
                latency_ms = int((time.monotonic() - start_time) * 1000)
                try:
                    await self._receipt_repository.fail(
                        claim.receipt_id,
                        WorkerErrorCode.MODEL_RUNTIME,
                        latency_ms,
                        execution_events=_error_events(recorder, WorkerErrorCode.MODEL_RUNTIME),
                    )
                except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                    logger.warning(
                        "WorkerService.chat receipt fail failed type=%s",
                        type(exc).__name__,
                    )
                    return WorkerChatResult(
                        protocol_version=1,
                        request_id=rid,
                        response="",
                        error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                    )
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.MODEL_RUNTIME,
                )
            except Exception:
                await self._record_usage(task, config, usage_acc)
                self._count_request("chat", WorkerErrorCode.MODEL_RUNTIME)
                logger.warning("WorkerService.chat unexpected error")
                latency_ms = int((time.monotonic() - start_time) * 1000)
                try:
                    await self._receipt_repository.fail(
                        claim.receipt_id,
                        WorkerErrorCode.MODEL_RUNTIME,
                        latency_ms,
                        execution_events=_error_events(recorder, WorkerErrorCode.MODEL_RUNTIME),
                    )
                except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                    logger.warning(
                        "WorkerService.chat receipt fail failed type=%s",
                        type(exc).__name__,
                    )
                    return WorkerChatResult(
                        protocol_version=1,
                        request_id=rid,
                        response="",
                        error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                    )
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.MODEL_RUNTIME,
                )

            if paused_id is not None:
                # receipt was completed atomically by the pause transaction
                await self._record_usage(task, config, usage_acc)
                self._count_request("chat", None)
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response=pending_reply_for(paused_id),
                    error_code=None,
                )
            latency_ms = int((time.monotonic() - start_time) * 1000)
            # Output gate (sync): the buffered reply is checked BEFORE the
            # receipt body persists and before it is ever returned.
            final_text, _out_blocked = self._apply_output_decision(governance, recorder, final_text, latency_ms)
            try:
                await self._receipt_repository.complete(
                    claim.receipt_id,
                    final_text,
                    latency_ms,
                    execution_events=recorder.snapshot(),
                )
            except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                logger.warning(
                    "WorkerService.chat receipt completion failed type=%s",
                    type(exc).__name__,
                )
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                )
            await self._record_usage(task, config, usage_acc)
            self._count_request("chat", None)
            return WorkerChatResult(protocol_version=1, request_id=rid, response=final_text, error_code=None)

        # No receipt repository (legacy in-memory mode): governance still
        # applies, but there is no terminal transaction to carry audit.
        if governance.enabled:
            blocked, _decision = governance.inspect_input(task.message)
            if blocked:
                return WorkerChatResult(
                    protocol_version=1,
                    request_id=rid,
                    response="",
                    error_code=WorkerErrorCode.CONTENT_INPUT_BLOCKED,
                )
        final_text = ""
        try:
            events = self._convert_events(task, config, context)
            async with aclosing(events):  # early return ends the agent spans
                async for event in events:
                    if event.type == "error":
                        return WorkerChatResult(
                            protocol_version=1,
                            request_id=rid,
                            response="",
                            error_code=event.error_code,
                        )
                    if event.type == "delta" and isinstance(event.data, str):
                        final_text += event.data
        except TenantAgentConfigurationError:
            return WorkerChatResult(
                protocol_version=1,
                request_id=rid,
                response="",
                error_code=WorkerErrorCode.TENANT_AGENT_CONFIGURATION,
            )
        except ModelConfigurationError:
            return WorkerChatResult(
                protocol_version=1,
                request_id=rid,
                response="",
                error_code=WorkerErrorCode.MODEL_CONFIGURATION,
            )
        except SessionBusyError:
            return WorkerChatResult(
                protocol_version=1,
                request_id=rid,
                response="",
                error_code=WorkerErrorCode.SESSION_BUSY,
            )
        except SessionExecutionLostError:
            logger.warning("WorkerService.chat session execution lost")
            return WorkerChatResult(
                protocol_version=1,
                request_id=rid,
                response="",
                error_code=WorkerErrorCode.MODEL_RUNTIME,
            )
        except Exception:
            logger.warning("WorkerService.chat unexpected error")
            return WorkerChatResult(
                protocol_version=1,
                request_id=rid,
                response="",
                error_code=WorkerErrorCode.MODEL_RUNTIME,
            )

        if governance.output_enforced:
            blocked, _decision = governance.inspect_output(final_text)
            if blocked:
                final_text = governance.safe_output_text()

        return WorkerChatResult(protocol_version=1, request_id=rid, response=final_text, error_code=None)

    async def stream(self, task: WorkerTask) -> AsyncIterator[WorkerEvent]:
        rid = task.request_id
        try:
            config, context = await self._resolve(task)
        except _WorkerProtocolError as exc:
            yield WorkerEvent(
                protocol_version=1,
                request_id=rid,
                type="error",
                data=None,
                error_code=exc.code,
            )
            return
        except (TenantRepositoryUnavailableError, TenantRepositoryDataError):
            logger.warning("WorkerService.stream tenant repository unavailable")
            yield WorkerEvent(
                protocol_version=1,
                request_id=rid,
                type="error",
                data=None,
                error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
            )
            return
        except Exception:
            logger.warning("WorkerService.stream resolve failed")
            yield WorkerEvent(
                protocol_version=1,
                request_id=rid,
                type="error",
                data=None,
                error_code=WorkerErrorCode.MODEL_RUNTIME,
            )
            return

        governance = ContentGovernance(config.governance.content_policy)

        if self._receipt_repository is not None:
            try:
                claim = await self._receipt_repository.claim(task, task.message)
            except MessageReceiptRepositoryUnavailableError:
                logger.warning("WorkerService.stream receipt repository unavailable")
                yield WorkerEvent(
                    protocol_version=1,
                    request_id=rid,
                    type="error",
                    data=None,
                    error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                )
                return
            except MessageReceiptRepositoryDataError:
                logger.warning("WorkerService.stream receipt claim data error")
                yield WorkerEvent(
                    protocol_version=1,
                    request_id=rid,
                    type="error",
                    data=None,
                    error_code=WorkerErrorCode.MODEL_RUNTIME,
                )
                return

            if claim.action == ReceiptAction.REPLAY:
                if claim.error_code is not None:
                    yield WorkerEvent(
                        protocol_version=1,
                        request_id=rid,
                        type="error",
                        data=None,
                        error_code=claim.error_code,
                    )
                    return
                if claim.response_text:
                    yield WorkerEvent(
                        protocol_version=1,
                        request_id=rid,
                        type="delta",
                        data=claim.response_text,
                        error_code=None,
                    )
                yield WorkerEvent(
                    protocol_version=1,
                    request_id=rid,
                    type="done",
                    data=None,
                    error_code=None,
                )
                return
            if claim.action == ReceiptAction.IN_PROGRESS:
                yield WorkerEvent(
                    protocol_version=1,
                    request_id=rid,
                    type="error",
                    data=None,
                    error_code=WorkerErrorCode.MESSAGE_IN_PROGRESS,
                )
                return
            if claim.action == ReceiptAction.CONFLICT:
                yield WorkerEvent(
                    protocol_version=1,
                    request_id=rid,
                    type="error",
                    data=None,
                    error_code=WorkerErrorCode.IDEMPOTENCY_CONFLICT,
                )
                return

            start_time = time.monotonic()
            recorder = ExecutionRecorder(task, claim.receipt_id)

            # Input gate: identical decision path to chat() — after claim,
            # before ANY model/tool call.
            if recorder.record_input_decision(governance, task.message):
                latency_ms = int((time.monotonic() - start_time) * 1000)
                try:
                    await self._receipt_repository.fail(
                        claim.receipt_id,
                        WorkerErrorCode.CONTENT_INPUT_BLOCKED,
                        latency_ms,
                        execution_events=recorder.snapshot(),
                    )
                except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                    logger.warning(
                        "WorkerService.stream receipt fail failed type=%s",
                        type(exc).__name__,
                    )
                    yield WorkerEvent(
                        protocol_version=1,
                        request_id=rid,
                        type="error",
                        data=None,
                        error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                    )
                    return
                yield WorkerEvent(
                    protocol_version=1,
                    request_id=rid,
                    type="error",
                    data=None,
                    error_code=WorkerErrorCode.CONTENT_INPUT_BLOCKED,
                )
                return

            # Budget gate (Stage 6C): identical decision to chat() — after
            # claim/input gate, before the model.
            budget_code = await self._budget_block_reason(task, config)
            if budget_code is not None:
                latency_ms = int((time.monotonic() - start_time) * 1000)
                self._metrics().record_counter(
                    "trpc.budget.rejections",
                    operation="stream",
                    result="rejected",
                    error_code=budget_code.value,
                )
                self._metrics().record_counter(
                    "trpc.requests",
                    operation="stream",
                    result="rejected_budget",
                    error_code=budget_code.value,
                )
                try:
                    await self._receipt_repository.fail(
                        claim.receipt_id,
                        budget_code,
                        latency_ms,
                        execution_events=recorder.snapshot() +
                        (recorder.derive("agent_result", "error", error_code=budget_code), ),
                    )
                except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                    logger.warning(
                        "WorkerService.stream receipt fail failed type=%s",
                        type(exc).__name__,
                    )
                    yield WorkerEvent(
                        protocol_version=1,
                        request_id=rid,
                        type="error",
                        data=None,
                        error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                    )
                    return
                yield WorkerEvent(
                    protocol_version=1,
                    request_id=rid,
                    type="error",
                    data=None,
                    error_code=budget_code,
                )
                return

            from trpc_service.usage.models import UsageAccumulator

            usage_acc = UsageAccumulator()

            # Output enforcement buffers every public delta inside the
            # Worker: nothing sensitive can reach the Gateway, SSE, an IM
            # writer or the receipt body before the check ran.
            buffer_mode = governance.output_enforced
            buffered: list[WorkerEvent] = []
            final_text = ""
            paused_id: uuid.UUID | None = None
            try:
                # aclosing: disconnect/cancellation arriving at any yield
                # below, and the error-return path, close the agent stream
                # immediately (Stage 6B1 cancellation rule).
                events = self._convert_events(
                    task,
                    config,
                    context,
                    pause_handler=(make_pause_handler(
                        self._approval_repository,
                        task,
                        claim.receipt_id,
                        start_time,
                        review_pending_events=lambda name: _review_pending_events(recorder, name),
                    ) if self._approval_repository is not None else None),
                    usage_acc=usage_acc)
                async with aclosing(events):
                    async for event in events:
                        if event.type == "approval":
                            # The pause transaction has already committed at
                            # this point.  Buffer mode never leaks unchecked
                            # pre-pause text: tool events pass (fixed public
                            # observability), buffered deltas are released
                            # only if the accumulated text itself passes the
                            # output check, otherwise they are suppressed.
                            paused_id = event.data.approval_id
                            if buffer_mode:
                                _blocked, _decision = governance.inspect_output(final_text)
                                if not _blocked:
                                    for pending_event in buffered:
                                        yield pending_event
                                buffered.clear()
                            yield event
                            continue
                        if event.type == "delta" and isinstance(event.data, str):
                            final_text += event.data
                        if event.type == "done":
                            continue
                        if event.type == "error":
                            latency_ms = int((time.monotonic() - start_time) * 1000)
                            try:
                                await self._receipt_repository.fail(
                                    claim.receipt_id,
                                    event.error_code,
                                    latency_ms,
                                    execution_events=_error_events(recorder, event.error_code),
                                )
                                await self._record_usage(task, config, usage_acc)
                                self._count_request("stream", event.error_code)
                                yield event
                            except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                                logger.warning(
                                    "WorkerService.stream receipt fail failed type=%s",
                                    type(exc).__name__,
                                )
                                yield WorkerEvent(
                                    protocol_version=1,
                                    request_id=rid,
                                    type="error",
                                    data=None,
                                    error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                                )
                            return
                        if event.type == "tool":
                            self._record_tool_decision(recorder, config, event)
                        if buffer_mode and event.type in ("delta", "tool"):
                            buffered.append(event)
                            continue
                        yield event
            except TenantAgentConfigurationError:
                async for terminal in self._stream_fail(claim.receipt_id, recorder,
                                                        WorkerErrorCode.TENANT_AGENT_CONFIGURATION, start_time, rid,
                                                        task, config, usage_acc):
                    yield terminal
                return
            except ModelConfigurationError:
                async for terminal in self._stream_fail(claim.receipt_id, recorder, WorkerErrorCode.MODEL_CONFIGURATION,
                                                        start_time, rid, task, config, usage_acc):
                    yield terminal
                return
            except SessionBusyError:
                async for terminal in self._stream_fail(claim.receipt_id, recorder, WorkerErrorCode.SESSION_BUSY,
                                                        start_time, rid, task, config, usage_acc):
                    yield terminal
                return
            except SessionExecutionLostError:
                logger.warning("WorkerService.stream session execution lost")
                async for terminal in self._stream_fail(claim.receipt_id, recorder, WorkerErrorCode.MODEL_RUNTIME,
                                                        start_time, rid, task, config, usage_acc):
                    yield terminal
                return
            except Exception:
                logger.warning("WorkerService.stream unexpected error")
                async for terminal in self._stream_fail(claim.receipt_id, recorder, WorkerErrorCode.MODEL_RUNTIME,
                                                        start_time, rid, task, config, usage_acc):
                    yield terminal
                return

            if paused_id is None:
                latency_ms = int((time.monotonic() - start_time) * 1000)
                if not buffer_mode:
                    # Same decision path as sync chat(): records the
                    # agent_result terminal fact even without buffering.
                    final_text, _b = self._apply_output_decision(governance, recorder, final_text, latency_ms)
                if buffer_mode:
                    # Check the FULL reply first, then emit at most ONE
                    # delta (the checked text or the fixed replacement).
                    final_text, out_blocked = self._apply_output_decision(governance, recorder, final_text, latency_ms)
                    if out_blocked:
                        if final_text:
                            yield WorkerEvent(
                                protocol_version=1,
                                request_id=rid,
                                type="delta",
                                data=final_text,
                                error_code=None,
                            )
                    else:
                        for pending_event in buffered:
                            if pending_event.type == "tool":
                                yield pending_event
                        if final_text:
                            yield WorkerEvent(
                                protocol_version=1,
                                request_id=rid,
                                type="delta",
                                data=final_text,
                                error_code=None,
                            )
                    try:
                        await self._receipt_repository.complete(
                            claim.receipt_id,
                            final_text,
                            latency_ms,
                            execution_events=recorder.snapshot(),
                        )
                    except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                        logger.warning(
                            "WorkerService.stream receipt completion failed type=%s",
                            type(exc).__name__,
                        )
                        await self._record_usage(task, config, usage_acc)
                        self._count_request("stream", WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE)
                        yield WorkerEvent(
                            protocol_version=1,
                            request_id=rid,
                            type="error",
                            data=None,
                            error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                        )
                        return
                else:
                    try:
                        await self._receipt_repository.complete(
                            claim.receipt_id,
                            final_text,
                            latency_ms,
                            execution_events=recorder.snapshot(),
                        )
                    except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
                        logger.warning(
                            "WorkerService.stream receipt completion failed type=%s",
                            type(exc).__name__,
                        )
                        await self._record_usage(task, config, usage_acc)
                        self._count_request("stream", WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE)
                        yield WorkerEvent(
                            protocol_version=1,
                            request_id=rid,
                            type="error",
                            data=None,
                            error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
                        )
                        return
            await self._record_usage(task, config, usage_acc)
            self._count_request("stream", None)
            yield WorkerEvent(
                protocol_version=1,
                request_id=rid,
                type="done",
                data=None,
                error_code=None,
            )
            return

        # No receipt repository (legacy in-memory mode): input gate only.
        if governance.enabled:
            blocked, _decision = governance.inspect_input(task.message)
            if blocked:
                yield WorkerEvent(
                    protocol_version=1,
                    request_id=rid,
                    type="error",
                    data=None,
                    error_code=WorkerErrorCode.CONTENT_INPUT_BLOCKED,
                )
                return

        try:
            # aclosing: consumer disconnect at any forwarded yield closes
            # the agent stream immediately (Stage 6B1 cancellation rule).
            events = self._convert_events(task, config, context)
            async with aclosing(events):
                async for event in events:
                    yield event
        except TenantAgentConfigurationError:
            yield WorkerEvent(
                protocol_version=1,
                request_id=rid,
                type="error",
                data=None,
                error_code=WorkerErrorCode.TENANT_AGENT_CONFIGURATION,
            )
        except ModelConfigurationError:
            yield WorkerEvent(
                protocol_version=1,
                request_id=rid,
                type="error",
                data=None,
                error_code=WorkerErrorCode.MODEL_CONFIGURATION,
            )
        except SessionBusyError:
            yield WorkerEvent(
                protocol_version=1,
                request_id=rid,
                type="error",
                data=None,
                error_code=WorkerErrorCode.SESSION_BUSY,
            )
        except SessionExecutionLostError:
            logger.warning("WorkerService.stream session execution lost")
            yield WorkerEvent(
                protocol_version=1,
                request_id=rid,
                type="error",
                data=None,
                error_code=WorkerErrorCode.MODEL_RUNTIME,
            )
        except Exception:
            logger.warning("WorkerService.stream unexpected error")
            yield WorkerEvent(
                protocol_version=1,
                request_id=rid,
                type="error",
                data=None,
                error_code=WorkerErrorCode.MODEL_RUNTIME,
            )

    async def _stream_fail(
        self,
        receipt_id: uuid.UUID,
        recorder: ExecutionRecorder,
        code: WorkerErrorCode,
        start_time: float,
        rid: uuid.UUID,
        task: WorkerTask,
        config: TenantConfig,
        usage_acc=None,
    ) -> AsyncIterator[WorkerEvent]:
        latency_ms = int((time.monotonic() - start_time) * 1000)
        await self._record_usage(task, config, usage_acc)
        self._count_request("stream", code)
        try:
            await self._receipt_repository.fail(
                receipt_id,
                code,
                latency_ms,
                execution_events=_error_events(recorder, code),
            )
        except (MessageReceiptRepositoryUnavailableError, MessageReceiptRepositoryDataError) as exc:
            logger.warning(
                "WorkerService.stream receipt fail failed type=%s",
                type(exc).__name__,
            )
            yield WorkerEvent(
                protocol_version=1,
                request_id=rid,
                type="error",
                data=None,
                error_code=WorkerErrorCode.TENANT_REPOSITORY_UNAVAILABLE,
            )
            return
        yield WorkerEvent(
            protocol_version=1,
            request_id=rid,
            type="error",
            data=None,
            error_code=code,
        )

    @staticmethod
    def _record_tool_decision(recorder: ExecutionRecorder, config: TenantConfig, event: WorkerEvent) -> None:
        """tool_decision/allow facts for calls the governance explicitly
        allowed (deny/review never surface as public call events)."""
        data = event.data
        if isinstance(data, WorkerToolCallData) and data.kind == "call":
            if config.governance.tool_decisions.get(data.name) == "allow":
                recorder.add("tool_decision", "allow", tool_name=data.name)

    @staticmethod
    def _apply_output_decision(
        governance: ContentGovernance,
        recorder: ExecutionRecorder,
        final_text: str,
        latency_ms: int,
    ) -> tuple[str, bool]:
        """Shared by sync and SSE: record the output decision fact, apply the
        fixed replacement, then the agent_result success."""
        blocked = False
        if governance.output_enforced:
            blocked, decision = governance.inspect_output(final_text)
            recorder.add(
                "content_decision",
                "blocked" if blocked else "allow",
                category=decision.category,
            )
            if blocked:
                final_text = governance.safe_output_text()
        recorder.add("agent_result", "success", latency_ms=latency_ms)
        return final_text, blocked

    async def _resolve(self, task: WorkerTask) -> tuple[TenantConfig, TenantContext]:
        # Receipt identity is authoritative: a rollout must never cause an
        # already selected request to drift to the repository head.
        get_version = getattr(self._tenant_repository, "get_version", None)
        if get_version is None:
            config = await self._tenant_repository.get(task.tenant_id)
        else:
            config = await get_version(task.tenant_id, task.config_version)
        if config is None or not config.enabled:
            raise _WorkerProtocolError(WorkerErrorCode.TENANT_CONFIG_MISMATCH)
        if config.app.app_id != task.app_id:
            raise _WorkerProtocolError(WorkerErrorCode.TENANT_CONFIG_MISMATCH)
        if config.version != task.config_version:
            raise _WorkerProtocolError(WorkerErrorCode.TENANT_CONFIG_MISMATCH)

        context = TenantContext(
            tenant_id=task.tenant_id,
            app_id=config.app.app_id,
            user_id=task.user_id,
            channel=task.channel,
            session_id=task.session_id,
        )
        return config, context

    async def _convert_events(
        self,
        task: WorkerTask,
        config: TenantConfig,
        context: TenantContext,
        pause_handler=None,
        usage_acc=None,
    ) -> AsyncIterator[WorkerEvent]:
        """Convert SDK events from AgentApp.run() into WorkerEvent protocol objects.

        A genuine ``LongRunningEvent`` carrying exactly the 6A1
        ``approval_required`` verdict triggers the pause handler (when
        persistence is wired): the handler atomically creates the pending
        approval and completes the original receipt with the fixed pending
        reply; the converted stream then emits ``approval`` and the caller's
        normal ``done``.  Any other long-running content must NOT create an
        approval (fail-safe to a runtime error).
        """
        logger.warning("processing session session_id=%s tenant=%s", task.session_id, task.tenant_id)
        rid = task.request_id
        seen_partial_ids: set[str] = set()
        # P1-1: tools under `review` must never appear as public tool events
        # (no args leak); their lifecycle is observable only via approval.
        review_tool_names = {
            name
            for name, decision in config.governance.tool_decisions.items() if decision == "review"
        }

        agent_stream = self._agent_app.run(
            config=config,
            context=context,
            session_id=task.session_id,
            user_input=task.message,
        )
        # aclosing: early returns below and consumer cancellation close the
        # agent chain (and its agent.turn span) immediately.
        async with aclosing(agent_stream) as events:
            async for event in events:
                if usage_acc is not None:
                    # Stage 6C: SDK usage is the ONLY token source; it rides
                    # the final (partial=False) event of every LLM call and
                    # is deduplicated per event id inside the accumulator.
                    usage_acc.add_event(getattr(event, "id", None), getattr(event, "usage_metadata", None))
                if isinstance(event, LongRunningEvent):
                    response_ok = (event.function_response is not None
                                   and event.function_response.response == _APPROVAL_REQUIRED_RESPONSE)
                    if not response_ok or pause_handler is None:
                        logger.warning("worker unexpected long-running event")
                        yield WorkerEvent(
                            protocol_version=1,
                            request_id=rid,
                            type="error",
                            data=None,
                            error_code=WorkerErrorCode.MODEL_RUNTIME,
                        )
                        return
                    approval_id, err = await pause_handler(
                        event.function_call.id,
                        event.function_call.name,
                        dict(event.function_call.args or {}),
                    )
                    if err is not None:
                        yield WorkerEvent(
                            protocol_version=1,
                            request_id=rid,
                            type="error",
                            data=None,
                            error_code=err,
                        )
                        return
                    yield WorkerEvent(
                        protocol_version=1,
                        request_id=rid,
                        type="approval",
                        data=WorkerApprovalData(approval_id=approval_id, tool_name=event.function_call.name),
                        error_code=None,
                    )
                    continue
                if event.error_code:
                    yield WorkerEvent(
                        protocol_version=1,
                        request_id=rid,
                        type="error",
                        data=None,
                        error_code=WorkerErrorCode.MODEL_RUNTIME,
                    )
                    return
                if not event.content or not event.content.parts:
                    continue
                for part in event.content.parts:
                    if part.thought:
                        continue
                    if part.text:
                        if event.partial:
                            seen_partial_ids.add(event.id)
                            yield WorkerEvent(
                                protocol_version=1,
                                request_id=rid,
                                type="delta",
                                data=part.text,
                                error_code=None,
                            )
                        elif event.id in seen_partial_ids:
                            continue
                        else:
                            yield WorkerEvent(
                                protocol_version=1,
                                request_id=rid,
                                type="delta",
                                data=part.text,
                                error_code=None,
                            )
                    elif part.function_call:
                        if part.function_call.name in review_tool_names:
                            continue
                        yield WorkerEvent(
                            protocol_version=1,
                            request_id=rid,
                            type="tool",
                            data=WorkerToolCallData(
                                kind="call",
                                name=part.function_call.name,
                                args=dict(part.function_call.args or {}),
                            ),
                            error_code=None,
                        )
                    elif part.function_response:
                        if part.function_response.name in review_tool_names:
                            continue
                        yield WorkerEvent(
                            protocol_version=1,
                            request_id=rid,
                            type="tool",
                            data=WorkerToolResultData(
                                kind="result",
                                name=part.function_response.name,
                                response=part.function_response.response,
                            ),
                            error_code=None,
                        )

        yield WorkerEvent(
            protocol_version=1,
            request_id=rid,
            type="done",
            data=None,
            error_code=None,
        )

    async def close(self) -> None:
        await self._agent_app.close()


class _WorkerProtocolError(Exception):

    def __init__(self, code: WorkerErrorCode) -> None:
        self.code = code
        super().__init__(code.value)


__all__ = ["WorkerService"]
