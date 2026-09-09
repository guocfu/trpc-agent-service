"""R2C delivery retry and terminal-audit contracts."""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.mark.asyncio
async def test_unsent_send_is_retried_at_most_three_times_with_fixed_backoff(monkeypatch):
    from trpc_service.channels.delivery import ChannelSendError, send_with_retry

    attempts = 0
    delays: list[float] = []

    async def send():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ChannelSendError(sent=False)

    async def sleep(delay: float) -> None:
        delays.append(delay)

    monkeypatch.setattr(asyncio, "sleep", sleep)
    assert await send_with_retry(send) is True
    assert attempts == 3
    assert delays == [0.1, 0.2]


@pytest.mark.asyncio
async def test_sent_failure_is_never_retried():
    from trpc_service.channels.delivery import ChannelSendError, send_with_retry

    attempts = 0

    async def send():
        nonlocal attempts
        attempts += 1
        raise ChannelSendError(sent=True)

    assert await send_with_retry(send) is False
    assert attempts == 1


@pytest.mark.asyncio
async def test_cancellation_is_not_swallowed():
    from trpc_service.channels.delivery import send_with_retry

    async def send():
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await send_with_retry(send)


@pytest.mark.asyncio
async def test_success_is_suppressed_but_delivery_failure_is_audited_for_failures_policy():
    from trpc_service.channels.delivery import ChannelExecutionStream
    from trpc_service.gateway.channel_service import ChannelIngressService
    from trpc_service.transport.models import WorkerErrorCode, WorkerTask

    async def empty():
        if False:
            yield None

    repository = MagicMock()
    repository.append = AsyncMock()
    service = ChannelIngressService(MagicMock(), MagicMock(), execution_repository=repository)
    execution = ChannelExecutionStream(empty())
    execution.task = WorkerTask(
        protocol_version=1,
        request_id=uuid.uuid4(),
        tenant_id="tenant_default",
        app_id="app_default",
        config_version=1,
        user_id="usr_v1_" + "a" * 48,
        channel="wecom",
        session_id="ses_v1_" + "a" * 48,
        message_id="message-1",
        message="hello",
    )
    execution.delivery_events = "failures"

    await service.record_external_delivery(execution, None)
    repository.append.assert_not_awaited()

    await service.record_external_delivery(execution, WorkerErrorCode.CHANNEL_DELIVERY_FAILED)
    repository.append.assert_awaited_once()
    event = repository.append.await_args.args[0]
    assert event.outcome == "failed"
    assert event.error_code == "channel_delivery_failed"
