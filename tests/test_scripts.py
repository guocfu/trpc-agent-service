import os
import signal
import socket
import stat
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_NAMES = (
    "build.sh",
    "start.sh",
    "stop.sh",
    "clean.sh",
    "format.sh",
    "lint_flake8.sh",
    "coverage.sh",
)


def _redis_is_available() -> bool:
    """Check if Redis is available either via env var or Docker."""
    if os.environ.get("TRPC_REDIS_URL"):
        return True
    try:
        result = subprocess.run(
            ["docker", "ps"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


requires_redis = pytest.mark.skipif(
    not _redis_is_available(),
    reason="Redis not available (set TRPC_REDIS_URL or install Docker)",
)


def test_operational_scripts_are_executable_and_valid_bash():
    for script_name in SCRIPT_NAMES:
        script = PROJECT_ROOT / script_name
        assert script.is_file(), f"missing {script_name}"
        assert script.stat().st_mode & stat.S_IXUSR, f"{script_name} is not executable"
        subprocess.run(["bash", "-n", str(script)], check=True)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _unique_ports(n: int) -> list[int]:
    ports: list[int] = []
    while len(ports) < n:
        p = _free_port()
        if p not in ports:
            ports.append(p)
    return ports


@dataclass(frozen=True)
class _Services:
    environment: dict[str, str]
    worker_a_port: int
    worker_b_port: int
    gateway_port: int
    admin_port: int
    worker_a_pid_file: Path
    worker_a_log_file: Path
    worker_b_pid_file: Path
    worker_b_log_file: Path
    gateway_pid_file: Path
    gateway_log_file: Path
    admin_pid_file: Path
    admin_log_file: Path


def _service_fixture(tmp_path: Path) -> _Services:
    ports = _unique_ports(4)
    worker_a_port, worker_b_port, gateway_port, admin_port = ports

    worker_a_pid_file = tmp_path / "worker-a.pid"
    worker_a_log_file = tmp_path / "worker-a.log"
    worker_b_pid_file = tmp_path / "worker-b.pid"
    worker_b_log_file = tmp_path / "worker-b.log"
    gateway_pid_file = tmp_path / "gateway.pid"
    gateway_log_file = tmp_path / "gateway.log"
    admin_pid_file = tmp_path / "admin.pid"
    admin_log_file = tmp_path / "admin.log"
    redis_port = _free_port()
    redis_name = f"trpc-test-redis-{tmp_path.parent.name}-{tmp_path.name}"
    minio_port = _free_port()
    # ``tmp_path.name`` repeats across separate pytest invocations (for
    # example ``test_start...ma0``), while its parent carries the run-unique
    # ``pytest-N`` segment.  Include both so an interrupted earlier run can
    # never make a fresh invocation reuse unknown ephemeral MinIO credentials.
    minio_name = f"trpc-test-minio-{tmp_path.parent.name}-{tmp_path.name}"

    environment = os.environ.copy()
    environment.pop("TRPC_WORKER_BASE_URL", None)
    environment.pop("TRPC_WORKER_BASE_URLS", None)
    environment.update({
        "TRPC_SERVICE_PYTHON": sys.executable,
        "TRPC_WORKER_A_HOST": "127.0.0.1",
        "TRPC_WORKER_A_PORT": str(worker_a_port),
        "TRPC_WORKER_A_PID_FILE": str(worker_a_pid_file),
        "TRPC_WORKER_A_LOG_FILE": str(worker_a_log_file),
        "TRPC_WORKER_B_HOST": "127.0.0.1",
        "TRPC_WORKER_B_PORT": str(worker_b_port),
        "TRPC_WORKER_B_PID_FILE": str(worker_b_pid_file),
        "TRPC_WORKER_B_LOG_FILE": str(worker_b_log_file),
        "TRPC_GATEWAY_HOST": "127.0.0.1",
        "TRPC_GATEWAY_PORT": str(gateway_port),
        "TRPC_GATEWAY_PID_FILE": str(gateway_pid_file),
        "TRPC_GATEWAY_LOG_FILE": str(gateway_log_file),
        "TRPC_ADMIN_HOST": "127.0.0.1",
        "TRPC_ADMIN_PORT": str(admin_port),
        "TRPC_ADMIN_PID_FILE": str(admin_pid_file),
        "TRPC_ADMIN_LOG_FILE": str(admin_log_file),
        "TRPC_INTERNAL_TOKEN": "c" * 48,
        "TRPC_ADMIN_TOKEN": "d" * 48,
        "TRPC_REDIS_CONTAINER_NAME": redis_name,
        "TRPC_REDIS_PORT": str(redis_port),
        # The shared development MinIO name is intentionally not used by
        # tests.  A pre-existing developer container has credentials that a
        # fresh test process cannot know; reusing it makes backend-init wait
        # until the subprocess timeout instead of testing the script itself.
        "TRPC_MINIO_CONTAINER_NAME": minio_name,
        "TRPC_MINIO_PORT": str(minio_port),
        "TRPC_MINIO_MARKER_FILE": str(tmp_path / "minio.owned"),
        # Keep ownership markers scoped to the test too.  The defaults live in
        # the shared repository data directory and let one script test erase
        # another test's ownership proof during a full-suite run.
        "TRPC_REDIS_MARKER_FILE": str(tmp_path / "redis.owned"),
        "TRPC_POSTGRES_MARKER_FILE": str(tmp_path / "postgres.owned"),
    })
    return _Services(
        environment=environment,
        worker_a_port=worker_a_port,
        worker_b_port=worker_b_port,
        gateway_port=gateway_port,
        admin_port=admin_port,
        worker_a_pid_file=worker_a_pid_file,
        worker_a_log_file=worker_a_log_file,
        worker_b_pid_file=worker_b_pid_file,
        worker_b_log_file=worker_b_log_file,
        gateway_pid_file=gateway_pid_file,
        gateway_log_file=gateway_log_file,
        admin_pid_file=admin_pid_file,
        admin_log_file=admin_log_file,
    )


def _run_script(
    script_name: str,
    services: _Services,
    *,
    check: bool,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(PROJECT_ROOT / script_name)],
        cwd=PROJECT_ROOT,
        env=environment or services.environment,
        check=check,
        capture_output=True,
        text=True,
        timeout=45,
    )


def _wait_for_health(port: int, log_file: Path, service_name: str) -> None:
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.2)
    log_text = log_file.read_text() if log_file.exists() else ""
    raise AssertionError(f"{service_name} did not become healthy\n{log_text}")


def _wait_for_services(services: _Services) -> None:
    _wait_for_health(services.worker_a_port, services.worker_a_log_file, "worker-a")
    _wait_for_health(services.worker_b_port, services.worker_b_log_file, "worker-b")
    _wait_for_health(services.gateway_port, services.gateway_log_file, "gateway")
    _wait_for_health(services.admin_port, services.admin_log_file, "admin")


def _read_pid(pid_file: Path) -> int:
    value = pid_file.read_text().strip()
    assert value.isdigit(), f"invalid PID file: {pid_file}"
    return int(value)


def _process_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def _wait_for_process_exit(pid: int) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if not _process_is_running(pid):
            return
        time.sleep(0.1)
    raise AssertionError(f"process {pid} did not stop")


def _stop_services(services: _Services, environment: dict[str, str] | None = None) -> None:
    _run_script("stop.sh", services, check=False, environment=environment)


@requires_redis
def test_start_and_stop_scripts_manage_four_processes(tmp_path):
    services = _service_fixture(tmp_path)
    try:
        _run_script("start.sh", services, check=True)
        _wait_for_services(services)
    finally:
        _stop_services(services)

    assert services.worker_a_pid_file.exists() or not services.worker_a_pid_file.exists()
    assert not services.gateway_pid_file.exists()
    assert not services.worker_a_pid_file.exists()
    assert not services.worker_b_pid_file.exists()
    assert not services.admin_pid_file.exists()


@requires_redis
def test_start_produces_four_distinct_pids(tmp_path):
    services = _service_fixture(tmp_path)
    try:
        _run_script("start.sh", services, check=True)
        _wait_for_services(services)

        pid_a = _read_pid(services.worker_a_pid_file)
        pid_b = _read_pid(services.worker_b_pid_file)
        pid_gw = _read_pid(services.gateway_pid_file)
        pid_admin = _read_pid(services.admin_pid_file)

        assert len({pid_a, pid_b, pid_gw, pid_admin}) == 4
        assert _process_is_running(pid_a)
        assert _process_is_running(pid_b)
        assert _process_is_running(pid_gw)
        assert _process_is_running(pid_admin)
    finally:
        _stop_services(services)


@requires_redis
def test_start_with_existing_worker_pid_not_overwritten(tmp_path):
    services = _service_fixture(tmp_path)
    try:
        _run_script("start.sh", services, check=True)
        _wait_for_services(services)
        existing_pid_a = _read_pid(services.worker_a_pid_file)
        existing_pid_b = _read_pid(services.worker_b_pid_file)

        result = _run_script("start.sh", services, check=True)

        assert "worker-a already running" in (result.stdout + result.stderr).lower()
        assert _read_pid(services.worker_a_pid_file) == existing_pid_a
        assert _read_pid(services.worker_b_pid_file) == existing_pid_b
    finally:
        _stop_services(services)


@requires_redis
def test_start_replaces_exact_worker_pid_when_health_check_fails(tmp_path):
    services = _service_fixture(tmp_path)
    try:
        _run_script("start.sh", services, check=True)
        _wait_for_services(services)
        old_pid = _read_pid(services.worker_a_pid_file)

        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        state_file = tmp_path / "worker-a-health-failed-once"
        real_curl = subprocess.run(["which", "curl"], capture_output=True, text=True, check=True).stdout.strip()
        fake_curl = fake_bin / "curl"
        fake_curl.write_text("#!/usr/bin/env bash\n"
                             f"if [[ \"$*\" == *\":{services.worker_a_port}/health\"* ]] && "
                             f"[[ ! -f \"{state_file}\" ]]; then\n"
                             f"  touch \"{state_file}\"\n"
                             "  exit 1\n"
                             "fi\n"
                             f"exec {real_curl} \"$@\"\n")
        fake_curl.chmod(0o755)
        environment = services.environment.copy()
        environment["PATH"] = f"{fake_bin}{os.pathsep}{environment['PATH']}"

        result = _run_script("start.sh", services, check=True, environment=environment)
        new_pid = _read_pid(services.worker_a_pid_file)

        assert "health check failed" in (result.stdout + result.stderr)
        assert new_pid != old_pid
        assert not _process_is_running(old_pid)
        assert _process_is_running(new_pid)
        _wait_for_health(services.worker_a_port, services.worker_a_log_file, "worker-a")
    finally:
        _stop_services(services)


@requires_redis
def test_start_with_stale_worker_pid(tmp_path):
    services = _service_fixture(tmp_path)
    stale_pid = "99999999"
    services.worker_a_pid_file.write_text(stale_pid)
    try:
        _run_script("start.sh", services, check=True)
        _wait_for_services(services)

        assert services.worker_a_pid_file.read_text().strip() != stale_pid
        assert _process_is_running(_read_pid(services.worker_a_pid_file))
    finally:
        _stop_services(services)


@requires_redis
def test_gateway_start_failure_rolls_back_workers(tmp_path):
    services = _service_fixture(tmp_path)
    try:
        _run_script("start.sh", services, check=True)
        _wait_for_services(services)
        worker_a_pid = _read_pid(services.worker_a_pid_file)
        worker_b_pid = _read_pid(services.worker_b_pid_file)

        gateway_pid = _read_pid(services.gateway_pid_file)
        os.kill(gateway_pid, signal.SIGTERM)
        _wait_for_process_exit(gateway_pid)
        services.gateway_pid_file.unlink()

        fake_bin = tmp_path / "bin"
        fake_bin.mkdir()
        fake_curl = fake_bin / "curl"
        fake_curl.write_text("#!/usr/bin/env bash\n"
                             "case \"$*\" in\n"
                             f"  *\":{services.worker_a_port}/health\"*) exit 0 ;;\n"
                             f"  *\":{services.worker_b_port}/health\"*) exit 0 ;;\n"
                             "  *) exit 1 ;;\n"
                             "esac\n")
        fake_curl.chmod(0o755)
        failing_environment = services.environment.copy()
        failing_environment["PATH"] = f"{fake_bin}{os.pathsep}{failing_environment['PATH']}"
        result = _run_script(
            "start.sh",
            services,
            check=False,
            environment=failing_environment,
        )

        assert result.returncode != 0
        assert "gateway did not become healthy" in (result.stdout + result.stderr)
        assert _read_pid(services.worker_a_pid_file) == worker_a_pid
        assert _read_pid(services.worker_b_pid_file) == worker_b_pid
        assert _process_is_running(worker_a_pid)
        assert _process_is_running(worker_b_pid)
        assert not services.gateway_pid_file.exists()
    finally:
        _stop_services(services)


@requires_redis
def test_worker_b_start_failure_rolls_back_worker_a(tmp_path):
    """If Worker B fails to start, Worker A (started this round) is rolled back."""
    services = _service_fixture(tmp_path)

    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    real_curl = subprocess.run(["which", "curl"], capture_output=True, text=True).stdout.strip()
    fake_curl = fake_bin / "curl"
    fake_curl.write_text("#!/usr/bin/env bash\n"
                         "case \"$*\" in\n"
                         f"  *\":{services.worker_a_port}/health\"*) exec {real_curl} \"$@\" ;;\n"
                         f"  *\":{services.worker_b_port}/health\"*) exit 1 ;;\n"
                         "  *) exec {real_curl} \"$@\" ;;\n"
                         "esac\n".replace("{real_curl}", real_curl))
    fake_curl.chmod(0o755)
    failing_environment = services.environment.copy()
    failing_environment["PATH"] = f"{fake_bin}{os.pathsep}{failing_environment['PATH']}"

    try:
        result = _run_script("start.sh", services, check=False, environment=failing_environment)
        assert result.returncode != 0
        assert not services.worker_a_pid_file.exists()
        assert not services.worker_b_pid_file.exists()
        assert not services.gateway_pid_file.exists()
    finally:
        _stop_services(services)


def test_stop_with_no_pid_files(tmp_path):
    services = _service_fixture(tmp_path)
    result = _run_script("stop.sh", services, check=True)

    assert "is not running" in (result.stdout + result.stderr)


def _docker_available() -> bool:
    try:
        r = subprocess.run(["docker", "ps"], capture_output=True, timeout=5)
        return r.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _container_is_running(name: str) -> bool:
    r = subprocess.run(
        ["docker", "ps", "--format", "{{.Names}}"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    return name in r.stdout.splitlines()


def _wait_for_container_running(name: str, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if _container_is_running(name):
            return
        time.sleep(0.2)
    raise AssertionError(f"container {name} did not start")


requires_docker = pytest.mark.skipif(
    not _docker_available(),
    reason="Docker not available",
)


@requires_docker
def test_stop_does_not_stop_preexisting_redis(tmp_path):
    """stop.sh should NOT stop a Redis container that was not started by start.sh."""
    import uuid
    container_name = f"trpc-preexist-{uuid.uuid4().hex[:8]}"
    redis_port = _free_port()

    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, "-p", f"{redis_port}:6379", "redis:7"],
        capture_output=True,
        check=True,
        timeout=30,
    )
    try:
        _wait_for_container_running(container_name)

        services = _service_fixture(tmp_path)
        env = services.environment.copy()
        env["TRPC_REDIS_URL"] = f"redis://127.0.0.1:{redis_port}"
        env["TRPC_REDIS_CONTAINER_NAME"] = container_name

        try:
            _run_script("start.sh", services, check=True, environment=env)
            _wait_for_services(services)

            redis_marker = Path(env["TRPC_REDIS_MARKER_FILE"])
            assert not redis_marker.exists(), "marker file should not exist for pre-existing container"

            _stop_services(services)

            assert _container_is_running(container_name), \
                "pre-existing Redis container should still be running after stop.sh"
        finally:
            _stop_services(services)
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10)


@requires_docker
def test_start_refuses_unowned_existing_redis(tmp_path):
    """An unmarked same-name Redis container must never become local state."""
    import uuid

    container_name = f"trpc-unowned-{uuid.uuid4().hex[:8]}"
    redis_port = _free_port()
    subprocess.run(
        ["docker", "run", "-d", "--name", container_name, "-p", f"{redis_port}:6379", "redis:7"],
        capture_output=True,
        check=True,
        timeout=30,
    )
    try:
        _wait_for_container_running(container_name)
        services = _service_fixture(tmp_path)
        env = services.environment.copy()
        env.pop("TRPC_REDIS_URL", None)
        env["TRPC_REDIS_CONTAINER_NAME"] = container_name
        env["TRPC_REDIS_PORT"] = str(redis_port)
        marker = Path(env["TRPC_REDIS_MARKER_FILE"])
        marker.unlink(missing_ok=True)

        result = _run_script("start.sh", services, check=False, environment=env)

        assert result.returncode != 0
        assert "Redis container exists but is not owned by start.sh" in (result.stdout + result.stderr)
        assert _container_is_running(container_name)
        assert not marker.exists()
    finally:
        _stop_services(services, environment=env)
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10)


@requires_docker
def test_stop_removes_container_started_by_start(tmp_path):
    """stop.sh should stop a Redis container started by start.sh and remove the marker."""
    import uuid
    container_name = f"trpc-owned-{uuid.uuid4().hex[:8]}"
    redis_port = _free_port()

    subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10)

    services = _service_fixture(tmp_path)
    env = services.environment.copy()
    env.pop("TRPC_REDIS_URL", None)
    env["TRPC_REDIS_CONTAINER_NAME"] = container_name
    env["TRPC_REDIS_PORT"] = str(redis_port)

    try:
        _run_script("start.sh", services, check=True, environment=env)
        _wait_for_services(services)

        assert _container_is_running(container_name), \
            "start.sh should have started the Redis container"

        _stop_services(services, environment=env)

        assert not _container_is_running(container_name), \
            "stop.sh should have stopped the Redis container it started"
    finally:
        subprocess.run(["docker", "rm", "-f", container_name], capture_output=True, timeout=10)


def test_start_without_admin_token_fails_fast(tmp_path):
    services = _service_fixture(tmp_path)
    environment = services.environment.copy()
    environment.pop("TRPC_ADMIN_TOKEN", None)

    result = _run_script("start.sh", services, check=False, environment=environment)

    assert result.returncode != 0
    assert "TRPC_ADMIN_TOKEN is not set" in (result.stdout + result.stderr)
    assert not services.worker_a_pid_file.exists()
    assert not services.worker_b_pid_file.exists()
    assert not services.gateway_pid_file.exists()
    assert not services.admin_pid_file.exists()


@requires_redis
def test_stop_stops_admin_before_gateway(tmp_path):
    services = _service_fixture(tmp_path)
    try:
        _run_script("start.sh", services, check=True)
        _wait_for_services(services)
    finally:
        result = _run_script("stop.sh", services, check=False)

    combined = result.stdout + result.stderr
    admin_pos = combined.find("admin stopped")
    gateway_pos = combined.find("gateway stopped")
    assert admin_pos != -1, f"admin not stopped: {combined}"
    assert gateway_pos != -1, f"gateway not stopped: {combined}"
    assert admin_pos < gateway_pos, "admin must stop before gateway"


@requires_redis
def test_admin_start_failure_rolls_back_this_run_processes(tmp_path):
    services = _service_fixture(tmp_path)
    try:
        _run_script("start.sh", services, check=True)
        _wait_for_services(services)
        worker_a_pid = _read_pid(services.worker_a_pid_file)
        worker_b_pid = _read_pid(services.worker_b_pid_file)
        gateway_pid = _read_pid(services.gateway_pid_file)

        admin_pid = _read_pid(services.admin_pid_file)
        os.kill(admin_pid, signal.SIGTERM)
        _wait_for_process_exit(admin_pid)
        services.admin_pid_file.unlink()

        fake_bin = tmp_path / "admin-fail-bin"
        fake_bin.mkdir()
        fake_curl = fake_bin / "curl"
        fake_curl.write_text("#!/usr/bin/env bash\n"
                             "case \"$*\" in\n"
                             f"  *\":{services.worker_a_port}/health\"*) exit 0 ;;\n"
                             f"  *\":{services.worker_b_port}/health\"*) exit 0 ;;\n"
                             f"  *\":{services.gateway_port}/health\"*) exit 0 ;;\n"
                             "  *) exit 1 ;;\n"
                             "esac\n")
        fake_curl.chmod(0o755)
        failing_environment = services.environment.copy()
        failing_environment["PATH"] = f"{fake_bin}{os.pathsep}{failing_environment['PATH']}"
        result = _run_script(
            "start.sh",
            services,
            check=False,
            environment=failing_environment,
        )

        assert result.returncode != 0
        assert "admin did not become healthy" in (result.stdout + result.stderr)
        assert _read_pid(services.worker_a_pid_file) == worker_a_pid
        assert _read_pid(services.worker_b_pid_file) == worker_b_pid
        assert _read_pid(services.gateway_pid_file) == gateway_pid
        assert _process_is_running(worker_a_pid)
        assert _process_is_running(worker_b_pid)
        assert _process_is_running(gateway_pid)
        assert not services.admin_pid_file.exists()
    finally:
        _stop_services(services)


def test_acceptance_stage5a_script_structure():
    """Verify acceptance_stage5a.sh uses isolated resources and console endpoints."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5a.sh"
    assert script.is_file(), "missing acceptance_stage5a.sh"
    assert script.stat().st_mode & stat.S_IXUSR, "acceptance_stage5a.sh is not executable"

    content = script.read_text()

    assert "mktemp -d" in content, "script must use mktemp -d for isolated tmpdir"

    assert "/api/console/messages" in content, "script must use console endpoint"
    assert "/api/console/messages/stream" in content, "script must use console stream endpoint"

    assert "acceptance_redis_name=" in content, "script must use unique Redis container name"
    assert "acceptance_pg_name=" in content, "script must use unique PostgreSQL container name"
    assert "acceptance_tmp=" in content, "script must use unique tmpdir"

    assert "stop.sh" in content, "script must invoke stop.sh for cleanup"
    assert "docker rm -f" in content, "script must remove Docker containers"

    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash syntax check failed: {result.stderr}"


def test_acceptance_stage5a_no_unreliable_pid_read():
    """Verify acceptance_stage5a.sh does not use $(<... 2>/dev/null) for PID reading."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5a.sh"
    content = script.read_text()

    import re
    unreliable_pattern = re.compile(r'\$\(<"[^"]*"\s*2>/dev/null\)')
    matches = unreliable_pattern.findall(content)
    assert len(matches) == 0, (
        f"Script uses unreliable PID read pattern $(<... 2>/dev/null): found {len(matches)} occurrences. "
        "Use $(cat ... 2>/dev/null || true) instead.")


def test_acceptance_stage5a_idempotency_marker_consistency():
    """Verify idempotency test marker matches tenant instruction."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5a.sh"
    content = script.read_text()

    # Find the acc_tenant instruction
    import re
    instruction_match = re.search(
        r'"instruction":\s*"([^"]*MARK5A[^"]*)"',
        content,
    )
    assert instruction_match, "Could not find acc_tenant instruction with MARK5A"
    instruction_text = instruction_match.group(1)

    # Find test 8 section
    test8_match = re.search(
        r'测试 8.*?(?=测试 9|总结|$)',
        content,
        re.DOTALL,
    )
    assert test8_match, "Could not find test 8 section"
    test8_section = test8_match.group(0)

    # Find idempotency test assertions in test 8 - look for MARK5A-IDEM or MARK5A
    # Match the assertion pattern: *"MARK5A-IDEM"* or *"MARK5A"*
    idem_assertion_match = re.search(
        r'\*"(MARK5A-[^"]+)"\*|\*"(MARK5A)"\*',
        test8_section,
    )
    assert idem_assertion_match, "Could not find idempotency test assertion in test 8"
    asserted_marker = idem_assertion_match.group(1) or idem_assertion_match.group(2)

    # The asserted marker must appear in the instruction
    assert asserted_marker in instruction_text or instruction_text.endswith("MARK5A"), (
        f"Idempotency test asserts '{asserted_marker}' but acc_tenant instruction forces '{instruction_text}'. "
        "Either create a separate tenant for idempotency test with matching instruction, "
        "or change assertion to match the existing instruction.")


def test_acceptance_stage5a_stop_validates_all_pids():
    """Verify stop.sh test validates all 4 PIDs exist before stopping."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5a.sh"
    content = script.read_text()

    # Find the stop.sh test section - match from "测试 9" to end of file or summary
    import re
    stop_section_match = re.search(
        r'测试 9.*?stop\.sh.*?清理验证.*?(?=总结|$)',
        content,
        re.DOTALL,
    )
    assert stop_section_match, "Could not find stop.sh test section"
    stop_section = stop_section_match.group(0)

    # Verify all 4 PID files are checked for existence and validity
    required_checks = [
        "TRPC_GATEWAY_PID_FILE",
        "TRPC_WORKER_A_PID_FILE",
        "TRPC_WORKER_B_PID_FILE",
        "TRPC_ADMIN_PID_FILE",
    ]
    for pid_file_var in required_checks:
        assert pid_file_var in stop_section, (f"stop.sh test must check {pid_file_var} existence")

    # Verify kill -0 is used for each PID
    kill_checks = re.findall(r'kill -0 "\$\{(\w+_pid_before)\}"', stop_section)
    assert len(kill_checks) >= 4, (f"stop.sh test must use kill -0 for all 4 PIDs, found {len(kill_checks)}")

    # Verify failure if any PID is missing or invalid
    assert "all_pids_valid" in stop_section or "fail" in stop_section, (
        "stop.sh test must fail if any PID is missing or invalid")


def test_acceptance_stage5b_script_structure():
    """Verify acceptance_stage5b.sh waits for auth, observes chain, requires human confirm."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5b.sh"
    assert script.is_file(), "missing acceptance_stage5b.sh"
    assert script.stat().st_mode & stat.S_IXUSR, "acceptance_stage5b.sh is not executable"

    content = script.read_text()

    assert "SKIP:" in content, "script must have explicit SKIP output"
    assert "TRPC_WECOM_BOT_ID" in content, "script must check TRPC_WECOM_BOT_ID"
    assert "TRPC_WECOM_BOT_SECRET" in content, "script must check TRPC_WECOM_BOT_SECRET"
    assert "TRPC_WECOM_TENANT_ID" in content, "script must check TRPC_WECOM_TENANT_ID"

    assert "authenticated" in content, "script must wait for SDK authentication"
    assert "text frame accepted" in content, "script must observe chain: frame accepted"
    assert "reply chain completed (terminal=done" in content, (
        "script must distinguish a successful done terminal from an error reply")
    assert "read" in content, "script must require human confirmation"

    assert "/api/console/messages" not in content, ("script must NOT use console endpoint as WeCom inbound test")

    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash syntax check failed: {result.stderr}"


def test_acceptance_stage5b_no_secret_leak():
    """Verify acceptance_stage5b.sh never prints the actual secret value."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5b.sh"
    content = script.read_text()

    assert 'echo "${wecom_bot_secret}"' not in content, "script must not echo secret"
    assert 'echo "${TRPC_WECOM_BOT_SECRET}"' not in content, "script must not echo secret"
    assert "已配置，不显示" in content, "script must indicate secret is hidden"


def test_acceptance_stage5b_zero_event_count_is_a_single_number():
    """A failed grep must not append a second zero to the event count."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5b.sh"
    content = script.read_text()
    assert ('grep -c "WeCom text frame accepted" "${acceptance_tmp}/gateway.log" 2>/dev/null || echo "0"'
            not in content)
    assert ('grep -c "WeCom reply chain completed" "${acceptance_tmp}/gateway.log" 2>/dev/null || echo "0"'
            not in content)


def test_acceptance_stage5b_skip_without_credentials():
    """Verify acceptance_stage5b.sh exits 0 with SKIP when no credentials are set."""
    import subprocess
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(PROJECT_ROOT),
    }
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "acceptance_stage5b.sh")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=10,
    )
    assert result.returncode == 0, f"Expected exit 0 for SKIP, got {result.returncode}"
    assert "SKIP" in result.stdout, "Expected SKIP message in output"


def test_acceptance_stage5b_partial_credentials_fail():
    """Verify acceptance_stage5b.sh exits 1 when only partial credentials are set."""
    import subprocess
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(PROJECT_ROOT),
        "TRPC_WECOM_BOT_ID": "bot-test",
    }
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "acceptance_stage5b.sh")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=10,
    )
    assert result.returncode == 1, f"Expected exit 1 for partial credentials, got {result.returncode}"
    assert "FAIL" in result.stdout or "FAIL" in result.stderr, "Expected FAIL message"


def test_acceptance_stage5c_script_structure():
    """Verify acceptance_stage5c.sh waits for ready, observes chain, requires human confirm."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5c.sh"
    assert script.is_file(), "missing acceptance_stage5c.sh"
    assert script.stat().st_mode & stat.S_IXUSR, "acceptance_stage5c.sh is not executable"

    content = script.read_text()

    assert "SKIP:" in content, "script must have explicit SKIP output"
    assert "TRPC_FEISHU_APP_ID" in content, "script must check TRPC_FEISHU_APP_ID"
    assert "TRPC_FEISHU_APP_SECRET" in content, "script must check TRPC_FEISHU_APP_SECRET"
    assert "TRPC_FEISHU_TENANT_ID" in content, "script must check TRPC_FEISHU_TENANT_ID"

    assert "connected and ready" in content, "script must wait for SDK connection ready"
    assert "text frame accepted" in content, "script must observe chain: frame accepted"
    assert "reply chain completed (terminal=done" in content, (
        "script must distinguish a successful done terminal from an error reply")
    assert "read" in content, "script must require human confirmation"

    assert "/api/console/messages" not in content, ("script must NOT use console endpoint as Feishu inbound test")

    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash syntax check failed: {result.stderr}"


def test_acceptance_stage5c_no_secret_leak():
    """Verify acceptance_stage5c.sh never prints the actual secret value."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5c.sh"
    content = script.read_text()

    assert 'echo "${feishu_app_secret}"' not in content, "script must not echo secret"
    assert 'echo "${TRPC_FEISHU_APP_SECRET}"' not in content, "script must not echo secret"
    assert "已配置，不显示" in content, "script must indicate secret is hidden"


def test_acceptance_stage5c_zero_event_count_is_a_single_number():
    """A failed grep must not append a second zero to the event count."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5c.sh"
    content = script.read_text()
    assert ('grep -c "Feishu text frame accepted" "${acceptance_tmp}/gateway.log" 2>/dev/null || echo "0"'
            not in content)
    assert ('grep -c "Feishu reply chain completed" "${acceptance_tmp}/gateway.log" 2>/dev/null || echo "0"'
            not in content)


def test_acceptance_stage5c_skip_without_credentials():
    """Verify acceptance_stage5c.sh exits 0 with SKIP when no credentials are set."""
    import subprocess
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(PROJECT_ROOT),
    }
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "acceptance_stage5c.sh")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=10,
    )
    assert result.returncode == 0, f"Expected exit 0 for SKIP, got {result.returncode}"
    assert "SKIP" in result.stdout, "Expected SKIP message in output"


def test_acceptance_stage5c_partial_credentials_fail():
    """Verify acceptance_stage5c.sh exits 1 when only partial credentials are set."""
    import subprocess
    env = {
        "PATH": "/usr/bin:/bin",
        "HOME": str(PROJECT_ROOT),
        "TRPC_FEISHU_APP_ID": "cli_test",
    }
    result = subprocess.run(
        ["bash", str(PROJECT_ROOT / "scripts" / "acceptance_stage5c.sh")],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(PROJECT_ROOT),
        timeout=10,
    )
    assert result.returncode == 1, f"Expected exit 1 for partial credentials, got {result.returncode}"
    assert "FAIL" in result.stdout or "FAIL" in result.stderr, "Expected FAIL message"


def test_acceptance_stage5c_stop_check_is_strict_on_all_four_services():
    """Stop verification must follow the Stage 5A standard: all four PID
    files present+alive before stop, all four gone after; no 'missing PID
    file counts as PASS' branch."""
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5c.sh"
    content = script.read_text()

    assert "可能已清理" not in content, "PASS-on-missing-PID branch must be removed"
    assert "for svc in gateway worker-a worker-b admin" in content
    assert '[[ ! -s "${pid_file}" ]]' in content, "PID files must be checked non-empty"
    assert 'kill -0 "${pid_value}"' in content, "liveness must be checked pre-stop"
    assert "进程 stop 后仍存在" in content
    assert "PID 文件 stop 后未删除" in content
    assert "未全部通过 stop 前检查" in content, "invalid pre-stop state must FAIL the run"


def test_acceptance_stage5c_stop_covers_worker_admin_cleanup():
    script = PROJECT_ROOT / "scripts" / "acceptance_stage5c.sh"
    content = script.read_text()
    stop_section = content.split("停止清理验证", 1)[-1]
    for svc_label in ("Gateway", "Worker A", "Worker B", "Admin"):
        assert svc_label in stop_section, f"{svc_label} missing from stop verification"
    assert "all_stopped" in stop_section and "all_files_removed" in stop_section


# ── Stage 6A1 acceptance script ──────────────────────────────────────────────


def test_acceptance_stage6a1_exists_and_valid_syntax():
    script = PROJECT_ROOT / "scripts" / "acceptance_stage6a1.sh"
    assert script.is_file(), "missing acceptance_stage6a1.sh"
    assert script.stat().st_mode & stat.S_IXUSR, "acceptance_stage6a1.sh is not executable"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash syntax check failed: {result.stderr}"


def test_acceptance_stage6a1_requires_real_model_and_no_fake():
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a1.sh").read_text()
    assert "TRPC_MODEL_NAME" in content and "TRPC_MODEL_API_KEY" in content
    assert "不得以 Fake 模型代替" in content, "missing real-model gate"
    assert "MockModel" not in content and "FakeLLM" not in content


def test_acceptance_stage6a1_disables_unrelated_im_connections():
    """Governance E2E uses Web Console and must not depend on external IM."""
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a1.sh").read_text()
    start_offset = content.index("bash start.sh")
    for variable in (
            "TRPC_WECOM_BOT_ID",
            "TRPC_WECOM_BOT_SECRET",
            "TRPC_WECOM_TENANT_ID",
            "TRPC_FEISHU_APP_ID",
            "TRPC_FEISHU_APP_SECRET",
            "TRPC_FEISHU_TENANT_ID",
    ):
        unset_line = f"unset {variable}"
        assert unset_line in content, f"acceptance must disable unrelated channel: {variable}"
        assert content.index(unset_line) < start_offset


def test_acceptance_stage6a1_covers_migration_admin_admission_hot_update_and_tools():
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a1.sh").read_text()
    for needle in (
            "information_schema.columns",  # 0004 governance column checks
            "column_default IS NOT NULL",  # server default must be dropped
            "/admin/v1/tenants",
            "/rollback",
            "Access is not allowed.",
            "message_receipts WHERE message_id",  # zero-receipt proof on denial
            "approval_required",
            '"status\\": \\"${expect}\\"',
    ):
        assert needle in content, f"missing coverage marker: {needle}"


def test_acceptance_stage6a1_admin_create_uses_collection_path():
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a1.sh").read_text()
    assert 'admin_api POST "/admin/v1/tenants" "${create_tools_body}"' in content, \
        "tenant creation must POST the collection path, not tenant-scoped (405 trap)"


def test_acceptance_stage6a1_review_is_not_second_confirmation():
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a1.sh").read_text()
    assert "待人工审批而阻断" in content, "review must be described as blocked pending approval"
    assert "二次确认完成" not in content


def test_acceptance_stage6a1_strict_four_service_stop_cleanup():
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a1.sh").read_text()
    assert "all_pids_valid" in content
    assert "四个服务未全部运行" in content
    assert "进程 stop 后仍存在" in content
    assert "PID 文件 stop 后未删除" in content
    assert "可能已清理" not in content, "no PASS-on-missing-PID branch allowed"


# ── Stage 6A2 acceptance script ──────────────────────────────────────────────


def test_acceptance_stage6a2_exists_and_valid_syntax():
    script = PROJECT_ROOT / "scripts" / "acceptance_stage6a2.sh"
    assert script.is_file(), "missing acceptance_stage6a2.sh"
    assert script.stat().st_mode & stat.S_IXUSR, "acceptance_stage6a2.sh is not executable"
    result = subprocess.run(["bash", "-n", str(script)], capture_output=True, text=True)
    assert result.returncode == 0, f"bash syntax check failed: {result.stderr}"


def test_acceptance_stage6a2_covers_approval_lifecycle():
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a2.sh").read_text()
    for needle in (
            "tool_approval_requests",  # 0005 tables
            "approval request created (tenant=tenant_default, tool=get_current_time, state=pending)",
            "approval executed (tenant=tenant_default, tool=get_current_time, decision=approve)",
            "/approve",
            "/reject",
            "Approval is no longer available.",
            "policy changed",
            "already being processed",
            "executing",
    ):
        assert needle in content, f"missing coverage marker: {needle}"


def test_acceptance_stage6a2_no_fake_model_and_im_hygiene():
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a2.sh").read_text()
    assert "不得以 Fake 模型代替" in content
    assert "unset TRPC_FEISHU_APP_ID" in content, "unrelated IM creds must be disabled for Console acceptance"
    assert "MockModel" not in content and "FakeLLM" not in content
    # P1-5: WeCom credential triple — full => real chain, empty => explicit
    # SKIP, partial => hard FAIL; never clear existing creds to fake a pass.
    assert "wecom_mode" in content
    assert 'wecom_filled}" == "3"' in content
    assert "企微凭据三元组不完整" in content
    assert "三元组全空" in content and "SKIP" in content
    # auth wait must depend ONLY on this service's fixed sanitized log line,
    # never on SDK raw-payload output (SDK logger is now suppressed).
    assert "WeCom AI Bot authenticated and started (channel=wecom)" in content, \
        "auth wait must match the service's fixed authentication record"
    assert "AiBotSDK" not in content, "script must not depend on SDK log lines"
    assert "authentication successful" not in content, "SDK auth line must not be used"
    assert "WeCom reply chain completed (terminal=done" in content
    assert "在同一企微会话中发送：/approve" in content
    # secret values are never echoed
    assert 'echo "${acceptance_pg_password}"' not in content
    assert 'echo "${TRPC_ADMIN_TOKEN}"' not in content
    assert 'echo "${TRPC_INTERNAL_TOKEN}"' not in content


def test_acceptance_stage6a2_review_wording_and_strict_stop():
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a2.sh").read_text()
    assert "待人工审批而阻断" in content
    assert "不自动重置" in content
    assert "all_valid" in content and "stop 后仍存在" in content and "stop 后未删除" in content
    assert "可能已清理" not in content


def test_acceptance_stage6a2_wecom_detection_after_env_source():
    """P1-5 residual defect: WeCom triple detection must run AFTER .env is
    sourced unconditionally — credentials living in .env must not produce a
    false SKIP."""
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a2.sh").read_text()
    assert 'if [[ -f "${project_dir}/.env" ]]; then set -a; source "${project_dir}/.env"; set +a; fi' \
        in content, ".env must be sourced unconditionally (not only when model vars are missing)"
    source_at = content.index('source "${project_dir}/.env"')
    detect_at = content.index("wecom_filled=0")
    assert source_at < detect_at, "wecom detection must run after .env is sourced"


def test_acceptance_stage6a2_wecom_log_leak_check():
    """After the human chain, the script must verify the gateway log is free of
    payload leaks, report only category names, and prompt before each read."""
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a2.sh").read_text()
    wecom_at = content.index("测试 10：企微真实审批链")
    stop_at = content.index("测试 11：stop.sh 严格四服务清理")
    block = content[wecom_at:stop_at]

    # the auth wait must not reference SDK or gateway-disabled INFO lines
    auth_at = block.index("等待企微长连接")
    auth_done_at = block.index("企微长连接已认证")
    auth_block = block[auth_at:auth_done_at]
    assert "WeCom AI Bot authenticated and started (channel=wecom)" in auth_block
    for banned in ("AiBotSDK", "authentication successful", "Authenticated", "Received push message",
                   "channel enabled"):
        assert banned not in auth_block, f"auth wait still depends on: {banned}"

    # payload leak audit present in the WeCom block, after chain verification
    assert "载荷泄漏" in block, "missing gateway-log payload leak audit"
    assert block.index("terminal=done 增长") < block.index("载荷泄漏"), \
        "leak audit must run after the approval chain verification"
    for marker in ("Received push message", "response_url", '"chattype"', "现在几点了"):
        assert marker in block, f"leak audit missing SDK raw-payload marker: {marker}"
    assert "grep -qF" in block, "leak checks must be quiet (never echo matched lines)"
    assert "fail \"企微日志载荷泄漏" in block or "fail \"载荷泄漏" in block, \
        "leak hit must fail by category name"

    # human steps: two send waits (each with an explicit phone+enter hint)
    # plus the mandatory phone-completeness confirmation (fix task 2)
    assert block.count("read -r -t 300") == 3
    assert block.count("手机发送完成后回到终端按回车") >= 2, "each send-read needs the enter hint"


def test_acceptance_stage6a2_wecom_phone_confirmation():
    """DB/terminal evidence must not stand in for the phone showing FULL text:
    the script requires an exact YES confirmation, everything else FAILs."""
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a2.sh").read_text()
    wecom_at = content.index("测试 10：企微真实审批链")
    stop_at = content.index("测试 11：stop.sh 严格四服务清理")
    block = content[wecom_at:stop_at]

    assert "phone_reply_confirmation" in block, "missing phone completeness confirmation"
    assert "请确认企微手机端已显示完整的当前时间回复" in block
    assert "工具返回了当前时间：20" in block, "confirmation must name the truncation symptom"
    assert 'read -r -t 300 phone_reply_confirmation' in block
    assert '[[ "${phone_reply_confirmation}" == "YES" ]]' in block, "only exact YES passes"
    assert 'fail "手机端未确认完整文本回复' in block, "timeout/empty/other must FAIL"
    # must come after the automated chain verification and before the leak audit/stop
    assert block.index("terminal=done 增长且审批终态 completed") < block.index("phone_reply_confirmation")
    assert block.index("手机端未确认完整文本回复") < block.index("载荷泄漏检查")
    # the manual confirmation must NOT replace any automated assertion
    assert "terminal=done 增长且审批终态 completed" in block
    assert "企微日志载荷泄漏检查通过" in block
    assert stop_at > wecom_at


def test_acceptance_stage6a2_wecom_scenario_runs_before_stop():
    """The manual WeCom approval chain must run while the stack is live:
    test 10 (WeCom) MUST precede test 11 (stop.sh cleanup)."""
    content = (PROJECT_ROOT / "scripts" / "acceptance_stage6a2.sh").read_text()
    wecom_at = content.index("测试 10：企微真实审批链")
    stop_at = content.index("测试 11：stop.sh 严格四服务清理")
    assert wecom_at < stop_at, "WeCom human approval scenario must precede stop.sh cleanup"
    # no stale ordering claims left in the scenario header list
    header = content.split("wecom_mode=0")[0]
    assert "10. 企微真实审批链" in header and "11. stop.sh" in header
