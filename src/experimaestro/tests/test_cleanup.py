"""Tests for workspace cleanup, in particular that active job events are not deleted."""

import json
import os
import socket
import stat
import subprocess
from pathlib import Path

import pytest

from experimaestro.locking import (
    LOCK_MODE_ENV,
    LOCK_MODE_INHERIT,
    LOCK_MODE_UMASK,
    create_file_lock,
    parse_lock_mode,
    register_workspace_lock_mode,
)
from experimaestro.locking import _workspace_lock_modes
from experimaestro.scheduler.cleanup import (
    _check_orphaned_job_events,
    _cleanup_stale_pid_files,
    _is_job_active,
)
from experimaestro.scheduler.state_status import task_id_hash


TASK_ID = "my.module.MyTask"
JOB_ID = "abc123def456" + "0" * 52  # 64 hex chars
SCRIPTNAME = "MyTask"


def _clear_workspace_lock_modes():
    """Forget the workspaces registered by a test"""
    _workspace_lock_modes.clear()


def _create_event_file(workspace: Path, task_id: str, job_id: str) -> Path:
    """Create a fake event file in .events/jobs/ (flat format)"""
    events_dir = workspace / ".events" / "jobs"
    events_dir.mkdir(parents=True, exist_ok=True)
    h = task_id_hash(task_id)
    event_file = events_dir / f"{h}-{job_id}-0.jsonl"
    event_file.write_text(
        json.dumps({"type": "job_state_changed", "job_id": job_id, "state": "running"})
        + "\n"
    )
    return event_file


def _create_job_dir(workspace: Path, task_id: str, job_id: str) -> Path:
    """Create a job directory with .experimaestro subdirectory."""
    job_path = workspace / "jobs" / task_id / job_id
    xpm_dir = job_path / ".experimaestro"
    xpm_dir.mkdir(parents=True, exist_ok=True)
    return job_path


def _write_status_json(job_path: Path, *, events_count: int | None = None):
    """Write a minimal status.json."""
    status = {
        "job_id": JOB_ID,
        "task_id": TASK_ID,
        "state": "RUNNING",
        "path": str(job_path),
    }
    if events_count is not None:
        status["events_count"] = events_count
    status_path = job_path / ".experimaestro" / "status.json"
    status_path.write_text(json.dumps(status))


def _write_pid_file(job_path: Path, pid: int, **extra):
    """Write a PID file simulating a running job process."""
    pid_path = job_path / f"{SCRIPTNAME}.pid"
    pid_path.write_text(json.dumps({"type": "local", "pid": pid, **extra}))
    return pid_path


def _dead_pid() -> int:
    """Return the PID of a process that has terminated."""
    process = subprocess.Popen(["true"])
    process.wait()
    return process.pid


def _write_done_marker(job_path: Path):
    """Write a .done marker file."""
    done_path = job_path / f"{SCRIPTNAME}.done"
    done_path.write_text("")


def _write_failed_marker(job_path: Path):
    """Write a .failed marker file."""
    failed_path = job_path / f"{SCRIPTNAME}.failed"
    failed_path.write_text(json.dumps({"reason": "FAILED"}))


class TestCleanupSkipsActiveJobs:
    """Tests that _check_orphaned_job_events does not delete event files
    for jobs that are currently active (running or starting)."""

    def test_skips_when_lock_held(self, tmp_path: Path):
        """Event files should NOT be deleted when the job lock is held."""
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)

        # Hold the lock (simulating a running job process)
        lock_path = job_path / ".experimaestro" / f"{SCRIPTNAME}.lock"
        lock = create_file_lock(lock_path)
        with lock:
            _check_orphaned_job_events(workspace, auto_fix=True)

        assert event_file.exists(), "Event file was deleted while job lock was held"

    def test_skips_when_pid_running(self, tmp_path: Path):
        """Event files should NOT be deleted when PID file shows a running process."""
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        # Use current process PID (which is definitely running)
        _write_pid_file(job_path, os.getpid())

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert event_file.exists(), (
            "Event file was deleted while job process is running"
        )

    def test_skips_when_no_terminal_markers(self, tmp_path: Path):
        """Event files should NOT be deleted when no .done/.failed markers exist.

        This handles the gap between the scheduler releasing the job lock
        and the job process acquiring it (steps 4-5 of the lifecycle).
        """
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        # No .done, no .failed, no lock, no PID — job may be starting

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert event_file.exists(), (
            "Event file was deleted while job has no terminal markers (may be starting)"
        )

    def test_skips_when_job_dir_missing(self, tmp_path: Path):
        """Event files should NOT be deleted when the job directory doesn't exist.

        The job directory may not yet exist during very early startup.
        """
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        # No job directory created

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert event_file.exists(), (
            "Event file was deleted while job directory doesn't exist (may be starting)"
        )

    def test_cleans_when_job_done(self, tmp_path: Path):
        """Event files SHOULD be deleted when the job is finished (.done exists)."""
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        _write_done_marker(job_path)

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert not event_file.exists(), (
            "Event file should have been deleted for a finished job"
        )

    def test_cleans_when_job_failed(self, tmp_path: Path):
        """Event files SHOULD be deleted when the job has failed (.failed exists)."""
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        _write_failed_marker(job_path)

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert not event_file.exists(), (
            "Event file should have been deleted for a failed job"
        )


class TestCleanupDoesNotTouchForeignJobs:
    """A monitor must never declare dead a job it cannot see: another host,
    another user, or a launcher unavailable here (issue #270)."""

    def test_other_host_pid_is_active(self, tmp_path: Path):
        """A PID recorded on another host cannot be checked here."""
        job_path = _create_job_dir(tmp_path, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        _write_pid_file(job_path, _dead_pid(), host="some-other-host")

        assert _is_job_active(job_path, TASK_ID), (
            "A job whose PID belongs to another host must be considered active"
        )
        assert not (job_path / f"{SCRIPTNAME}.failed").exists(), (
            "A .failed marker was written for a job running on another host"
        )
        assert (job_path / f"{SCRIPTNAME}.pid").exists(), (
            "The PID file of a job running on another host was removed"
        )

    def test_unknown_process_type_is_active(self, tmp_path: Path):
        """An unknown process type means the launcher is not available here."""
        job_path = _create_job_dir(tmp_path, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        pid_path = job_path / f"{SCRIPTNAME}.pid"
        pid_path.write_text(json.dumps({"type": "unknown-launcher", "pid": 1}))

        assert _is_job_active(job_path, TASK_ID)
        assert not (job_path / f"{SCRIPTNAME}.failed").exists()
        assert pid_path.exists()

    def test_other_host_events_are_kept(self, tmp_path: Path):
        """Event files of a job running on another host must be kept."""
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        _write_pid_file(job_path, _dead_pid(), host="some-other-host")

        _check_orphaned_job_events(workspace, auto_fix=True)

        assert event_file.exists()
        assert not (job_path / f"{SCRIPTNAME}.failed").exists()

    def test_stale_pid_cleanup_skips_other_host(self, tmp_path: Path):
        """_cleanup_stale_pid_files must not touch jobs of another host."""
        job_path = _create_job_dir(tmp_path, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        pid_path = _write_pid_file(job_path, _dead_pid(), host="some-other-host")

        assert _cleanup_stale_pid_files(tmp_path) == 0
        assert pid_path.exists()
        assert not (job_path / f"{SCRIPTNAME}.failed").exists()

    def test_stale_pid_cleanup_on_same_host(self, tmp_path: Path):
        """A dead process of this host is still cleaned up."""
        job_path = _create_job_dir(tmp_path, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        pid_path = _write_pid_file(job_path, _dead_pid(), host=socket.gethostname())

        assert _cleanup_stale_pid_files(tmp_path) == 1
        assert not pid_path.exists()
        assert (job_path / f"{SCRIPTNAME}.failed").exists()

    def test_permission_error_is_skipped(self, tmp_path: Path, monkeypatch):
        """An unreadable lock file (owned by another user) is skipped."""
        workspace = tmp_path
        event_file = _create_event_file(workspace, TASK_ID, JOB_ID)
        job_path = _create_job_dir(workspace, TASK_ID, JOB_ID)
        _write_status_json(job_path)
        _write_done_marker(job_path)

        def raise_permission_error(*args, **kwargs):
            raise PermissionError(13, "Permission denied")

        monkeypatch.setattr(
            "experimaestro.locking.create_file_lock", raise_permission_error
        )

        # Must not raise
        _check_orphaned_job_events(workspace, auto_fix=True)
        assert event_file.exists()


class TestSharedLockPermissions:
    """Lock files must stay usable by the whole group in a shared workspace."""

    @staticmethod
    def _lock_mode(lock_path: Path, umask: int = 0o022) -> int:
        old_umask = os.umask(umask)
        try:
            with create_file_lock(lock_path):
                return lock_path.stat().st_mode & 0o777
        finally:
            os.umask(old_umask)

    @pytest.fixture
    def workspace(self, tmp_path: Path):
        """A group-writable (shared) workspace directory"""
        workspace = tmp_path / "shared"
        (workspace / "jobs").mkdir(parents=True)
        workspace.chmod(0o2775)
        yield workspace
        _clear_workspace_lock_modes()

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_inherits_workspace_permissions(self, workspace: Path):
        """A registered group-writable workspace gives group-writable locks."""
        register_workspace_lock_mode(workspace, LOCK_MODE_INHERIT)

        mode = self._lock_mode(workspace / "jobs" / "test.lock")

        assert mode & stat.S_IWGRP, (
            "Lock file of a group-writable workspace must be group-writable"
        )

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_unregistered_path_uses_umask(self, tmp_path: Path, workspace: Path):
        """Outside any registered workspace, the umask decides (as before)."""
        register_workspace_lock_mode(workspace, LOCK_MODE_INHERIT)

        mode = self._lock_mode(tmp_path / "elsewhere.lock")

        assert not (mode & stat.S_IWGRP), (
            "Lock file outside a workspace must follow the umask"
        )

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_umask_mode(self, workspace: Path):
        """An explicit `umask` mode keeps the historical behaviour."""
        register_workspace_lock_mode(workspace, LOCK_MODE_UMASK)

        mode = self._lock_mode(workspace / "jobs" / "test.lock")

        assert mode == 0o644

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_explicit_mode(self, workspace: Path):
        """An octal mode from the settings is used as is."""
        register_workspace_lock_mode(workspace, "0666")

        assert self._lock_mode(workspace / "jobs" / "test.lock") == 0o666

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_environment_override(self, workspace: Path, monkeypatch):
        """XPM_LOCK_MODE overrides the workspace setting."""
        register_workspace_lock_mode(workspace, LOCK_MODE_UMASK)
        monkeypatch.setenv(LOCK_MODE_ENV, "0664")

        assert self._lock_mode(workspace / "jobs" / "test.lock") == 0o664

    @pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
    def test_innermost_workspace_wins(self, workspace: Path):
        """Nested workspaces: the innermost registered one is used."""
        nested = workspace / "jobs" / "nested"
        nested.mkdir()
        nested.chmod(0o755)
        register_workspace_lock_mode(workspace, LOCK_MODE_INHERIT)
        register_workspace_lock_mode(nested, LOCK_MODE_INHERIT)

        assert not (self._lock_mode(nested / "test.lock") & stat.S_IWGRP)
        assert self._lock_mode(workspace / "jobs" / "test.lock") & stat.S_IWGRP

    def test_invalid_mode_falls_back(self, workspace: Path):
        """An invalid specification falls back to inheriting."""
        assert parse_lock_mode("not-a-mode") == LOCK_MODE_INHERIT
        assert parse_lock_mode("0664") == 0o664
        assert parse_lock_mode(None) == LOCK_MODE_INHERIT
