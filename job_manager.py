"""Interactive job queue manager for Windows-focused ML workflows."""
from __future__ import annotations

import argparse
import enum
import logging
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple


LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_PATH = Path("job_manager.log")


class Notifier:
    """Send notifications when jobs complete or fail."""

    def __init__(self) -> None:
        self._toast = None
        if sys.platform.startswith("win"):
            try:
                from win10toast import ToastNotifier  # type: ignore

                self._toast = ToastNotifier()
            except Exception:
                self._toast = None

    def notify(self, title: str, message: str) -> None:
        if self._toast is not None:
            try:
                self._toast.show_toast(title, message, duration=5, threaded=True)
                return
            except Exception:
                pass
        print(f"[NOTIFY] {title}: {message}")


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )


@dataclass
class Job:
    command: List[str]
    name: str = field(default_factory=lambda: "Job")
    cwd: Optional[Path] = None
    env: Optional[Dict[str, str]] = None
    venv: Optional[Path] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)

    def __str__(self) -> str:
        cmd = " ".join(shlex.quote(part) for part in self.command)
        details = f"{self.name} ({self.id}) -> {cmd}"
        if self.venv:
            details += f" [venv={self.venv}]"
        return details


class JobStatus(enum.Enum):
    SUCCESS = "success"
    FAILED = "failed"
    ERROR = "error"


@dataclass
class JobResult:
    job: Job
    status: JobStatus
    finished_at: float
    return_code: Optional[int] = None
    log_file: Optional[Path] = None
    error_message: Optional[str] = None


def prepare_command(job: Job) -> Tuple[List[str], Dict[str, str]]:
    """Prepare the command and environment for execution.

    When a virtual environment is specified, the PATH and VIRTUAL_ENV variables
    are adjusted so the job runs inside the desired environment. If the command
    starts with ``python`` it is replaced with the interpreter from the
    environment for clarity.
    """

    command = list(job.command)
    env = os.environ.copy()
    if job.env:
        env.update(job.env)

    if job.venv:
        scripts_dir = job.venv / ("Scripts" if os.name == "nt" else "bin")
        python_name = "python.exe" if os.name == "nt" else "python"
        python_executable = scripts_dir / python_name
        env["VIRTUAL_ENV"] = str(job.venv)
        existing_path = env.get("PATH", "")
        env["PATH"] = str(scripts_dir) + (os.pathsep + existing_path if existing_path else "")
        if command and Path(command[0]).name in {"python", "python.exe"}:
            command[0] = str(python_executable)

    return command, env


class JobRunner(threading.Thread):
    def __init__(
        self,
        job_queue: "queue.Queue[Job]",
        notifier: Notifier,
        on_job_finished: "callable[[JobResult], None]",
    ):
        super().__init__(daemon=True)
        self.job_queue = job_queue
        self.notifier = notifier
        self.on_job_finished = on_job_finished
        self._stop_event = threading.Event()
        self.current_job: Optional[Job] = None

    def stop(self) -> None:
        self._stop_event.set()

    def run(self) -> None:
        while not self._stop_event.is_set():
            try:
                job = self.job_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                self.current_job = job
                self.execute(job)
            finally:
                self.current_job = None
                self.job_queue.task_done()

    def execute(self, job: Job) -> None:
        log_file = LOG_DIR / f"{job.id}.log"
        logging.info("Starting job %s", job)
        creationflags = 0
        if sys.platform.startswith("win"):
            creationflags = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]
        try:
            command, env = prepare_command(job)
            with log_file.open("w", encoding="utf-8", errors="replace") as log_fp:
                log_fp.write(f"Job: {job}\n")
                log_fp.write("=" * 80 + "\n")
                log_fp.flush()
                process = subprocess.Popen(
                    command,
                    cwd=str(job.cwd) if job.cwd else None,
                    env=env,
                    stdout=log_fp,
                    stderr=subprocess.STDOUT,
                    creationflags=creationflags,
                    text=True,
                )
                return_code = process.wait()
        except OSError as exc:
            message = f"Job '{job.name}' could not start: {exc}"
            logging.exception(message)
            self.notifier.notify("Job error", message)
            result = JobResult(
                job=job,
                status=JobStatus.ERROR,
                finished_at=time.time(),
                return_code=None,
                log_file=log_file,
                error_message=str(exc),
            )
            self.on_job_finished(result)
            return

        finished_at = time.time()
        if return_code == 0:
            message = f"Job '{job.name}' completed successfully."
            logging.info(message)
            self.notifier.notify("Job completed", message)
            result = JobResult(
                job=job,
                status=JobStatus.SUCCESS,
                finished_at=finished_at,
                return_code=return_code,
                log_file=log_file,
            )
        else:
            message = f"Job '{job.name}' failed with exit code {return_code}. See {log_file}."
            logging.error(message)
            self.notifier.notify("Job failed", message)
            result = JobResult(
                job=job,
                status=JobStatus.FAILED,
                finished_at=finished_at,
                return_code=return_code,
                log_file=log_file,
            )
        self.on_job_finished(result)


class JobManagerApp:
    def __init__(self) -> None:
        self.notifier = Notifier()
        self.job_queue: "queue.Queue[Job]" = queue.Queue()
        self.history: List[JobResult] = []
        self._history_lock = threading.Lock()
        self.runner = JobRunner(self.job_queue, self.notifier, self._on_job_finished)
        self.runner.start()

    def _on_job_finished(self, result: JobResult) -> None:
        with self._history_lock:
            self.history.append(result)

    def add_job(
        self,
        command_line: str,
        name: Optional[str] = None,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        venv: Optional[str] = None,
    ) -> Job:
        command = shlex.split(command_line)
        if not command:
            raise ValueError("Command cannot be empty")
        job = Job(
            command=command,
            name=name or command[0],
            cwd=Path(cwd) if cwd else None,
            env=env,
            venv=Path(venv) if venv else None,
        )
        self.job_queue.put(job)
        logging.info("Queued job %s", job)
        return job

    def list_jobs(self) -> List[str]:
        pending: List[str] = []
        with self.job_queue.mutex:
            for job in list(self.job_queue.queue):
                pending.append(str(job))
        if self.runner.current_job is not None:
            pending.insert(0, f"[RUNNING] {self.runner.current_job}")
        return pending

    def stop(self) -> None:
        self.runner.stop()
        self.runner.join(timeout=1.0)

    def list_history(self, limit: Optional[int] = None) -> List[str]:
        with self._history_lock:
            items = list(self.history[-limit:] if limit else self.history)
        formatted: List[str] = []
        for result in items:
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(result.finished_at))
            details = (
                f"[{timestamp}] {result.job.name} ({result.job.id}) -> {result.status.value}"
                f" (code={result.return_code}, log={result.log_file})"
            )
            if result.error_message:
                details += f" error={result.error_message}"
            formatted.append(details)
        return formatted


class InteractiveShell(threading.Thread):
    prompt = "(job-manager) "

    def __init__(self, app: JobManagerApp) -> None:
        super().__init__(daemon=True)
        self.app = app
        self._stop = threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                raw = input(self.prompt)
            except (EOFError, KeyboardInterrupt):
                print()
                self.app.stop()
                break
            if not raw:
                continue
            self.handle_command(raw)

    def handle_command(self, raw: str) -> None:
        parts = shlex.split(raw)
        if not parts:
            return
        command, *rest = parts
        if command in {"quit", "exit"}:
            self.app.stop()
            self._stop.set()
            return
        if command == "add":
            parser = argparse.ArgumentParser(
                prog="add",
                description="Add a job to the queue",
                add_help=True,
                allow_abbrev=False,
            )
            parser.add_argument("-n", "--name", dest="name")
            parser.add_argument("-c", "--cwd", dest="cwd")
            parser.add_argument("-e", "--env", action="append", default=[], help="Environment variable KEY=VALUE")
            parser.add_argument("--venv", dest="venv", help="Path to a virtual environment to use")
            try:
                ns, command_tokens = parser.parse_known_args(rest)
            except SystemExit:
                return
            if not command_tokens:
                print("No command provided. Example: add -n train python train.py")
                return
            env: Dict[str, str] = {}
            for item in ns.env:
                if "=" not in item:
                    print(f"Invalid env assignment: {item}. Use KEY=VALUE.")
                    return
                key, value = item.split("=", 1)
                env[key] = value
            cmd = " ".join(command_tokens)
            job = self.app.add_job(
                cmd,
                name=ns.name,
                cwd=ns.cwd,
                env=env or None,
                venv=ns.venv,
            )
            print(f"Queued: {job}")
            return
        if command == "list":
            jobs = self.app.list_jobs()
            if not jobs:
                print("No pending jobs.")
            else:
                for item in jobs:
                    print(item)
            return
        if command == "history":
            parser = argparse.ArgumentParser(prog="history", description="Show completed jobs", allow_abbrev=False)
            parser.add_argument("-n", "--limit", type=int, default=None)
            try:
                ns = parser.parse_args(rest)
            except SystemExit:
                return
            for entry in self.app.list_history(limit=ns.limit):
                print(entry)
            return
        print(f"Unknown command: {command}")

    def stop(self) -> None:
        self._stop.set()


def main() -> None:
    configure_logging()
    app = JobManagerApp()
    shell = InteractiveShell(app)
    shell.start()
    logging.info("Job manager started. Type 'add', 'list', 'exit'.")
    try:
        while shell.is_alive():
            shell.join(timeout=0.5)
    except KeyboardInterrupt:
        logging.info("Stopping job manager...")
        app.stop()


if __name__ == "__main__":
    main()
