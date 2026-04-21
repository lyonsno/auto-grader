from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

from auto_grader.narrator_run import load_recorded_messages


def replay_recorded_messages(
    messages: list[dict[str, Any]],
    *,
    dispatch: Callable[[dict[str, Any]], None],
    speed: float = 1.0,
    from_event: int = 0,
    sleep_fn: Callable[[float], None] = time.sleep,
) -> int:
    if speed <= 0:
        raise ValueError("speed must be > 0")
    if from_event < 0:
        raise ValueError("from_event must be >= 0")

    dispatched = 0
    previous_ts: float | None = None
    for index, msg in enumerate(messages[from_event:], start=from_event):
        msg_ts = msg.get("ts")
        current_ts = float(msg_ts) if isinstance(msg_ts, (int, float)) else None
        if index > from_event and previous_ts is not None and current_ts is not None:
            delay = max(0.0, (current_ts - previous_ts) / speed)
            sleep_fn(delay)
        dispatch(msg)
        dispatched += 1
        previous_ts = current_ts
    return dispatched


def _resolve_narrator_path(run_arg: str) -> Path:
    path = Path(run_arg)
    if path.name == "narrator.jsonl":
        return path
    return path / "narrator.jsonl"


def _open_fifo_writer_with_timeout(
    fifo_path: Path,
    *,
    timeout_s: float = 5.0,
    poll_s: float = 0.05,
):
    deadline = time.monotonic() + timeout_s
    last_error: OSError | None = None
    while time.monotonic() < deadline:
        try:
            fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
            os.set_blocking(fd, True)
            return os.fdopen(fd, "w", buffering=1)
        except OSError as exc:
            last_error = exc
            if exc.errno not in {errno.ENXIO, errno.ENOENT}:  # type: ignore[name-defined]
                raise RuntimeError(
                    f"could not open replay fifo writer at {fifo_path}: {exc}"
                ) from exc
            time.sleep(poll_s)
    detail = f"{last_error}" if last_error is not None else "reader never attached"
    raise RuntimeError(
        f"narrator reader did not connect to replay fifo within {timeout_s:.1f}s: {detail}"
    )


def replay_run(
    narrator_path: Path,
    *,
    speed: float = 1.0,
    from_event: int = 0,
) -> int:
    messages = load_recorded_messages(narrator_path)
    tmpdir = Path(tempfile.mkdtemp(prefix="paint-dry-replay-"))
    fifo_path = tmpdir / "narrator.fifo"
    reader_script = Path(__file__).resolve().parent / "narrator_reader.py"

    os.mkfifo(fifo_path)
    process = subprocess.Popen(
        [sys.executable, str(reader_script), str(fifo_path)],
        cwd=Path(__file__).resolve().parent.parent,
    )
    try:
        with _open_fifo_writer_with_timeout(fifo_path) as writer:
            replay_recorded_messages(
                messages,
                dispatch=lambda msg: _write_message(writer, msg),
                speed=speed,
                from_event=from_event,
            )
        return process.wait()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        try:
            fifo_path.unlink(missing_ok=True)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


def _write_message(writer, msg: dict[str, Any]) -> None:
    writer.write(json.dumps(msg) + "\n")
    writer.flush()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Replay a saved Project Paint Dry narrator.jsonl run through the "
            "real narrator reader surface."
        )
    )
    parser.add_argument("run", help="Run directory or narrator.jsonl path")
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Playback multiplier (default: 1.0 for original wallclock pacing).",
    )
    parser.add_argument(
        "--from-event",
        type=int,
        default=0,
        help="Zero-based event index to start replay from.",
    )
    args = parser.parse_args(argv)

    narrator_path = _resolve_narrator_path(args.run)
    if not narrator_path.exists():
        print(f"narrator log not found: {narrator_path}", file=sys.stderr)
        return 2
    return replay_run(
        narrator_path,
        speed=args.speed,
        from_event=args.from_event,
    )


if __name__ == "__main__":
    raise SystemExit(main())
