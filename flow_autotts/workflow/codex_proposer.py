"""Codex CLI wrapper for controller proposal rounds.

This mirrors the small proposer used by the AutoTTS reference workflow: render
one prompt, run ``codex exec`` in the repository, and persist enough logs for
the next round to diagnose what happened.
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class ProposerConfig:
    workdir: str
    prompt_path: str
    output_dir: str
    method_file: str
    history_dir: str
    context_file: str | None = None
    codex_bin: str = "codex"
    model: str | None = None
    exec_timeout_sec: float | None = None
    extra_codex_args: tuple[str, ...] = field(default_factory=tuple)
    exec_args: tuple[str, ...] = field(default_factory=tuple)
    plain_exec: bool = False


def load_prompt(prompt_file: str | Path, **kwargs: str) -> str:
    path = Path(prompt_file).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"prompt file not found: {path}")
    text = path.read_text(encoding="utf-8")
    for key, value in kwargs.items():
        text = text.replace("{" + key + "}", str(value))
    return text


_VERIFIED_CODEX_PATHS: set[str] = set()


def resolve_codex_executable(codex_bin: str) -> str:
    if codex_bin == "codex":
        failures: list[str] = []
        for candidate in _codex_candidates_from_environment():
            try:
                assert_openai_codex_cli(candidate)
            except RuntimeError as exc:
                failures.append(f"{candidate}: {exc}")
                continue
            return candidate
        if failures:
            raise RuntimeError(
                "Found `codex` candidates, but none could run `codex exec --help`.\n"
                + "\n\n".join(failures)
                + "\n\nSet CODEX_BIN to a working OpenAI Codex CLI binary."
            )
        raise RuntimeError("Cannot find 'codex' in PATH. Set CODEX_BIN explicitly.")
    else:
        path = Path(codex_bin).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Codex binary not found: {codex_bin}")
        codex_path = str(path.resolve())
    assert_openai_codex_cli(codex_path)
    return codex_path


def _codex_candidates_from_environment() -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: str | Path | None) -> None:
        if path is None:
            return
        resolved = Path(path).expanduser()
        try:
            resolved = resolved.resolve()
        except OSError:
            return
        key = str(resolved)
        if key in seen or not resolved.is_file():
            return
        seen.add(key)
        candidates.append(key)

    for path_dir in os.environ.get("PATH", "").split(os.pathsep):
        if path_dir:
            add(Path(path_dir) / "codex")

    for root in _candidate_extension_roots():
        for path in sorted(root.glob("openai.chatgpt-*/bin/*/codex"), reverse=True):
            add(path)

    resolved = shutil.which("codex")
    add(resolved)
    return candidates


def _candidate_extension_roots() -> list[Path]:
    home = Path.home()
    roots = [
        home / ".vscode-server" / "extensions",
        home / ".vscode" / "extensions",
        home / ".cursor-server" / "extensions",
        home / ".cursor" / "extensions",
    ]
    if sys.platform == "darwin":
        roots.extend(
            [
                home / "Library" / "Application Support" / "Code" / "User" / "globalStorage",
                home / "Library" / "Application Support" / "Cursor" / "User" / "globalStorage",
            ]
        )
    return [root for root in roots if root.is_dir()]


def assert_openai_codex_cli(codex_path: str) -> None:
    if codex_path in _VERIFIED_CODEX_PATHS:
        return
    try:
        completed = subprocess.run(
            [codex_path, "exec", "--help"],
            capture_output=True,
            text=True,
            timeout=15,
            env=os.environ.copy(),
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"Codex binary is not executable: {codex_path!r}") from exc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{codex_path!r} exec --help timed out.") from exc

    blob = ((completed.stdout or "") + (completed.stderr or "")).lower()
    wrong_markers = (
        "granian",
        "django",
        "migration",
        "librariandaemon",
        "listening at: http",
        ":9810",
        "broadcastlistener",
    )
    if any(marker in blob for marker in wrong_markers):
        raise RuntimeError(
            "The `codex` found on PATH does not look like the OpenAI Codex CLI. "
            f"Resolved path: {codex_path}. Set CODEX_BIN to the OpenAI CLI."
        )
    if completed.returncode == 0 or any(
        marker in blob
        for marker in ("--json", "--full-auto", "skip-git", "workspace-write", "approval")
    ):
        _VERIFIED_CODEX_PATHS.add(codex_path)
        return
    if "syntaxerror" in blob and "unexpected reserved word" in blob:
        node_version = _node_version()
        raise RuntimeError(
            f"{codex_path!r} looks like the npm Codex CLI, but it failed under "
            f"node {node_version or 'unknown'} before startup. Install/use Node >= 18, "
            "or set CODEX_BIN to the standalone Codex binary from the ChatGPT/VS Code extension."
        )
    raise RuntimeError(
        f"{codex_path!r} exec --help exited with {completed.returncode} and did not "
        f"look like the OpenAI Codex CLI. Output: {blob[:500]!r}"
    )


def _node_version() -> str | None:
    try:
        completed = subprocess.run(
            ["node", "--version"],
            capture_output=True,
            text=True,
            timeout=5,
            env=os.environ.copy(),
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return (completed.stdout or completed.stderr or "").strip() or None


def build_codex_command(
    config: ProposerConfig,
    prompt: str,
    last_message_path: str | None,
) -> list[str]:
    repo_dir = Path(config.workdir).expanduser().resolve()
    if not repo_dir.is_dir():
        raise FileNotFoundError(f"workdir is not a directory: {repo_dir}")
    command = [
        resolve_codex_executable(config.codex_bin),
        *config.extra_codex_args,
        "exec",
        *config.exec_args,
        "--skip-git-repo-check",
        "-C",
        str(repo_dir),
    ]
    if not config.plain_exec:
        command.append("--json")
        if not _exec_args_bypass_approvals_and_sandbox(config.exec_args):
            command.append("--full-auto")
    if config.model:
        command.extend(["-m", config.model])
    if last_message_path:
        command.extend(["-o", last_message_path])
    command.append(prompt)
    return command


def _exec_args_bypass_approvals_and_sandbox(exec_args: tuple[str, ...]) -> bool:
    return "--dangerously-bypass-approvals-and-sandbox" in exec_args


async def run_codex(
    config: ProposerConfig,
    prompt: str,
    stdout_path: str | Path,
    stderr_path: str | Path,
    last_message_path: str | Path | None = None,
) -> dict[str, Any]:
    command = build_codex_command(
        config=config,
        prompt=prompt,
        last_message_path=str(last_message_path) if last_message_path else None,
    )
    env = os.environ.copy()
    proc = await asyncio.create_subprocess_exec(
        *command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
        cwd=str(Path(config.workdir).expanduser().resolve()),
        env=env,
    )
    chunks: list[str] = []
    timed_out = False

    async def collect() -> None:
        assert proc.stdout is not None
        while True:
            chunk = await proc.stdout.read(4096)
            if not chunk:
                break
            text = chunk.decode("utf-8", errors="replace")
            chunks.append(text)
            print(text, end="", flush=True)
        await proc.wait()

    try:
        if config.exec_timeout_sec is None:
            await collect()
        else:
            await asyncio.wait_for(collect(), timeout=config.exec_timeout_sec)
    except asyncio.TimeoutError:
        timed_out = True
        proc.kill()
        try:
            await asyncio.wait_for(collect(), timeout=60.0)
        except asyncio.TimeoutError:
            pass

    stdout_text = "".join(chunks)
    stderr_text = "[codex_proposer] stderr merged into stdout.\n"
    if timed_out:
        stderr_text += (
            f"[codex_proposer] exec exceeded exec_timeout_sec={config.exec_timeout_sec}; "
            "subprocess was killed.\n"
        )

    stdout_target = Path(stdout_path)
    stdout_target.parent.mkdir(parents=True, exist_ok=True)
    stdout_target.write_text(stdout_text, encoding="utf-8")
    Path(stderr_path).write_text(stderr_text, encoding="utf-8")
    return {
        "returncode": proc.returncode,
        "command": command,
        "stdout": stdout_text,
        "stderr": stderr_text,
        "timed_out": timed_out,
    }


def parse_codex_jsonl(stdout_text: str) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    final_agent_message = None
    turn_failed = False
    for line in stdout_text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            events.append({"type": "unparsed_line", "raw": line})
            continue
        events.append(event)
        event_type = event.get("type")
        if event_type == "turn.failed":
            turn_failed = True
            errors.append(event)
        if event_type == "error":
            errors.append(event)
        if event_type == "item.completed":
            item = event.get("item", {})
            if item.get("type") == "agent_message":
                final_agent_message = item.get("text")
    return {
        "events": events,
        "num_events": len(events),
        "final_agent_message": final_agent_message,
        "turn_failed": turn_failed,
        "error_events": errors,
    }


async def propose(config: ProposerConfig) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prompt = load_prompt(
        config.prompt_path,
        method_file=config.method_file,
        history_dir=config.history_dir,
        context_file=config.context_file or "",
        workdir=config.workdir,
    )
    rendered_prompt_path = output_dir / "rendered_prompt.txt"
    rendered_prompt_path.write_text(prompt, encoding="utf-8")

    stdout_path = output_dir / "codex_stdout.jsonl"
    stderr_path = output_dir / "codex_stderr.txt"
    last_message_path = output_dir / "codex_last_message.txt"
    run_result = await run_codex(
        config=config,
        prompt=prompt,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
        last_message_path=last_message_path,
    )
    parsed = parse_codex_jsonl(run_result["stdout"])

    final_text = parsed["final_agent_message"]
    if not final_text and last_message_path.exists():
        final_text = last_message_path.read_text(encoding="utf-8", errors="replace").strip()
    if not final_text and config.plain_exec:
        final_text = run_result["stdout"].strip()

    if run_result.get("timed_out"):
        status = "timeout"
    elif run_result["returncode"] != 0 or parsed["turn_failed"]:
        status = "failed"
    else:
        status = "ok"

    result = {
        "config": asdict(config),
        "command": run_result["command"],
        "returncode": run_result["returncode"],
        "timed_out": run_result.get("timed_out", False),
        "rendered_prompt_path": str(rendered_prompt_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "last_message_path": str(last_message_path),
        "num_events": parsed["num_events"],
        "turn_failed": parsed["turn_failed"],
        "final_result": final_text,
        "error_events": parsed["error_events"],
        "status": status,
    }
    (output_dir / "proposal_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def extra_codex_args_from_env() -> tuple[str, ...]:
    raw = os.environ.get("CODEX_EXTRA_ARGS", "").strip()
    return tuple(shlex.split(raw)) if raw else ()


def exec_codex_args_from_env() -> tuple[str, ...]:
    raw = os.environ.get("CODEX_EXEC_ARGS", "").strip()
    return tuple(shlex.split(raw)) if raw else ()


def optional_float_env(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return None
    return float(raw)


def truthy_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
