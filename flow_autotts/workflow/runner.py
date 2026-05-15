"""AutoTTS-style iterative propose/evaluate/archive workflow."""

from __future__ import annotations

import asyncio
import json
import os
import re
import shlex
import shutil
import subprocess
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from flow_autotts.workflow.codex_proposer import (
    ProposerConfig,
    exec_codex_args_from_env,
    extra_codex_args_from_env,
    optional_float_env,
    propose,
    truthy_env,
)


@dataclass
class WorkflowConfig:
    workdir: Path
    method_file: str
    history_dir: str
    prompt_path: Path
    rounds: int
    codex_log_parent: Path
    result_dir: Path | None
    eval_cmd: tuple[str, ...]
    eval_cwd: Path | None = None
    eval_timeout_sec: float = 7200.0
    resume: bool = False
    template_file: str | None = None
    context_history_rounds: int = 5


_ROUND_ID_RE = re.compile(r"^r(\d{4})_(\d{8}_\d{6})_([0-9a-f]{8})$")


def parse_round_id(round_id: str) -> tuple[int, str, str] | None:
    match = _ROUND_ID_RE.match(round_id.strip())
    if not match:
        return None
    return int(match.group(1)), match.group(2), match.group(3)


def scan_history_resume(workdir: Path, history_dir: str) -> tuple[str | None, str | None, int]:
    base = workdir / history_dir
    if not base.is_dir():
        return None, None, 0
    candidates: list[tuple[Path, int, str, str, float]] = []
    for path in base.iterdir():
        if not path.is_dir():
            continue
        parsed = parse_round_id(path.name)
        if parsed is None:
            continue
        index, ts, uid = parsed
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((path, index, ts, uid, mtime))
    if not candidates:
        return None, None, 0
    _, _, run_ts, run_uid, _ = max(candidates, key=lambda item: item[4])
    next_index = max(idx for _, idx, ts, uid, _ in candidates if ts == run_ts and uid == run_uid) + 1
    return run_ts, run_uid, next_index


def archive_round(
    *,
    workdir: Path,
    history_dir: str,
    round_id: str,
    method_file: str,
    method_src: Path,
    result_dir: Path | None,
    dest_allow_exists: bool,
) -> Path:
    dest = workdir / history_dir / round_id
    dest.mkdir(parents=True, exist_ok=dest_allow_exists)

    if method_src.is_file():
        method_dest = dest / method_file
        method_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(method_src, method_dest)

    if result_dir is None:
        return dest

    src = result_dir if result_dir.is_absolute() else workdir / result_dir
    src = src.resolve()
    if not src.is_dir():
        (dest / "proposal_result_dir.txt").write_text(str(src), encoding="utf-8")
        return dest

    proposal_dest = dest / "proposal_results"
    proposal_dest.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for path in sorted(src.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(src)
        target = proposal_dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        copied.append(str(rel))
    if copied:
        (dest / "proposal_results_manifest.json").write_text(
            json.dumps({"source_dir": str(src), "copied_files": copied}, indent=2),
            encoding="utf-8",
        )

    for child in src.iterdir():
        try:
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
        except OSError:
            pass
    return dest


def append_workflow_index(workdir: Path, history_dir: str, row: dict[str, Any]) -> None:
    index_path = workdir / history_dir / "workflow_index.jsonl"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    with index_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def run_eval_subprocess(
    *,
    cmd: tuple[str, ...],
    cwd: Path,
    timeout_sec: float,
) -> tuple[int, str, str]:
    try:
        completed = subprocess.run(
            list(cmd),
            cwd=str(cwd.resolve()),
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=os.environ.copy(),
        )
        return completed.returncode, completed.stdout or "", completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        stdout = exc.stdout if isinstance(exc.stdout, str) else ""
        stderr = exc.stderr if isinstance(exc.stderr, str) else str(exc)
        return -1, stdout, stderr + "\n[workflow] eval timeout\n"
    except OSError as exc:
        return -1, "", str(exc)


def resolve_method_template(workdir: Path, method_file: str, template_file: str | None) -> Path | None:
    if template_file:
        template_path = workdir / template_file
        return template_path.resolve() if template_path.is_file() else None
    method_path = workdir / method_file
    suffix_template = method_path.with_suffix(".template.py")
    if suffix_template.is_file():
        return suffix_template.resolve()
    autotts_style = method_path.parent / "method.template.py"
    return autotts_style.resolve() if autotts_style.is_file() else None


def proposer_config_for_round(
    *,
    cfg: WorkflowConfig,
    round_output_dir: Path,
    context_file: Path | None,
) -> ProposerConfig:
    return ProposerConfig(
        workdir=str(cfg.workdir.resolve()),
        prompt_path=str(cfg.prompt_path.resolve()),
        output_dir=str(round_output_dir.resolve()),
        method_file=cfg.method_file,
        history_dir=cfg.history_dir,
        context_file=str(context_file.resolve()) if context_file is not None else None,
        codex_bin=os.environ.get("CODEX_BIN", "codex"),
        model=os.environ.get("CODEX_MODEL"),
        exec_timeout_sec=optional_float_env("CODEX_EXEC_TIMEOUT_SEC"),
        extra_codex_args=extra_codex_args_from_env(),
        exec_args=exec_codex_args_from_env(),
        plain_exec=truthy_env("CODEX_PLAIN_EXEC"),
    )


def build_context_pack(
    *,
    workdir: Path,
    method_file: str,
    history_dir: str,
    template_path: Path | None,
    output_dir: Path,
    max_history_rounds: int,
) -> Path:
    """Write a small per-round context file for the proposer.

    This follows AutoTTS' context discipline: the proposer is pointed at a
    narrow controller file, the environment API, baselines, and recent compact
    histories instead of being invited to scan raw logs or the full repository.
    """

    context_path = output_dir / "context_pack.md"
    lines: list[str] = [
        "# Flow AutoTTS Context Pack",
        "",
        "Read this file first. It is the intended context budget for this round.",
        "",
        "## Allowed First-Pass Reads",
        "",
        "- `flow_tts_controller_implementation_spec.md`",
        f"- `{method_file}`",
        "- `flow_autotts/controllers/baselines.py`",
        "- `flow_autotts/core/state.py`",
        "- `flow_autotts/core/errors.py`",
        "- `flow_autotts/experiments/pickscore_sd35/harness.py`",
        "- `flow_autotts/experiments/pickscore_sd35/env.py`",
        "- recent round summaries listed below",
        "",
        "## Write Boundary",
        "",
        f"- Edit only `{method_file}`.",
        "- Do not edit the harness, environment, dataset loader, workflow, tests, logs, model directories, or datasets.",
        "- Keep the controller self-contained. The workflow resets it from the template before every round.",
        "",
        "## Context Discipline",
        "",
        "- Do not run broad repository scans such as `find .` or unconstrained `rg` from repo root.",
        "- Do not bulk-read raw `history.json`, raw event logs, datasets, `SD_3.5_med/`, `PickScore_v1/`, `flow_grpo/`, `.git/`, or `logs/`.",
        "- If a compact summary points to a concrete anomaly, inspect only the relevant small snippet from that round.",
        "- Prefer targeted reads of the files listed above.",
        "",
    ]
    if template_path is not None:
        try:
            rel_template = template_path.relative_to(workdir)
        except ValueError:
            rel_template = template_path
        lines.extend(["## Template", "", f"- `{rel_template}`", ""])

    baseline_refs = _load_baseline_references(workdir)
    lines.extend(_baseline_context_lines(workdir, baseline_refs))

    recent_rounds = _recent_history_rounds(workdir / history_dir, max_history_rounds)
    lines.extend(_frontier_context_lines(workdir, recent_rounds, baseline_refs))

    lines.extend(["## Recent History", ""])
    if not recent_rounds:
        lines.extend(["No prior rounds found. Treat this as round 0.", ""])
    for round_path in recent_rounds:
        lines.extend(_round_context_lines(workdir, round_path))

    context_path.parent.mkdir(parents=True, exist_ok=True)
    context_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return context_path


def _recent_history_rounds(history_path: Path, max_rounds: int) -> list[Path]:
    if max_rounds <= 0 or not history_path.is_dir():
        return []
    candidates: list[tuple[int, float, Path]] = []
    for path in history_path.iterdir():
        if not path.is_dir():
            continue
        parsed = parse_round_id(path.name)
        if parsed is None:
            continue
        index, _ts, _uid = parsed
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((index, mtime, path))
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return [path for _index, _mtime, path in candidates[:max_rounds]]


def _round_context_lines(workdir: Path, round_path: Path) -> list[str]:
    try:
        rel_round = round_path.relative_to(workdir)
    except ValueError:
        rel_round = round_path
    lines = [f"### `{rel_round}`", ""]
    summary_path = round_path / "proposal_results" / "summary.json"
    method_paths = sorted(round_path.rglob("optimal.py"))
    if method_paths:
        try:
            rel_method = method_paths[0].relative_to(workdir)
        except ValueError:
            rel_method = method_paths[0]
        lines.append(f"- controller snapshot: `{rel_method}`")
    if summary_path.is_file():
        try:
            rel_summary = summary_path.relative_to(workdir)
        except ValueError:
            rel_summary = summary_path
        lines.append(f"- compact summary: `{rel_summary}`")
        summary_text = _bounded_text(summary_path, limit=20_000)
        if summary_text:
            lines.extend(["", "```json", summary_text, "```"])
    else:
        lines.append("- compact summary: not found")
    lines.append("")
    return lines


def _load_baseline_references(workdir: Path, max_files: int = 3) -> list[tuple[Path, list[dict[str, Any]]]]:
    root = workdir / "logs" / "flow_autotts" / "pickscore_sd35"
    if not root.is_dir():
        return []
    candidates: list[tuple[float, Path]] = []
    for path in root.glob("*baseline*/aggregate_summary.json"):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            mtime = 0.0
        candidates.append((mtime, path))
    candidates.sort(reverse=True)

    refs: list[tuple[Path, list[dict[str, Any]]]] = []
    for _mtime, path in candidates[:max_files]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data, list):
            rows = [row for row in data if isinstance(row, dict)]
            if rows:
                refs.append((path, rows))
    return refs


def _baseline_context_lines(
    workdir: Path,
    baseline_refs: list[tuple[Path, list[dict[str, Any]]]],
) -> list[str]:
    lines = ["## Baseline References", ""]
    if not baseline_refs:
        lines.extend(
            [
                "No compact baseline `aggregate_summary.json` files found under "
                "`logs/flow_autotts/pickscore_sd35/*baseline*/`.",
                "",
            ]
        )
        return lines

    lines.append(
        "These compact baseline files are injected by the workflow so the proposer can compare by nearest NFE."
    )
    lines.append("")
    for path, rows in baseline_refs:
        try:
            rel_path = path.relative_to(workdir)
        except ValueError:
            rel_path = path
        lines.extend([f"### `{rel_path}`", "", "```json"])
        lines.append(json.dumps(rows, ensure_ascii=False, indent=2))
        lines.extend(["```", ""])
    return lines


def _frontier_context_lines(
    workdir: Path,
    recent_rounds: list[Path],
    baseline_refs: list[tuple[Path, list[dict[str, Any]]]],
) -> list[str]:
    lines = ["## Recent Round Frontier Comparison", ""]
    if not recent_rounds:
        lines.extend(["No prior rounds found.", ""])
        return lines

    baseline_rows = baseline_refs[0][1] if baseline_refs else []
    table_rows: list[str] = []
    table_rows.append(
        "| round | beta | mean_nfe | reward | nearest_baseline_nfe | baseline_reward | delta | actions |"
    )
    table_rows.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |")

    for round_path in recent_rounds:
        summary_path = round_path / "proposal_results" / "summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        for sweep in _iter_summary_beta_sweeps(summary):
            beta = _optional_float(sweep.get("beta"))
            nfe = _optional_float(sweep.get("nfe"))
            reward = _first_optional_float(sweep, ("reward", "final_reward"))
            if beta is None or nfe is None or reward is None:
                continue
            baseline = _nearest_baseline_row(nfe, baseline_rows)
            if baseline is None:
                base_nfe = base_reward = delta = ""
            else:
                base_nfe_float = _optional_float(baseline.get("nfe"))
                base_reward_float = _optional_float(baseline.get("reward"))
                base_nfe = _format_float(base_nfe_float, 3)
                base_reward = _format_float(base_reward_float, 6)
                delta = (
                    _format_float(reward - base_reward_float, 6)
                    if base_reward_float is not None
                    else ""
                )
            table_rows.append(
                "| "
                + " | ".join(
                    [
                        round_path.name.split("_", 1)[0],
                        _format_float(beta, 3),
                        _format_float(nfe, 3),
                        _format_float(reward, 6),
                        base_nfe,
                        base_reward,
                        delta,
                        _format_action_summary(sweep),
                    ]
                )
                + " |"
            )

    if len(table_rows) <= 2:
        lines.extend(["No parseable beta-sweep rows found in recent summaries.", ""])
        return lines
    lines.extend(table_rows)
    lines.append("")
    return lines


def _iter_summary_beta_sweeps(summary: dict[str, Any]) -> list[dict[str, Any]]:
    sweeps: list[dict[str, Any]] = []
    for round_item in summary.get("rounds") or []:
        if not isinstance(round_item, dict):
            continue
        for sweep in round_item.get("beta_sweep") or []:
            if isinstance(sweep, dict):
                sweeps.append(sweep)
    return sweeps


def _nearest_baseline_row(nfe: float, rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    candidates: list[tuple[float, dict[str, Any]]] = []
    for row in rows:
        baseline_nfe = _optional_float(row.get("nfe"))
        if baseline_nfe is None:
            continue
        candidates.append((abs(baseline_nfe - nfe), row))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _first_optional_float(values: dict[str, Any], keys: tuple[str, ...]) -> float | None:
    for key in keys:
        result = _optional_float(values.get(key))
        if result is not None:
            return result
    return None


def _format_float(value: float | None, digits: int) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _format_action_summary(sweep: dict[str, Any]) -> str:
    behavior = sweep.get("behavior_summary")
    if isinstance(behavior, str) and behavior.strip():
        return behavior.replace("|", "/")

    stats = sweep.get("action_statistics")
    if not isinstance(stats, dict):
        return ""

    keys = ("spawn", "forward", "preview", "backward", "prune", "mean_nfe")
    parts: list[str] = []
    for key in keys:
        value = _optional_float(stats.get(key))
        if value is not None:
            parts.append(f"{key}={value:.2f}")
    return ", ".join(parts)


def _bounded_text(path: Path, limit: int) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "\n... <truncated by workflow context pack> ..."


async def run_workflow(cfg: WorkflowConfig) -> list[dict[str, Any]]:
    cfg.workdir.mkdir(parents=True, exist_ok=True)
    (cfg.workdir / cfg.history_dir).mkdir(parents=True, exist_ok=True)
    cfg.codex_log_parent.mkdir(parents=True, exist_ok=True)
    method_path = (cfg.workdir / cfg.method_file).resolve()
    template_path = resolve_method_template(cfg.workdir, cfg.method_file, cfg.template_file)
    eval_cwd = (cfg.eval_cwd or cfg.workdir).resolve()

    start_index = 0
    resumed = False
    if cfg.resume:
        run_ts, run_uid, next_index = scan_history_resume(cfg.workdir, cfg.history_dir)
        if run_ts and run_uid and next_index > 0:
            start_index = next_index
            resumed = True
            print(f"[workflow] Resuming {run_ts}_{run_uid} from round r{start_index:04d}.")
        else:
            run_ts = time.strftime("%Y%m%d_%H%M%S")
            run_uid = uuid.uuid4().hex[:8]
            print("[workflow] No resumable history found; starting a new run.")
    else:
        run_ts = time.strftime("%Y%m%d_%H%M%S")
        run_uid = uuid.uuid4().hex[:8]

    if start_index >= cfg.rounds:
        print("[workflow] Planned rounds are already complete.")
        return []

    results: list[dict[str, Any]] = []
    for round_index in range(start_index, cfg.rounds):
        round_id = f"r{round_index:04d}_{run_ts}_{run_uid}"
        round_log = cfg.codex_log_parent / round_id
        round_log.mkdir(parents=True, exist_ok=resumed)

        method_path.parent.mkdir(parents=True, exist_ok=True)
        if template_path is not None:
            shutil.copy2(template_path, method_path)

        context_file = build_context_pack(
            workdir=cfg.workdir,
            method_file=cfg.method_file,
            history_dir=cfg.history_dir,
            template_path=template_path,
            output_dir=round_log,
            max_history_rounds=cfg.context_history_rounds,
        )

        proposal_result = await propose(
            proposer_config_for_round(
                cfg=cfg,
                round_output_dir=round_log,
                context_file=context_file,
            )
        )

        eval_rc: int | None = None
        if cfg.eval_cmd:
            eval_rc, stdout, stderr = await asyncio.to_thread(
                run_eval_subprocess,
                cmd=cfg.eval_cmd,
                cwd=eval_cwd,
                timeout_sec=cfg.eval_timeout_sec,
            )
            (round_log / "eval_stdout.txt").write_text(stdout, encoding="utf-8")
            (round_log / "eval_stderr.txt").write_text(stderr, encoding="utf-8")

        history_path = archive_round(
            workdir=cfg.workdir,
            history_dir=cfg.history_dir,
            round_id=round_id,
            method_file=cfg.method_file,
            method_src=method_path,
            result_dir=cfg.result_dir,
            dest_allow_exists=resumed,
        )
        archived_results = history_path / "proposal_results"
        row = {
            "round_index": round_index,
            "round_id": round_id,
            "proposal_status": proposal_result.get("status"),
            "eval_returncode": eval_rc,
            "codex_output_dir": str(round_log),
            "history_archive": str(history_path),
            "proposal_results_archive": str(archived_results) if archived_results.is_dir() else "",
        }
        append_workflow_index(cfg.workdir, cfg.history_dir, row)
        results.append(row)

    summary_path = cfg.codex_log_parent / f"workflow_summary_{run_ts}_{run_uid}.json"
    prior_rounds: list[dict[str, Any]] = []
    if summary_path.is_file():
        try:
            prior_rounds = list(json.loads(summary_path.read_text(encoding="utf-8")).get("rounds") or [])
        except (json.JSONDecodeError, OSError):
            prior_rounds = []
    summary = {
        "workdir": str(cfg.workdir),
        "method_file": cfg.method_file,
        "history_dir": cfg.history_dir,
        "rounds_planned": cfg.rounds,
        "eval_cmd": list(cfg.eval_cmd),
        "result_dir": str(cfg.result_dir) if cfg.result_dir is not None else "",
        "run_ts": run_ts,
        "run_uid": run_uid,
        "resumed": resumed,
        "resume_from_index": start_index if resumed else 0,
        "rounds": prior_rounds + results,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return results


def workflow_from_env() -> WorkflowConfig:
    workdir = Path(os.environ["WORKFLOW_WORKDIR"]).expanduser().resolve()
    method_file = os.environ["WORKFLOW_METHOD_FILE"]
    history_dir = os.environ.get("WORKFLOW_HISTORY_DIR", "history")
    prompt_path = Path(os.environ["WORKFLOW_PROMPT_PATH"]).expanduser().resolve()
    rounds = int(os.environ.get("WORKFLOW_ROUNDS", "5"))
    log_parent = Path(
        os.environ.get("WORKFLOW_CODEX_LOG_PARENT", str(workdir / ".workflow_logs"))
    ).expanduser().resolve()
    result_dir_raw = os.environ.get("WORKFLOW_RESULT_DIR", "").strip()
    if result_dir_raw in {"", "-", "0"}:
        result_dir = None
    else:
        result_dir = Path(result_dir_raw).expanduser().resolve()
    eval_raw = os.environ.get("WORKFLOW_EVAL_CMD", "").strip()
    eval_cmd = tuple(shlex.split(eval_raw)) if eval_raw else ()
    eval_cwd_raw = os.environ.get("WORKFLOW_EVAL_CWD", "").strip()
    eval_cwd = Path(eval_cwd_raw).expanduser().resolve() if eval_cwd_raw else None
    template_raw = os.environ.get("WORKFLOW_TEMPLATE_FILE", "").strip()
    return WorkflowConfig(
        workdir=workdir,
        method_file=method_file,
        history_dir=history_dir,
        prompt_path=prompt_path,
        rounds=rounds,
        codex_log_parent=log_parent,
        result_dir=result_dir,
        eval_cmd=eval_cmd,
        eval_cwd=eval_cwd,
        eval_timeout_sec=float(os.environ.get("WORKFLOW_EVAL_TIMEOUT_SEC", "7200")),
        resume=truthy_env("WORKFLOW_RESUME"),
        template_file=template_raw or None,
        context_history_rounds=int(os.environ.get("WORKFLOW_CONTEXT_HISTORY_ROUNDS", "5")),
    )


async def main() -> None:
    result = await run_workflow(workflow_from_env())
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
