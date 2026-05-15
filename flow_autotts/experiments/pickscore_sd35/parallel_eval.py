"""Run sharded PickScore SD3.5 harness evaluations across multiple devices."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Sequence

from flow_autotts.experiments.pickscore_sd35.harness import (
    _default_dataset_dir,
    _default_model_path,
    _default_pickscore_path,
    compact_summary,
)
from flow_autotts.experiments.pickscore_sd35.merge_shards import merge_histories


def _split_devices(value: str) -> list[str]:
    devices = [item.strip() for item in value.replace(",", " ").split()]
    return [device for device in devices if device]


def _optional_device(devices: Sequence[str], index: int) -> str | None:
    if not devices:
        return None
    if len(devices) == 1:
        return devices[0]
    return devices[index]


def _add_if(cmd: list[str], flag: str, value: object | None) -> None:
    if value is not None and str(value) != "":
        cmd.extend([flag, str(value)])


def _run_shard(
    *,
    args: argparse.Namespace,
    shard_index: int,
    num_shards: int,
    device: str,
    text_encoder_device: str | None,
    score_device: str | None,
    shard_dir: Path,
) -> Path:
    shard_dir.mkdir(parents=True, exist_ok=True)
    history_path = shard_dir / "history.json"
    summary_path = shard_dir / "summary.json"

    cmd = [
        sys.executable,
        "-m",
        "flow_autotts.experiments.pickscore_sd35.harness",
        "--dataset",
        str(args.dataset),
        "--split",
        str(args.split),
        "--sample-size",
        str(args.sample_size),
        "--sample-seed",
        str(args.sample_seed),
        "--num-shards",
        str(num_shards),
        "--shard-index",
        str(shard_index),
        "--rounds",
        str(args.rounds),
        "--controllers",
        *args.controllers,
        "--betas",
        *[str(beta) for beta in args.betas],
        "--budget",
        str(args.budget),
        "--output",
        str(history_path),
        "--summary-output",
        str(summary_path),
        "--model",
        str(args.model),
        "--pickscore-model",
        str(args.pickscore_model),
        "--num-steps",
        str(args.num_steps),
        "--resolution",
        str(args.resolution),
        "--guidance-scale",
        str(args.guidance_scale),
        "--noise-level",
        str(args.noise_level),
        "--sde-type",
        str(args.sde_type),
        "--score-dtype",
        str(args.score_dtype),
        "--device",
        device,
    ]
    _add_if(cmd, "--pickscore-processor", args.pickscore_processor)
    _add_if(cmd, "--text-encoder-device", text_encoder_device)
    _add_if(cmd, "--score-device", score_device)
    _add_if(cmd, "--dtype", args.dtype)
    if args.compact:
        cmd.append("--compact")
    if args.allow_remote_files:
        cmd.append("--allow-remote-files")
    if args.progress:
        cmd.append("--progress")
    if args.offload_text_encoders_after_encode:
        cmd.append("--offload-text-encoders-after-encode")

    stdout = (shard_dir / "stdout.log").open("w", encoding="utf-8")
    stderr = (shard_dir / "stderr.log").open("w", encoding="utf-8")
    try:
        completed = subprocess.run(cmd, stdout=stdout, stderr=stderr, text=True)
    finally:
        stdout.close()
        stderr.close()
    if completed.returncode != 0:
        raise RuntimeError(
            f"shard {shard_index} on {device} failed with return code "
            f"{completed.returncode}; see {shard_dir}"
        )
    return history_path


def run_parallel_eval(args: argparse.Namespace) -> dict[str, object]:
    devices = _split_devices(args.devices)
    if not devices:
        raise ValueError("--devices must contain at least one device")
    text_devices = _split_devices(args.text_encoder_devices or "")
    score_devices = _split_devices(args.score_devices or "")
    if text_devices and len(text_devices) not in {1, len(devices)}:
        raise ValueError("--text-encoder-devices must have length 1 or match --devices")
    if score_devices and len(score_devices) not in {1, len(devices)}:
        raise ValueError("--score-devices must have length 1 or match --devices")

    output = Path(args.output).expanduser().resolve()
    summary_output = (
        Path(args.summary_output).expanduser().resolve()
        if args.summary_output is not None
        else None
    )
    shard_root = Path(args.shard_output_dir).expanduser().resolve() if args.shard_output_dir else output.parent / "shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    procs: list[tuple[int, str, Path, subprocess.Popen[str]]] = []
    shard_paths: list[Path] = []
    for shard_index, device in enumerate(devices):
        shard_dir = shard_root / f"shard_{shard_index:02d}"
        shard_dir.mkdir(parents=True, exist_ok=True)
        history_path = shard_dir / "history.json"
        summary_path = shard_dir / "summary.json"
        shard_paths.append(history_path)
        cmd = [
            sys.executable,
            "-m",
            "flow_autotts.experiments.pickscore_sd35.harness",
            "--dataset",
            str(args.dataset),
            "--split",
            str(args.split),
            "--sample-size",
            str(args.sample_size),
            "--sample-seed",
            str(args.sample_seed),
            "--num-shards",
            str(len(devices)),
            "--shard-index",
            str(shard_index),
            "--rounds",
            str(args.rounds),
            "--controllers",
            *args.controllers,
            "--betas",
            *[str(beta) for beta in args.betas],
            "--budget",
            str(args.budget),
            "--output",
            str(history_path),
            "--summary-output",
            str(summary_path),
            "--model",
            str(args.model),
            "--pickscore-model",
            str(args.pickscore_model),
            "--num-steps",
            str(args.num_steps),
            "--resolution",
            str(args.resolution),
            "--guidance-scale",
            str(args.guidance_scale),
            "--noise-level",
            str(args.noise_level),
            "--sde-type",
            str(args.sde_type),
            "--score-dtype",
            str(args.score_dtype),
            "--device",
            device,
        ]
        _add_if(cmd, "--pickscore-processor", args.pickscore_processor)
        _add_if(cmd, "--text-encoder-device", _optional_device(text_devices, shard_index))
        _add_if(cmd, "--score-device", _optional_device(score_devices, shard_index))
        _add_if(cmd, "--dtype", args.dtype)
        if args.compact:
            cmd.append("--compact")
        if args.allow_remote_files:
            cmd.append("--allow-remote-files")
        if args.progress:
            cmd.append("--progress")
        if args.offload_text_encoders_after_encode:
            cmd.append("--offload-text-encoders-after-encode")

        (shard_dir / "command.json").write_text(
            json.dumps(cmd, indent=2),
            encoding="utf-8",
        )
        stdout = (shard_dir / "stdout.log").open("w", encoding="utf-8")
        stderr = (shard_dir / "stderr.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, stdout=stdout, stderr=stderr, text=True)
        stdout.close()
        stderr.close()
        procs.append((shard_index, device, shard_dir, proc))

    failures: list[str] = []
    for shard_index, device, shard_dir, proc in procs:
        returncode = proc.wait()
        if returncode != 0:
            failures.append(f"shard {shard_index} on {device}: rc={returncode}, dir={shard_dir}")
    if failures:
        raise RuntimeError("; ".join(failures))

    merged = merge_histories(shard_paths, output)
    summary = compact_summary(merged)
    if summary_output is not None:
        summary_output.parent.mkdir(parents=True, exist_ok=True)
        summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "output": str(output),
        "summary_output": str(summary_output) if summary_output is not None else "",
        "shard_root": str(shard_root),
        "devices": devices,
        "summary": summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", required=True, help="Comma or space separated device list")
    parser.add_argument("--text-encoder-devices", default="")
    parser.add_argument("--score-devices", default="")
    parser.add_argument("--shard-output-dir", default=None)
    parser.add_argument("--dataset", default=str(_default_dataset_dir()))
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--sample-seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--controllers", nargs="+", choices=["optimal"], default=["optimal"])
    parser.add_argument("--betas", type=float, nargs="+", default=[0.5])
    parser.add_argument("--budget", type=int, default=64)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    parser.add_argument("--compact", action="store_true")
    parser.add_argument("--model", default=str(_default_model_path()))
    parser.add_argument("--pickscore-model", default=str(_default_pickscore_path()))
    parser.add_argument("--pickscore-processor", default=None)
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--score-dtype", default="float32")
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--guidance-scale", type=float, default=4.5)
    parser.add_argument("--noise-level", type=float, default=0.7)
    parser.add_argument("--sde-type", choices=["sde", "cps"], default="sde")
    parser.add_argument("--allow-remote-files", action="store_true")
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--offload-text-encoders-after-encode", action="store_true")
    args = parser.parse_args()

    result = run_parallel_eval(args)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
