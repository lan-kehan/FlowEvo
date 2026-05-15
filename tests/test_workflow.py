from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from flow_autotts.workflow.runner import (
    archive_round,
    build_context_pack,
    parse_round_id,
    resolve_method_template,
)


class WorkflowTests(unittest.TestCase):
    def test_parse_round_id(self):
        self.assertEqual(
            parse_round_id("r0003_20260513_120102_abcdef12"),
            (3, "20260513_120102", "abcdef12"),
        )
        self.assertIsNone(parse_round_id("round3"))

    def test_resolve_suffix_template(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            method = root / "pkg" / "optimal.py"
            template = root / "pkg" / "optimal.template.py"
            method.parent.mkdir()
            method.write_text("# method\n", encoding="utf-8")
            template.write_text("# template\n", encoding="utf-8")

            self.assertEqual(
                resolve_method_template(root, "pkg/optimal.py", None),
                template.resolve(),
            )

    def test_archive_round_copies_method_results_and_clears_source_results(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            method = root / "flow_autotts" / "controllers" / "optimal.py"
            method.parent.mkdir(parents=True)
            method.write_text("class OptimalController: pass\n", encoding="utf-8")
            result_dir = root / "training_results"
            result_dir.mkdir()
            (result_dir / "summary.json").write_text("{}", encoding="utf-8")

            dest = archive_round(
                workdir=root,
                history_dir="history",
                round_id="r0000_20260513_120102_abcdef12",
                method_file="flow_autotts/controllers/optimal.py",
                method_src=method,
                result_dir=result_dir,
                dest_allow_exists=False,
            )

            self.assertTrue((dest / "flow_autotts" / "controllers" / "optimal.py").is_file())
            self.assertTrue((dest / "proposal_results" / "summary.json").is_file())
            self.assertEqual(list(result_dir.iterdir()), [])

    def test_context_pack_points_proposer_at_narrow_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            history_round = root / "history" / "r0000_20260513_120102_abcdef12"
            summary = history_round / "proposal_results" / "summary.json"
            snapshot = history_round / "flow_autotts" / "controllers" / "optimal.py"
            summary.parent.mkdir(parents=True)
            snapshot.parent.mkdir(parents=True)
            summary.write_text(
                """
{
      "rounds": [
    {
      "beta_sweep": [
        {
          "beta": 0.5,
          "nfe": 37.0,
          "reward": 0.84,
          "action_statistics": {"spawn": 3, "preview": 4, "backward": 1, "mean_nfe": 37}
        }
      ]
    }
  ]
}
""".strip(),
                encoding="utf-8",
            )
            snapshot.write_text("class OptimalController: pass\n", encoding="utf-8")
            baseline = (
                root
                / "logs"
                / "flow_autotts"
                / "pickscore_sd35"
                / "ode_baseline_equiv"
                / "aggregate_summary.json"
            )
            baseline.parent.mkdir(parents=True)
            baseline.write_text(
                '[{"nfe": 37, "reward": 0.83, "num_samples": 500}]',
                encoding="utf-8",
            )
            out = root / "logs"

            context = build_context_pack(
                workdir=root,
                method_file="flow_autotts/controllers/optimal.py",
                history_dir="history",
                template_path=None,
                output_dir=out,
                max_history_rounds=1,
            )
            text = context.read_text(encoding="utf-8")

            self.assertIn("Allowed First-Pass Reads", text)
            self.assertIn("Edit only `flow_autotts/controllers/optimal.py`", text)
            self.assertIn("r0000_20260513_120102_abcdef12", text)
            self.assertIn("Do not bulk-read raw `history.json`", text)
            self.assertIn("Baseline References", text)
            self.assertIn("Recent Round Frontier Comparison", text)
            self.assertIn(
                "| r0000 | 0.500 | 37.000 | 0.840000 | 37.000 | 0.830000 | 0.010000 | spawn=3.00, preview=4.00, backward=1.00, mean_nfe=37.00 |",
                text,
            )


if __name__ == "__main__":
    unittest.main()
