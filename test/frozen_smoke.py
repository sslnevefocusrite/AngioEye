"""Exercise a built Windows executable with synthetic old and new EyeFlow files."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
from test_current_eyeflow_schema import write_current_fixture
from test_eyeflow_v2_schema import write_eyeflow_fixture


def smoke(executable: Path) -> None:
    executable = executable.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="angioeye-frozen-") as tmp:
        root = Path(tmp)
        cohort = root / "cohort"
        cohort.mkdir()
        sources = []
        for name, writer in (
            ("old", write_eyeflow_fixture),
            ("current", write_current_fixture),
        ):
            (cohort / f"{name}.holo").write_bytes(b"synthetic acquisition marker")
            source = cohort / name / f"{name}_EF" / "h5" / f"{name}_EF.h5"
            source.parent.mkdir(parents=True)
            writer(source)
            sources.append(source)
        before = [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
        env = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "ANGIOEYE_VERSION"):
            env.pop(key, None)
        env.update(APPDATA=str(root / "settings"), LOCALAPPDATA=str(root / "local"))
        for selection in (
            "Processing",
            "ProcessingAlt",
            "ProcessingMoments",
            "ProcessingBandRatio",
        ):
            output = root / selection
            command = [
                str(executable),
                "--data",
                str(cohort),
                "--output",
                str(output),
                "--pipelines",
                "HTML summary",
                "Pipeline Metrics Manifest",
                "--processing-branch",
                selection,
            ]
            result = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, timeout=120
            )
            if result.returncode:
                raise RuntimeError(
                    f"Frozen batch failed ({selection}): {result.returncode}\n{result.stdout}\n{result.stderr}"
                )
            batch = json.loads(
                (output / "cohort-results" / "batch_run.json").read_text()
            )
            assert not batch["failures"], batch["failures"]
            assert len(batch["input_files"]) == 2
            assert all(p["status"] == "completed" for p in batch["pipelines"])
            manifest = json.loads(
                (
                    output / "cohort-results" / "pipeline_metrics_manifest.json"
                ).read_text()
            )
            current = next(f for f in manifest["files"] if "current_AE" in f["path"])
            assert (
                "Artery/dynamicEdges/value"
                in current["source_products"]["BloodVolumeRate"]
            )
            alternate = selection in ("ProcessingAlt", "ProcessingBandRatio")
            for name in ("old", "current"):
                app_dir = output / name / f"{name}_AE"
                assert (app_dir / "html" / f"{name}_AE.html").stat().st_size > 1000
                with h5py.File(app_dir / "h5" / f"{name}_AE.h5") as staged:
                    expected = (
                        (100 if alternate else 1)
                        if name == "current"
                        else (9 if alternate else 1)
                    )
                    np.testing.assert_array_equal(
                        staged[
                            "Processing/Metrics/waveform_shape_metrics/artery/global/bandlimited/RI"
                        ],
                        [expected, expected + 1],
                    )
                    assert (
                        json.loads(staged["app_versions"][()])["AE_version"] == "1.24.1"
                    )
            print(
                f"PASS: frozen {selection}, old + current, HTML + manifest + source method values"
            )
        assert before == [hashlib.sha256(p.read_bytes()).hexdigest() for p in sources]
        print("PASS: all source hashes unchanged")


if __name__ == "__main__":
    smoke(Path(sys.argv[1]))
