"""Compatibility fixtures derived from EyeFlow v1.22.0's published schema.

Source: src/input_output/schema/eyeflow_output.py at commit
b1a5063e69a21d65c3492ef16d8634bf20bba4b9 (DigitalHolography/EyeFlow).
These are synthetic data, not patient recordings.
"""

from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app_settings import AppSettingsStore  # noqa: E402
from input_output import stage_analysis_file  # noqa: E402
from input_output.eyeflow_schema import BEAT_PERIOD, require_dataset  # noqa: E402


def write_current_fixture(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as output:
        output.create_dataset("app_versions", data=json.dumps({"EF_version": "1.22.0"}))
        for root, method, offset in (
            ("ProcessingMoments", "doppler_moments", 1),
            ("ProcessingBandRatio", "frequency_bands", 100),
        ):
            branch = output.create_group(root)
            branch.attrs["velocity_estimation_method"] = method
            branch.create_dataset(
                "CardiacCycle/Systole/CycleDurationSeconds/value", data=[0.8, 1.2]
            )
            for vessel in ("Artery", "Vein"):
                metric = branch.create_dataset(
                    f"Metrics/waveform_shape_metrics/{vessel.lower()}/global/bandlimited/RI",
                    data=[offset, offset + 1],
                )
                metric.attrs["source_representation"] = "filtered_bandlimited"
                bvr = branch.create_dataset(
                    f"BloodVolumeRate/{vessel}/dynamicEdges/value",
                    data=np.arange(24).reshape(2, 3, 4) + offset,
                )
                bvr.attrs["units"] = "uL/min"
                for representation, increment in (("Raw", 0), ("Filtered", 1000)):
                    for suffix, extra in (("", 0), ("BandLimited/", 10000)):
                        dataset = branch.create_dataset(
                            f"Velocity/PerBeat/Global/{representation}/{suffix}{vessel}Signal",
                            data=np.arange(16).reshape(2, 8)
                            + offset
                            + increment
                            + extra,
                        )
                        dataset.attrs["dimDesc"] = ["beat", "sample"]
                        dataset.attrs["units"] = "mm/s"
                        dataset.attrs["representation"] = representation
            branch.create_dataset(
                "Velocity/PerBeat/Segments/Raw/ArterySignal",
                data=np.zeros((8, 2, 3, 4)),
            )
        output.create_dataset("Meta/FrameIntervalSeconds/value", data=0.01)


class CurrentEyeFlowSchemaTests(unittest.TestCase):
    def test_current_roots_signals_bvr_and_provenance(self):
        for selection, root, offset in (
            ("Processing", "ProcessingMoments", 1),
            ("ProcessingMoments", "ProcessingMoments", 1),
            ("ProcessingAlt", "ProcessingBandRatio", 100),
            ("ProcessingBandRatio", "ProcessingBandRatio", 100),
        ):
            with (
                self.subTest(selection=selection),
                tempfile.TemporaryDirectory() as tmp,
            ):
                source, target = Path(tmp) / "input.h5", Path(tmp) / "output.h5"
                write_current_fixture(source)
                before = hashlib.sha256(source.read_bytes()).digest()
                stage_analysis_file(source, target, processing_branch=selection)
                self.assertEqual(before, hashlib.sha256(source.read_bytes()).digest())
                with h5py.File(target) as staged, h5py.File(source) as original:
                    self.assertEqual(staged.attrs["angioeye_processing_branch"], root)
                    self.assertEqual(
                        staged.attrs["angioeye_velocity_estimation_method"],
                        original[root].attrs["velocity_estimation_method"],
                    )
                    np.testing.assert_array_equal(
                        require_dataset(staged, BEAT_PERIOD), [0.8, 1.2]
                    )
                    versions = json.loads(staged["app_versions"][()])
                    self.assertEqual(versions["EF_version"], "1.22.0")
                    self.assertEqual(versions["AE_version"], "1.24.1")
                    for vessel in ("Artery", "Vein"):
                        for mode, extra in (("Raw", 0), ("BandLimited", 10000)):
                            signal = staged[
                                f"Processing/VelocityPerBeat/{vessel}/{mode}/value"
                            ]
                            np.testing.assert_array_equal(
                                signal, np.arange(16).reshape(2, 8) + offset + extra
                            )
                            self.assertEqual(signal.attrs["units"], "mm/s")
                            self.assertEqual(
                                list(signal.attrs["dimDesc"]), ["beat", "sample"]
                            )
                            self.assertTrue(
                                signal.attrs["angioeye_source_dataset"].startswith(
                                    f"/{root}/Velocity/PerBeat/Global/Raw/"
                                )
                            )
                        filtered = staged[
                            f"Processing/Velocity/PerBeat/Global/Filtered/{vessel}Signal"
                        ]
                        np.testing.assert_array_equal(
                            filtered, np.arange(16).reshape(2, 8) + offset + 1000
                        )
                        bvr = f"BloodVolumeRate/{vessel}/dynamicEdges/value"
                        np.testing.assert_array_equal(
                            staged[f"Processing/{bvr}"], original[f"{root}/{bvr}"]
                        )
                        self.assertEqual(
                            staged[f"Processing/{bvr}"].attrs["units"], "uL/min"
                        )
                    self.assertNotIn("Processing/Velocity/PerBeat/Segments", staged)

    def test_filtered_is_never_used_as_missing_raw(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "in.h5", Path(tmp) / "out.h5"
            write_current_fixture(source)
            with h5py.File(source, "a") as original:
                del original["ProcessingMoments/Velocity/PerBeat/Global/Raw"]
            stage_analysis_file(source, target)
            with h5py.File(target) as staged:
                self.assertNotIn("Processing/VelocityPerBeat", staged)
                self.assertIn(
                    "Processing/Velocity/PerBeat/Global/Filtered/ArterySignal", staged
                )

    def test_errors_preserve_destination_and_never_change_methods(self):
        for problem, expected in (
            ("missing", "ProcessingMoments/Metrics"),
            ("ambiguous", "Ambiguous"),
            ("metadata", "declares"),
            ("signals", "Ambiguous EyeFlow waveform"),
        ):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as tmp:
                source, target = Path(tmp) / "in.h5", Path(tmp) / "out.h5"
                write_current_fixture(source)
                target.write_bytes(b"previous result")
                with h5py.File(source, "a") as original:
                    if problem == "missing":
                        del original["ProcessingMoments"]
                    elif problem == "ambiguous":
                        original.create_group("Processing/Metrics")
                    elif problem == "metadata":
                        original["ProcessingMoments"].attrs[
                            "velocity_estimation_method"
                        ] = "frequency_bands"
                    else:
                        original.create_dataset(
                            "ProcessingMoments/VelocityPerBeat/Artery/Raw/value",
                            data=[99],
                        )
                with self.assertRaisesRegex(ValueError, expected):
                    stage_analysis_file(source, target)
                self.assertEqual(target.read_bytes(), b"previous result")
                self.assertEqual(list(Path(tmp).glob("*.tmp")), [])

    def test_identical_root_hard_link_is_not_ambiguous(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, target = Path(tmp) / "in.h5", Path(tmp) / "out.h5"
            write_current_fixture(source)
            with h5py.File(source, "a") as original:
                original["Processing"] = original["ProcessingMoments"]
            stage_analysis_file(source, target)

    def test_current_names_and_labels_roundtrip_in_settings(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = AppSettingsStore(Path(tmp) / "settings.json")
            for selection, expected in (
                ("ProcessingMoments", "Processing"),
                ("ProcessingBandRatio", "ProcessingAlt"),
                ("Doppler moments", "Processing"),
                ("Frequency-band ratio", "ProcessingAlt"),
            ):
                store.save_processing_branch(selection)
                self.assertEqual(store.load_processing_branch(), expected)


if __name__ == "__main__":
    unittest.main()
