from pathlib import Path

import numpy as np
import pytest

from general_motion_retargeting.utils.asap_smpl import (
    convert_asap_smpl_to_smplx,
)


def write_asap_clip(path: Path, **overrides) -> None:
    values = {
        "trans": np.array([[3.0, -2.0, 0.9], [3.2, -1.8, 1.0]]),
        "gender": np.asarray("neutral"),
        "mocap_framerate": np.asarray(30),
        "betas": np.arange(16, dtype=float),
        "poses": np.arange(2 * 72, dtype=float).reshape(2, 72) / 100.0,
    }
    values.update(overrides)
    np.savez(path, **values)


def test_convert_asap_smpl_preserves_body_pose_and_normalizes_origin(
    tmp_path: Path,
) -> None:
    source = tmp_path / "motion.npz"
    write_asap_clip(source)

    converted, metadata = convert_asap_smpl_to_smplx(source)

    with np.load(source, allow_pickle=False) as original:
        np.testing.assert_array_equal(
            converted["root_orient"], original["poses"][:, :3]
        )
        np.testing.assert_array_equal(
            converted["pose_body"], original["poses"][:, 3:66]
        )
        np.testing.assert_array_equal(converted["betas"], original["betas"])
    np.testing.assert_allclose(converted["trans"], [[0.0, 0.0, 0.9], [0.2, 0.2, 1.0]])
    assert float(converted["mocap_frame_rate"]) == 30.0
    assert metadata["frames"] == 2
    assert metadata["duration_seconds"] == pytest.approx(2 / 30)
    assert len(metadata["source_sha256"]) == 64
    assert str(converted["source_sha256"]) == metadata["source_sha256"]


def test_convert_asap_smpl_pads_ten_shape_coefficients(tmp_path: Path) -> None:
    source = tmp_path / "motion.npz"
    write_asap_clip(source, betas=np.arange(10, dtype=float))

    converted, _ = convert_asap_smpl_to_smplx(source)

    np.testing.assert_array_equal(converted["betas"][:10], np.arange(10))
    np.testing.assert_array_equal(converted["betas"][10:], np.zeros(6))


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"poses": np.zeros((2, 65))}, "poses shape"),
        ({"trans": np.zeros((3, 3))}, "translation shape"),
        ({"mocap_framerate": np.asarray(0)}, "frame rate"),
        ({"gender": np.asarray("unknown")}, "unsupported"),
    ],
)
def test_convert_asap_smpl_rejects_invalid_data(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    source = tmp_path / "motion.npz"
    write_asap_clip(source, **overrides)

    with pytest.raises(ValueError, match=message):
        convert_asap_smpl_to_smplx(source)
