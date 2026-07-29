from pathlib import Path
import time

import grpc
import numpy as np
import pytest

from general_motion_retargeting import BelloReferenceServer
from general_motion_retargeting import reference_pb2


MODEL_PATH = (
    Path(__file__).parents[1]
    / "assets"
    / "bello"
    / "mjcf"
    / "bello_full_body_boxes.xml"
)


def _watch(server: BelloReferenceServer, model_hash: str):
    channel = grpc.insecure_channel(f"127.0.0.1:{server.port}")
    method = channel.unary_stream(
        "/bello.reference.v1.ReferenceService/WatchReference",
        request_serializer=lambda value: value.SerializeToString(),
        response_deserializer=reference_pb2.WatchReferenceResponse.FromString,
    )
    call = method(
        reference_pb2.WatchReferenceRequest(
            protocol_version=1,
            robot_model_sha256=model_hash,
        )
    )
    return channel, call


def test_reference_server_publishes_exact_model_and_joint_schema() -> None:
    with BelloReferenceServer(MODEL_PATH, listen="127.0.0.1:0") as server:
        qpos = server.model.qpos0.copy()
        qpos[3] = 1.0
        server.publish_qpos(qpos, source_monotonic_ns=time.monotonic_ns())
        channel, call = _watch(server, server.model_sha256)
        try:
            info = next(call).info
            frame = next(call).frame
        finally:
            call.cancel()
            channel.close()

    assert info.robot_model_sha256 == server.model_sha256
    assert tuple(info.joint_names) == server.joint_names
    assert len(frame.joint_position) == 25
    np.testing.assert_allclose(frame.root_orientation_wxyz, (1.0, 0.0, 0.0, 0.0))


def test_reference_server_rejects_model_mismatch() -> None:
    with BelloReferenceServer(MODEL_PATH, listen="127.0.0.1:0") as server:
        channel, call = _watch(server, "0" * 64)
        try:
            with pytest.raises(grpc.RpcError) as error:
                next(call)
            assert error.value.code() == grpc.StatusCode.FAILED_PRECONDITION
        finally:
            call.cancel()
            channel.close()


def test_reference_server_rejects_nonpositive_timestamps() -> None:
    with BelloReferenceServer(MODEL_PATH, listen="127.0.0.1:0") as server:
        qpos = server.model.qpos0.copy()
        qpos[3] = 1.0
        with pytest.raises(ValueError, match="positive"):
            server.publish_qpos(qpos, source_monotonic_ns=0)
