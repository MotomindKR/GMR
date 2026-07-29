"""Versioned latest-frame gRPC stream for Bello retargeting output."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
from threading import Condition
import time
from types import TracebackType
from typing import Iterator, Literal

import grpc
import mujoco
import numpy as np

from . import reference_pb2


PROTOCOL_VERSION = 1
ReferenceMode = Literal["active", "hold", "stop"]
_MODE_VALUES = {
    "active": reference_pb2.REFERENCE_MODE_ACTIVE,
    "hold": reference_pb2.REFERENCE_MODE_HOLD,
    "stop": reference_pb2.REFERENCE_MODE_STOP,
}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def actuator_joint_names(model: mujoco.MjModel) -> tuple[str, ...]:
    names = []
    for actuator_id in range(model.nu):
        joint_id = int(model.actuator_trnid[actuator_id, 0])
        name = model.joint(joint_id).name
        if not name:
            raise ValueError(f"actuator {actuator_id} has no joint name")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("robot actuators contain duplicate joint names")
    return tuple(names)


@dataclass(frozen=True)
class _Sample:
    sequence: int
    source_monotonic_ns: int
    qpos: np.ndarray
    linear_velocity_local: np.ndarray
    angular_velocity_local: np.ndarray
    mode: ReferenceMode
    produced_at: float


class BelloReferenceServer:
    """Publishes retargeted Bello qpos through a latest-wins gRPC stream."""

    def __init__(
        self,
        robot_xml_path: str | Path,
        *,
        listen: str = "127.0.0.1:50053",
        source_id: str = "gmr",
        nominal_rate_hz: float = 50.0,
    ) -> None:
        self.robot_xml_path = Path(robot_xml_path).resolve()
        self.model = mujoco.MjModel.from_xml_path(str(self.robot_xml_path))
        if self.model.nq != 7 + self.model.nu or self.model.nv != 6 + self.model.nu:
            raise ValueError(
                "Bello reference model must contain one free joint and one "
                "one-DoF joint per actuator"
            )
        if not math.isfinite(nominal_rate_hz) or nominal_rate_hz <= 0.0:
            raise ValueError("nominal_rate_hz must be positive and finite")
        self.model_sha256 = file_sha256(self.robot_xml_path)
        self.joint_names = actuator_joint_names(self.model)
        self.source_id = source_id
        self.nominal_rate_hz = nominal_rate_hz
        self._condition = Condition()
        self._latest: _Sample | None = None
        self._previous_qpos: np.ndarray | None = None
        self._previous_source_ns: int | None = None
        self._sequence = 0
        self._closed = False
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._server = grpc.server(self._executor)
        handler = grpc.method_handlers_generic_handler(
            "bello.reference.v1.ReferenceService",
            {
                "WatchReference": grpc.unary_stream_rpc_method_handler(
                    self._watch_reference,
                    request_deserializer=reference_pb2.WatchReferenceRequest.FromString,
                    response_serializer=reference_pb2.WatchReferenceResponse.SerializeToString,
                )
            },
        )
        self._server.add_generic_rpc_handlers((handler,))
        self.port = self._server.add_insecure_port(listen)
        if self.port == 0:
            self._executor.shutdown(wait=False, cancel_futures=True)
            raise RuntimeError(f"cannot bind ReferenceService to {listen!r}")
        self._started = False

    def __enter__(self) -> "BelloReferenceServer":
        self._server.start()
        self._started = True
        return self

    def __exit__(
        self,
        _exception_type: type[BaseException] | None,
        _exception: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()
        if self._started:
            self._server.stop(grace=1.0).wait(timeout=2.0)
        self._executor.shutdown(wait=True, cancel_futures=True)

    def publish_qpos(
        self,
        qpos,
        *,
        source_monotonic_ns: int | None = None,
        mode: ReferenceMode = "active",
        produced_at: float | None = None,
    ) -> None:
        if mode not in _MODE_VALUES:
            raise ValueError(f"unsupported reference mode {mode!r}")
        qpos = np.asarray(qpos, dtype=np.float64)
        if qpos.shape != (self.model.nq,):
            raise ValueError(
                f"qpos must have shape {(self.model.nq,)}, got {qpos.shape}"
            )
        if not np.all(np.isfinite(qpos)):
            raise ValueError("qpos must be finite")
        quaternion_norm = float(np.linalg.norm(qpos[3:7]))
        if abs(quaternion_norm - 1.0) > 1.0e-3:
            raise ValueError("root quaternion must be normalized")
        now_ns = (
            time.monotonic_ns() if source_monotonic_ns is None else source_monotonic_ns
        )
        if now_ns <= 0:
            raise ValueError("source_monotonic_ns must be positive")
        timestamp = time.monotonic() if produced_at is None else produced_at
        if not math.isfinite(timestamp):
            raise ValueError("produced_at must be finite")

        velocity = np.zeros(self.model.nv, dtype=np.float64)
        if self._previous_qpos is not None and self._previous_source_ns is not None:
            dt = (now_ns - self._previous_source_ns) * 1.0e-9
            if dt <= 0.0:
                raise ValueError("source timestamps must increase")
            mujoco.mj_differentiatePos(
                self.model,
                velocity,
                dt,
                self._previous_qpos,
                qpos,
            )
        if mode != "active":
            velocity[:] = 0.0

        linear_local = _quat_rotate_inverse(qpos[3:7], velocity[:3])
        with self._condition:
            if self._closed:
                return
            self._sequence += 1
            self._latest = _Sample(
                sequence=self._sequence,
                source_monotonic_ns=now_ns,
                qpos=qpos.copy(),
                linear_velocity_local=linear_local,
                angular_velocity_local=velocity[3:6].copy(),
                mode=mode,
                produced_at=timestamp,
            )
            self._previous_qpos = qpos.copy()
            self._previous_source_ns = now_ns
            self._condition.notify_all()

    def _watch_reference(
        self,
        request: reference_pb2.WatchReferenceRequest,
        context: grpc.ServicerContext,
    ) -> Iterator[reference_pb2.WatchReferenceResponse]:
        if request.protocol_version != PROTOCOL_VERSION:
            context.abort(
                grpc.StatusCode.FAILED_PRECONDITION,
                f"protocol version {request.protocol_version} is unsupported",
            )
        if (
            request.robot_model_sha256
            and request.robot_model_sha256 != self.model_sha256
        ):
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "robot model mismatch")
        yield reference_pb2.WatchReferenceResponse(
            info=reference_pb2.ReferenceStreamInfo(
                protocol_version=PROTOCOL_VERSION,
                robot="bello",
                robot_model_sha256=self.model_sha256,
                joint_names=self.joint_names,
                source_id=self.source_id,
                nominal_rate_hz=self.nominal_rate_hz,
            )
        )
        last_sequence = 0
        while context.is_active():
            with self._condition:
                self._condition.wait_for(
                    lambda last_sequence=last_sequence: (
                        self._closed
                        or (
                            self._latest is not None
                            and self._latest.sequence != last_sequence
                        )
                    ),
                    timeout=0.5,
                )
                if self._closed:
                    return
                sample = self._latest
            if sample is None or sample.sequence == last_sequence:
                continue
            last_sequence = sample.sequence
            age_ms = max(0, int((time.monotonic() - sample.produced_at) * 1000.0))
            yield reference_pb2.WatchReferenceResponse(
                frame=reference_pb2.ReferenceFrame(
                    sequence=sample.sequence,
                    source_monotonic_ns=sample.source_monotonic_ns,
                    root_translation_metres=sample.qpos[:3],
                    root_orientation_wxyz=sample.qpos[3:7],
                    root_linear_velocity_local=sample.linear_velocity_local,
                    root_angular_velocity_local=sample.angular_velocity_local,
                    joint_position=sample.qpos[7:],
                    mode=_MODE_VALUES[sample.mode],
                    sample_age_ms=age_ms,
                )
            )


def _quat_rotate_inverse(quaternion_wxyz: np.ndarray, vector: np.ndarray) -> np.ndarray:
    w = quaternion_wxyz[0]
    xyz = quaternion_wxyz[1:]
    return (
        vector * (2.0 * w * w - 1.0)
        - 2.0 * w * np.cross(xyz, vector)
        + 2.0 * xyz * np.dot(xyz, vector)
    )
