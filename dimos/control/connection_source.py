# Copyright 2025-2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""The coordinator's record of one robot it drives through a connection module.

A connection module reports a robot's readings and takes its commands as
standard messages (``JointState``, ``Twist``, ``PoseStamped``, ``Imu``) on its
own ports. The coordinator keeps one ``ConnectionSource`` per robot. It turns
each reading into named values with the shared converters in
``dimos.control.contract.convert``, keeps the latest, and builds each tick's
complete command and the messages that carry it, each addressed to the
driver's port for it.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import threading
import time
from typing import TYPE_CHECKING, Any

from dimos.control.contract.convert import (
    JOINT_STATE_FIELDS,
    TWIST_INTERFACES,
    imu_to_values,
    joint_state_from_values,
    joint_state_to_values,
    pose_to_values,
    twist_from_values,
    twist_to_values,
)
from dimos.control.contract.description import Limits, ResourceKind
from dimos.control.contract.keys import SEPARATOR, Interface
from dimos.control.contract.ports import JointPort, MotorPort, TwistPort
from dimos.control.contract.sequence import Freshness
from dimos.control.task import JointStateSnapshot
from dimos.hardware.manipulators.spec import ControlMode
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState
from dimos.utils.logging_config import setup_logger

if TYPE_CHECKING:
    from dimos.control.connection.connection_module import DriverPorts

logger = setup_logger()

Values = dict[str, float]

# Which part of a joint each task control mode drives.
_INTERFACE_BY_MODE: dict[ControlMode, str] = {
    ControlMode.POSITION: Interface.POSITION,
    ControlMode.SERVO_POSITION: Interface.POSITION,
    ControlMode.VELOCITY: Interface.VELOCITY,
    ControlMode.TORQUE: Interface.EFFORT,
}


@dataclass(frozen=True, slots=True)
class PortMessage:
    """One command message and the driver port it goes to.

    Attributes:
        port: The driver's input port name, e.g. "cmd_vel".
        message: The message to send on it, e.g. a ``Twist``.
    """

    port: str
    message: Any


@dataclass(frozen=True, slots=True)
class _SensorReading:
    """One number a sensor reports.

    Attributes:
        sensor: The sensor's full name, e.g. "g1/imu".
        reading: What it measures, e.g. "qw".
        key: Its full key, e.g. "g1/imu/qw".
    """

    sensor: str
    reading: str
    key: str


@dataclass(frozen=True, slots=True)
class _Joint:
    """One joint's name and the keys it reports and accepts."""

    name: str
    state_keys: dict[str, str]
    command_keys: dict[str, str]
    position_limit: Limits | None


class ConnectionSource:
    """One robot behind a connection module, as the coordinator's tick loop sees it.

    Joints are named by robot and part, e.g. "arm/joint1", the same names
    tasks claim. Each tick the coordinator sends the robot a complete
    command: what tasks asked for, and for every other joint:

    - in a tick where tasks drive this robot by velocity: velocity 0;
    - otherwise: hold its position. That is the last position a task sent,
      or, if none did, where the joint was when the hold began. A held
      position is kept inside the joint's position limit. A joint that
      cannot be held at a position it reports gets velocity 0 instead.

    A base is told 0 on every axis. Sensors are never told anything.

    Commands go out as one message per driver port: a ``JointState`` on each
    joint port, a ``Twist`` on the base port. What tasks ask for is sent as
    is, even past a limit; the connection module then refuses that whole
    message.

    A robot whose readings stop for longer than its deadman timeout is no
    longer commanded; its connection's deadman halts it. When its readings
    come back, e.g. after its driver restarted, it is held where it then is.

    Does no I/O: the coordinator feeds it messages and publishes what it
    builds.

    Args:
        driver: The robot and its driver's ports, from
            ``ConnectionModule.driver_ports``.

    Raises:
        ValueError: If the coordinator cannot drive the robot: it reports
            nothing, so the coordinator cannot tell it is there; a joint can
            neither be told a velocity nor held at a position it reports (an
            arm left with effort 0 falls under gravity); it has a motor port,
            which needs stiffness and damping the coordinator has no source
            for; or it has two joint ports of one kind, e.g. two position
            ports.
    """

    def __init__(self, driver: DriverPorts) -> None:
        description = driver.description
        source = description.source
        if not driver.readings:
            raise ValueError(
                f"{source!r} reports nothing, so the coordinator cannot tell it is there; "
                "its driver needs a reading output, e.g. `Out[JointState]`"
            )
        unholdable = [
            r.name
            for r in description.resources
            if r.kind is ResourceKind.JOINT
            and Interface.VELOCITY not in r.command_interfaces
            and not (
                Interface.POSITION in r.command_interfaces
                and Interface.POSITION in r.state_interfaces
            )
        ]
        if unholdable:
            raise ValueError(
                f"{source!r}: joint(s) {unholdable} cannot be held when no task drives "
                "them; each needs to take a velocity, or take a position it reports"
            )

        self._description = description
        self._state_keys = frozenset(description.state_keys())
        self._joint_ports: dict[str, str] = {}
        self._base_port: str | None = None
        for port, spec in driver.commands.items():
            if isinstance(spec, MotorPort):
                raise ValueError(
                    f"{source!r}: port {port!r} needs stiffness and damping, which the "
                    "coordinator cannot send"
                )
            if isinstance(spec, TwistPort):
                self._base_port = port
            elif isinstance(spec, JointPort):
                if spec.interface in self._joint_ports:
                    raise ValueError(
                        f"{source!r}: ports {self._joint_ports[spec.interface]!r} and {port!r} "
                        f"both take joint {spec.interface}s; the coordinator sends each kind "
                        "of joint command on one port"
                    )
                self._joint_ports[spec.interface] = port

        self._joints: list[_Joint] = []
        self._sensors: list[_SensorReading] = []
        self._base: str | None = None
        imu: str | None = None
        for resource in description.resources:
            part = f"{source}{SEPARATOR}{resource.name}"
            if resource.kind is ResourceKind.JOINT:
                self._joints.append(
                    _Joint(
                        name=part,
                        state_keys={i: f"{part}{SEPARATOR}{i}" for i in resource.state_interfaces},
                        command_keys={
                            i: f"{part}{SEPARATOR}{i}" for i in resource.command_interfaces
                        },
                        position_limit=description.limits.get(
                            f"{part}{SEPARATOR}{Interface.POSITION}"
                        ),
                    )
                )
            elif resource.kind is ResourceKind.BASE:
                self._base = part
            else:
                imu = part
                self._sensors += [
                    _SensorReading(part, i, f"{part}{SEPARATOR}{i}")
                    for i in sorted(resource.state_interfaces)
                ]

        self._readers: dict[str, Callable[[Any], Values]] = {}
        for message, port in driver.readings.items():
            if message is JointState:
                self._readers[port] = _read_joint_state
            elif message is PoseStamped and self._base is not None:
                self._readers[port] = _reader(pose_to_values, self._base)
            elif message is Twist and self._base is not None:
                self._readers[port] = _reader(twist_to_values, self._base)
            elif message is Imu and imu is not None:
                self._readers[port] = _reader(imu_to_values, imu)

        # Guards the reading and the flag below, which readings change and
        # the tick loop reads.
        self._lock = threading.Lock()
        self._latest: Values = {}
        self._received_at = 0.0
        # When the last reading arrived, by time.monotonic().
        self._freshness = Freshness()
        # Readings came back after stopping for longer than the deadman
        # timeout. The tick loop drops the holds when it sees this.
        self._resumed = False
        # Joint name -> the position it is held at. Only the tick loop touches it.
        self._holds: dict[str, float] = {}
        self._last_warned = float("-inf")

    @property
    def hardware_id(self) -> str:
        """The robot's name, e.g. "arm"."""
        return self._description.source

    @property
    def joint_names(self) -> list[str]:
        """Every joint, e.g. ["arm/joint1", "arm/joint2"]."""
        return [joint.name for joint in self._joints]

    @property
    def command_ports(self) -> list[str]:
        """The driver's ports the coordinator sends commands on, e.g. ["position_command"]."""
        ports = list(self._joint_ports.values())
        if self._base_port is not None:
            ports.append(self._base_port)
        return ports

    @property
    def reading_ports(self) -> list[str]:
        """The driver's ports the coordinator takes readings from, e.g. ["joint_state"]."""
        return list(self._readers)

    def on_message(self, port: str, msg: Any) -> None:
        """Take one reading message from the driver's ``port``, e.g. "joint_state".

        Every message counts: a driver may send several per tick, e.g. one for
        joints that report effort and one for a gripper that does not. Values
        for other robots, which share a topic with this one, are ignored. A
        message that cannot be read is dropped and logged at most once a
        second; if they keep coming, the robot's readings go stale and it is
        no longer commanded.
        """
        try:
            values = self._readers[port](msg)
        except ValueError as error:
            now = time.monotonic()
            if now - self._last_warned >= 1.0:
                self._last_warned = now
                logger.warning(f"Dropped a {port} message for {self.hardware_id!r}: {error}")
            return
        mine = {key: value for key, value in values.items() if key in self._state_keys}
        if not mine:
            return
        now = time.monotonic()
        with self._lock:
            latest = self._latest
            last = self._freshness.last_receipt
            if last is not None and now - last > self._description.deadman_timeout_s:
                # Back after a silence: wait for a whole new reading, and take
                # the holds again from it.
                latest = {}
                self._resumed = True
            # Replaced whole, so the tick loop never sees half of one update.
            self._latest = {**latest, **mine}
            self._received_at = time.time()
            self._freshness.mark(now)

    def ready_for_control(self, now: float | None = None) -> bool:
        """Whether a command can be sent: everything the robot reports has
        arrived, and its last reading is no older than its deadman timeout.

        Args:
            now: The current ``time.monotonic()``, in seconds. Defaults to now.
        """
        if now is None:
            now = time.monotonic()
        with self._lock:
            return len(self._latest) == len(self._state_keys) and not self._freshness.is_stale(
                now, self._description.deadman_timeout_s
            )

    def read_joints(self) -> JointStateSnapshot:
        """The joints' latest positions, velocities and efforts.

        A joint that does not report something is left out of that lookup,
        rather than given a zero that looks like a real reading.
        """
        latest = self._latest
        snapshot = JointStateSnapshot(timestamp=self._received_at)
        for joint in self._joints:
            for interface, into in (
                (Interface.POSITION, snapshot.joint_positions),
                (Interface.VELOCITY, snapshot.joint_velocities),
                (Interface.EFFORT, snapshot.joint_efforts),
            ):
                key = joint.state_keys.get(interface)
                if key is not None and key in latest:
                    into[joint.name] = latest[key]
        return snapshot

    def read_sensors(self) -> dict[str, dict[str, float]]:
        """Each sensor's latest readings, e.g. ``{"g1/imu": {"qw": 1.0, ...}}``."""
        latest = self._latest
        readings: dict[str, dict[str, float]] = {}
        for each in self._sensors:
            if each.key in latest:
                readings.setdefault(each.sensor, {})[each.reading] = latest[each.key]
        return readings

    def command(self, winners: Mapping[str, float], mode: ControlMode | None) -> Values:
        """The complete command to send the robot this tick, by key.

        Args:
            winners: What tasks asked for, by joint name, e.g.
                ``{"arm/joint1": 0.3}``. Only this robot's joints.
            mode: How the winners drive their joints. ``None`` when no task
                drives this robot this tick.

        Raises:
            ValueError: If a task drives a joint in a way it does not accept,
                e.g. by velocity when it only takes a position. The robot then
                gets no command this tick.
        """
        with self._lock:
            latest = self._latest
            if self._resumed:
                self._resumed = False
                self._holds.clear()
        interface = _INTERFACE_BY_MODE.get(mode) if mode is not None else None
        values: Values = {}
        for joint in self._joints:
            if joint.name in winners:
                if interface is None or interface not in joint.command_keys:
                    raise ValueError(f"{joint.name} cannot be told its {interface}")
                value = winners[joint.name]
                values[joint.command_keys[interface]] = value
                if interface == Interface.POSITION:
                    self._holds[joint.name] = value
                else:
                    self._holds.pop(joint.name, None)
                continue
            velocity_key = joint.command_keys.get(Interface.VELOCITY)
            if mode is ControlMode.VELOCITY and velocity_key is not None:
                values[velocity_key] = 0.0
                self._holds.pop(joint.name, None)
                continue
            hold = self._hold(joint, latest)
            if hold is not None:
                values[joint.command_keys[Interface.POSITION]] = hold
            elif velocity_key is not None:
                values[velocity_key] = 0.0
        if self._base is not None and self._base_port is not None:
            values.update((f"{self._base}{SEPARATOR}{i}", 0.0) for i in TWIST_INTERFACES)
        return values

    def command_messages(self, values: Mapping[str, float], ts: float) -> list[PortMessage]:
        """The messages that carry ``values``, each with the driver's port to send it on.

        One per port: a ``JointState`` on each joint port, of the joints told
        that port's kind of value, and a ``Twist`` on the base port.

        Args:
            values: A command from ``command``.
            ts: When it was made, in Unix seconds.
        """
        messages: list[PortMessage] = []
        for interface, port in self._joint_ports.items():
            joints = [j.name for j in self._joints if f"{j.name}{SEPARATOR}{interface}" in values]
            if joints:
                messages.append(PortMessage(port, joint_state_from_values(values, joints, ts)))
        if self._base is not None and self._base_port is not None:
            messages.append(PortMessage(self._base_port, twist_from_values(values, self._base)))
        return messages

    def _hold(self, joint: _Joint, latest: Values) -> float | None:
        """Where to hold ``joint``, inside its position limit, or ``None`` if
        it cannot be told a position or its position is unknown.

        Args:
            joint: The joint to hold.
            latest: The robot's latest reading, to start a new hold from.
        """
        if Interface.POSITION not in joint.command_keys:
            return None
        held = self._holds.get(joint.name)
        if held is None:
            measured_key = joint.state_keys.get(Interface.POSITION)
            if measured_key is None or measured_key not in latest:
                return None
            held = latest[measured_key]
            self._holds[joint.name] = held
        limit = joint.position_limit
        if limit is not None:
            if limit.lo is not None:
                held = max(held, limit.lo)
            if limit.hi is not None:
                held = min(held, limit.hi)
        return held


def _read_joint_state(msg: Any) -> Values:
    return joint_state_to_values(msg, JOINT_STATE_FIELDS)


def _reader(convert: Callable[[Any, str], Values], part: str) -> Callable[[Any], Values]:
    """``convert(msg, part)`` as a function of the message alone."""
    return lambda msg: convert(msg, part)
