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

"""Tests for ``ConnectionSource``: reading messages in, complete commands out.

The drivers here name their ports their own way ("target_angles",
"angles", ...), so the tests show each message goes to and comes from the
driver's own port.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from dimos.control._control_test_helpers import turning_joints
from dimos.control.connection.connection_module import DriverPorts
from dimos.control.connection_source import ConnectionSource
from dimos.control.contract.description import ControlDescription, Limits, Resource, ResourceKind
from dimos.control.contract.keys import Interface, Key, Unit
from dimos.control.contract.ports import JointPort, MotorPort, TwistPort
from dimos.control.contract.presets import imu_resource, twist_base_description
from dimos.hardware.manipulators.spec import ControlMode
from dimos.msgs.geometry_msgs.PoseStamped import PoseStamped
from dimos.msgs.geometry_msgs.Quaternion import Quaternion
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.msgs.sensor_msgs.Imu import Imu
from dimos.msgs.sensor_msgs.JointState import JointState

J1 = "arm/joint1"
J2 = "arm/joint2"
GRIPPER = "arm/gripper"

#: The arm driver's ports: positions and speeds in, joint readings out.
ARM_COMMANDS = {
    "target_angles": JointPort(),
    "spin_rates": JointPort(interface=Interface.VELOCITY),
}
ARM_READINGS: dict[type, str] = {JointState: "angles"}


def arm(**kwargs: Any) -> ControlDescription:
    """A two-joint arm that reports position and takes position or velocity,
    limited to +-1 rad."""
    joints = ["joint1", "joint2"]
    settings: dict[str, Any] = {
        "limits": {Key.of("arm", j, Interface.POSITION): Limits(-1.0, 1.0) for j in joints},
        "state": (Interface.POSITION,),
        "command": (Interface.POSITION, Interface.VELOCITY),
    }
    settings.update(kwargs)
    return turning_joints("arm", joints, **settings)


def arm_with_gripper() -> ControlDescription:
    """A one-joint arm that reports position and effort, and a gripper that
    reports only its position."""
    gripper = Resource(
        name="gripper",
        kind=ResourceKind.JOINT,
        state_interfaces=(Interface.POSITION,),
        command_interfaces=(Interface.POSITION,),
        units={Interface.POSITION: Unit.M},
    )
    return turning_joints(
        "arm",
        ["joint1"],
        state=(Interface.POSITION, Interface.EFFORT),
        command=(Interface.POSITION, Interface.VELOCITY),
        limits={Key.of("arm", "gripper", Interface.POSITION): Limits(0.0, 0.1)},
        others=[gripper],
    )


def arm_driver(description: ControlDescription | None = None) -> DriverPorts:
    """The arm behind a driver with ARM_COMMANDS and ARM_READINGS."""
    return DriverPorts(description or arm(), ARM_COMMANDS, ARM_READINGS)


def base_driver() -> DriverPorts:
    """A base called "go2" that takes speeds on "cmd_vel" and reports where it is on "odom"
    and how fast it goes on "speed"."""
    limits: dict[str, Limits] = {
        Key.of("go2", "base", axis): Limits(-1.0, 1.0)
        for axis in (Interface.VX, Interface.VY, Interface.WZ)
    }
    return DriverPorts(
        twist_base_description("go2", limits=limits),
        {"cmd_vel": TwistPort(range=Twist(Vector3(1.0, 1.0, 0.0), Vector3(0.0, 0.0, 1.0)))},
        {PoseStamped: "odom", Twist: "speed"},
    )


def reading(*positions: float) -> JointState:
    """The arm's joint positions, joint1 first."""
    return JointState(name=[J1, J2][: len(positions)], position=list(positions))


def source_at(*positions: float, description: ControlDescription | None = None) -> ConnectionSource:
    """A source for the two-joint arm that has had one reading."""
    source = ConnectionSource(arm_driver(description))
    source.on_message("angles", reading(*positions))
    return source


def test_it_uses_the_drivers_own_port_names() -> None:
    arm_source = ConnectionSource(arm_driver())
    assert arm_source.command_ports == ["target_angles", "spin_rates"]
    assert arm_source.reading_ports == ["angles"]

    base = ConnectionSource(base_driver())
    assert base.command_ports == ["cmd_vel"]
    assert base.reading_ports == ["odom", "speed"]


def test_joints_and_sensors_come_from_readings() -> None:
    driver = DriverPorts(
        arm(others=[imu_resource()]), ARM_COMMANDS, {JointState: "angles", Imu: "tilt"}
    )
    source = ConnectionSource(driver)
    source.on_message("angles", reading(0.3, -0.2))
    source.on_message("tilt", Imu(orientation=Quaternion(0.0, 0.0, 0.0, 1.0)))

    joints = source.read_joints()
    assert joints.joint_positions == {J1: 0.3, J2: -0.2}
    # Not reported, so left out rather than read as 0.0.
    assert joints.get_velocity(J1) is None
    sensors = source.read_sensors()
    assert list(sensors) == ["arm/imu"]
    assert sensors["arm/imu"]["qw"] == 1.0
    assert len(sensors["arm/imu"]) == 10


def test_every_joint_state_message_counts() -> None:
    source = ConnectionSource(arm_driver(arm_with_gripper()))

    source.on_message("angles", JointState(name=[J1], position=[0.3], effort=[1.5]))
    assert not source.ready_for_control()
    source.on_message("angles", JointState(name=[GRIPPER], position=[0.05]))

    assert source.ready_for_control()
    joints = source.read_joints()
    assert joints.joint_positions == {J1: 0.3, GRIPPER: 0.05}
    assert joints.joint_efforts == {J1: 1.5}


def test_other_robots_and_bad_messages_are_ignored() -> None:
    source = source_at(0.3, 0.0)

    source.on_message("angles", JointState(name=["other/joint1"], position=[0.9]))
    source.on_message("angles", JointState(name=[J1, J2], position=[0.9]))

    assert source.read_joints().joint_positions == {J1: 0.3, J2: 0.0}


def test_not_ready_until_everything_has_reported() -> None:
    source = ConnectionSource(arm_driver())
    assert not source.ready_for_control()

    source.on_message("angles", reading(0.3))
    assert not source.ready_for_control()

    source.on_message("angles", reading(0.3, 0.0))
    assert source.ready_for_control()


def test_with_no_task_every_joint_is_held_where_it_is() -> None:
    source = source_at(0.3, -0.2)

    assert source.command({}, None) == {f"{J1}/position": 0.3, f"{J2}/position": -0.2}


def test_task_values_are_sent_and_the_other_joints_held() -> None:
    source = source_at(0.3, -0.2)

    assert source.command({J1: 0.5}, ControlMode.POSITION) == {
        f"{J1}/position": 0.5,
        f"{J2}/position": -0.2,
    }

    # Once the task stops, the joint stays at what it was last told, not
    # wherever it has got to.
    source.on_message("angles", reading(0.45, -0.2))
    assert source.command({}, None)[f"{J1}/position"] == 0.5


def test_a_velocity_tick_zeroes_the_other_joints_then_holds_where_they_stopped() -> None:
    source = source_at(0.3, -0.2)
    source.command({J1: 0.1}, ControlMode.POSITION)

    assert source.command({J1: 0.2}, ControlMode.VELOCITY) == {
        f"{J1}/velocity": 0.2,
        f"{J2}/velocity": 0.0,
    }

    source.on_message("angles", reading(0.6, -0.1))
    assert source.command({}, None) == {f"{J1}/position": 0.6, f"{J2}/position": -0.1}


def test_a_held_position_stays_inside_the_limit_but_task_values_are_sent_as_is() -> None:
    source = source_at(1.0001, 0.0)

    assert source.command({}, None)[f"{J1}/position"] == 1.0
    assert source.command({J1: 1.5}, ControlMode.POSITION)[f"{J1}/position"] == 1.5
    assert source.command({}, None)[f"{J1}/position"] == 1.0


def test_a_joint_that_cannot_take_a_position_gets_zero_velocity() -> None:
    source = source_at(0.3, 0.0, description=arm(command=(Interface.VELOCITY,), limits={}))

    assert source.command({}, None) == {f"{J1}/velocity": 0.0, f"{J2}/velocity": 0.0}


def test_a_task_driving_a_joint_in_a_way_it_does_not_accept_raises() -> None:
    source = source_at(0.3, 0.0, description=arm(command=(Interface.POSITION,)))

    with pytest.raises(ValueError, match="velocity"):
        source.command({J1: 0.2}, ControlMode.VELOCITY)


def test_each_kind_of_command_goes_to_its_own_port() -> None:
    source = ConnectionSource(arm_driver(arm_with_gripper()))
    source.on_message("angles", JointState(name=[J1], position=[0.3], effort=[0.0]))
    source.on_message("angles", JointState(name=[GRIPPER], position=[0.05]))

    values = source.command({J1: 0.2}, ControlMode.VELOCITY)
    messages = dict(source.command_messages(values, ts=1.0))

    assert list(messages) == ["target_angles", "spin_rates"]
    assert list(messages["target_angles"].name) == [GRIPPER]
    assert list(messages["target_angles"].position) == [0.05]
    assert list(messages["spin_rates"].name) == [J1]
    assert list(messages["spin_rates"].velocity) == [0.2]


def test_a_base_is_told_zero_on_every_axis_on_its_port() -> None:
    source = ConnectionSource(base_driver())
    source.on_message("odom", PoseStamped())
    source.on_message("speed", Twist())

    values = source.command({}, None)
    (port, twist), *rest = source.command_messages(values, ts=1.0)

    assert source.ready_for_control()
    assert values == {"go2/base/vx": 0.0, "go2/base/vy": 0.0, "go2/base/wz": 0.0}
    assert port == "cmd_vel" and rest == []
    assert (twist.linear.x, twist.linear.y, twist.angular.z) == (0.0, 0.0, 0.0)
    assert source.joint_names == []


def test_a_robot_that_goes_quiet_is_held_where_it_is_when_it_comes_back() -> None:
    source = source_at(0.3, 0.0, description=arm(deadman_timeout_s=0.05))
    source.command({J1: 0.5}, ControlMode.POSITION)

    time.sleep(0.08)
    assert not source.ready_for_control()
    # Its joints stay in what tasks see.
    assert source.read_joints().joint_positions[J1] == 0.3

    # Back with only part of a reading: it waits for the rest.
    source.on_message("angles", reading(0.8))
    assert not source.ready_for_control()
    source.on_message("angles", reading(0.8, 0.0))
    assert source.ready_for_control()
    assert source.command({}, None)[f"{J1}/position"] == 0.8


def test_a_robot_it_cannot_drive_is_refused() -> None:
    effort_only = turning_joints(
        "arm", ["joint1"], state=(Interface.POSITION,), command=(Interface.EFFORT,)
    )
    unmeasured = turning_joints(
        "arm", ["joint1"], state=(Interface.EFFORT,), command=(Interface.POSITION,)
    )

    with pytest.raises(ValueError, match="joint1"):
        ConnectionSource(arm_driver(effort_only))
    with pytest.raises(ValueError, match="joint1"):
        ConnectionSource(arm_driver(unmeasured))
    with pytest.raises(ValueError, match="reports nothing"):
        ConnectionSource(DriverPorts(arm(), ARM_COMMANDS, {}))
    with pytest.raises(ValueError, match="'legs'.*stiffness"):
        ConnectionSource(DriverPorts(arm(), {"legs": MotorPort()}, ARM_READINGS))
    with pytest.raises(ValueError, match="'left' and 'right'"):
        ConnectionSource(
            DriverPorts(arm(), {"left": JointPort(), "right": JointPort()}, ARM_READINGS)
        )
