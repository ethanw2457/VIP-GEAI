#!/usr/bin/env python3
"""
=================================================
Copyright (C) 2018 Vikash Kumar
Adapted by Raghava Uppuluri for GE-AI Course
=================================================
"""

from robohive.physics.sim_scene import SimScene
from robohive.utils.inverse_kinematics import qpos_from_site_pose
from robohive.utils.min_jerk import generate_joint_space_min_jerk
from robohive.utils.quat_math import euler2quat, quat2mat, mat2quat
from pick_n_place.utils.xml_utils import replace_simhive_path

from pathlib import Path
import click
import numpy as np

# Constants for bin and arm
BIN_POS = np.array([0.235, 0.5, 0.85])
BIN_DIM = np.array([0.2, 0.3, 0])
BIN_TOP = 0.10
ARM_nJnt = 7

DESC = """
DESCRIPTION: Simple Box Pick and Place
HOW TO RUN:
    - Ensure poetry shell is activated
    - python3 main.py
"""

@click.command(help=DESC)
@click.option(
    "-s",
    "--sim_path",
    type=str,
    help="environment to load",
    required=True,
    default="pick_place.xml",
)
@click.option("-h", "--horizon", type=int, help="time (s) to simulate", default=2)
def main(sim_path, horizon):
    # Prep
    sim_path = Path(__file__).parent.parent.absolute() / "env" / sim_path
    sim_xml = replace_simhive_path(str(sim_path))
    print(f"Loading {sim_xml}")
    sim = SimScene.get_sim(model_handle=sim_xml)

    # Setup simulation sites and bodies
    target_sid = sim.model.site_name2id("drop_target")
    box_sid = sim.model.body_name2id("box")
    eef_sid = sim.model.site_name2id("end_effector")
    # The pick target is defined in the XML as "pick_target"
    start_sid = sim.model.site_name2id("pick_target")

    # Initial joint configuration (home pose)
    ARM_JNT0 = np.mean(sim.model.jnt_range[:ARM_nJnt], axis=-1)

    while True:
        if sim.data.time == 0:
            print("Resampling new target")
            # Sample random drop target location
            target_pos = (
                BIN_POS
                + np.random.uniform(high=BIN_DIM, low=-BIN_DIM)
                + np.array([0, 0, BIN_TOP])
            )
            target_elr = np.random.uniform(high=[3.14, 0, 0], low=[3.14, 0, -3.14])
            target_quat = euler2quat(target_elr)

            # Get current box pose and pick target pose from the simulation
            box_pos = sim.data.xpos[box_sid]
            start_pos = sim.data.xpos[start_sid]
            start_mat = sim.data.xmat[start_sid]

            # Update drop target visualization
            sim.model.site_pos[target_sid][:] = target_pos - np.array([0, 0, BIN_TOP])
            sim.model.site_quat[target_sid][:] = target_quat

            # Reset arm to home for IK visualization
            sim.data.qpos[:ARM_nJnt] = ARM_JNT0
            sim.forward()

            # ---------------- 1. Pre-grasp: Move above the box ----------------
            ik_result = qpos_from_site_pose(
                physics=sim,
                site_name="end_effector",
                target_pos=box_pos + np.array([0, -0.01, 0.2]),
                target_quat=np.array([0, 1, 0, 0]),
                inplace=False,
                regularization_strength=1.0,
            )
            print("IK 1:", ik_result.success, ik_result.steps, ik_result.err_norm)
            waypoints = generate_joint_space_min_jerk(
                start=ARM_JNT0,
                goal=ik_result.qpos[:ARM_nJnt],
                time_to_go=horizon,
                dt=sim.model.opt.timestep,
            )

            # ---------------- 2. Approach: Move to pick target ----------------
            ik_result2 = qpos_from_site_pose(
                physics=sim,
                site_name="end_effector",
                target_pos=start_pos,
                target_quat=mat2quat(start_mat),
                inplace=False,
                regularization_strength=1.0,
            )
            waypoints2 = generate_joint_space_min_jerk(
                start=ik_result.qpos[:ARM_nJnt],
                goal=ik_result2.qpos[:ARM_nJnt],
                time_to_go=horizon,
                dt=sim.model.opt.timestep,
            )

            # ---------------- 3. Grasp: Lower to grasp the box ----------------
            ik_result3 = qpos_from_site_pose(
                physics=sim,
                site_name="end_effector",
                target_pos=start_pos - np.array([0, 0, 0.02]),
                target_quat=mat2quat(start_mat),
                inplace=False,
                regularization_strength=1.0,
            )
            waypoints3 = generate_joint_space_min_jerk(
                start=ik_result2.qpos[:ARM_nJnt],
                goal=ik_result3.qpos[:ARM_nJnt],
                time_to_go=horizon,
                dt=sim.model.opt.timestep,
            )

            # ---------------- 4. Lift: Raise the box after grasp ----------------
            ik_result4 = qpos_from_site_pose(
                physics=sim,
                site_name="end_effector",
                target_pos=start_pos + np.array([0, 0, 0.2]),
                target_quat=mat2quat(start_mat),
                inplace=False,
                regularization_strength=1.0,
            )
            waypoints4 = generate_joint_space_min_jerk(
                start=ik_result3.qpos[:ARM_nJnt],
                goal=ik_result4.qpos[:ARM_nJnt],
                time_to_go=horizon,
                dt=sim.model.opt.timestep,
            )

            # ---------------- 5. Transit: Move above drop target ----------------
            ik_result5 = qpos_from_site_pose(
                physics=sim,
                site_name="end_effector",
                target_pos=target_pos + np.array([0, 0, 0.2]),
                target_quat=target_quat,
                inplace=False,
                regularization_strength=1.0,
            )
            waypoints5 = generate_joint_space_min_jerk(
                start=ik_result4.qpos[:ARM_nJnt],
                goal=ik_result5.qpos[:ARM_nJnt],
                time_to_go=horizon,
                dt=sim.model.opt.timestep,
            )

            # ---------------- 6. Lower: Lower the box onto the drop target ----------------
            ik_result6 = qpos_from_site_pose(
                physics=sim,
                site_name="end_effector",
                target_pos=target_pos,
                target_quat=target_quat,
                inplace=False,
                regularization_strength=1.0,
            )
            waypoints6 = generate_joint_space_min_jerk(
                start=ik_result5.qpos[:ARM_nJnt],
                goal=ik_result6.qpos[:ARM_nJnt],
                time_to_go=horizon,
                dt=sim.model.opt.timestep,
            )

            # ---------------- 7. Release: Retract while releasing the box ----------------
            ik_result7 = qpos_from_site_pose(
                physics=sim,
                site_name="end_effector",
                target_pos=target_pos + np.array([0, 0, 0.2]),
                target_quat=target_quat,
                inplace=False,
                regularization_strength=1.0,
            )
            waypoints7 = generate_joint_space_min_jerk(
                start=ik_result6.qpos[:ARM_nJnt],
                goal=ik_result7.qpos[:ARM_nJnt],
                time_to_go=horizon,
                dt=sim.model.opt.timestep,
            )

            # ---------------- 8. Retract: Return to home ----------------
            waypoints8 = generate_joint_space_min_jerk(
                start=ik_result7.qpos[:ARM_nJnt],
                goal=ARM_JNT0,
                time_to_go=horizon,
                dt=sim.model.opt.timestep,
            )
            # ---------------- End of trajectory generation ----------------

        # Propagate the waypoints in simulation based on the elapsed time
        waypoint_ind = int(sim.data.time / sim.model.opt.timestep)
        total_steps = int(horizon / sim.model.opt.timestep)
        if waypoint_ind < total_steps:
            sim.data.ctrl[:ARM_nJnt] = waypoints[waypoint_ind]["position"]
            sim.data.ctrl[-1] = 1  # gripper open
        elif waypoint_ind < 2 * total_steps:
            sim.data.ctrl[:ARM_nJnt] = waypoints2[waypoint_ind - total_steps]["position"]
            sim.data.ctrl[-1] = 1
        elif waypoint_ind < 3 * total_steps:
            sim.data.ctrl[:ARM_nJnt] = waypoints3[waypoint_ind - 2 * total_steps]["position"]
            sim.data.ctrl[-1] = 0
            sim.data.ctrl[-2] = 0  # close gripper
        elif waypoint_ind < 4 * total_steps:
            sim.data.ctrl[:ARM_nJnt] = waypoints4[waypoint_ind - 3 * total_steps]["position"]
            sim.data.ctrl[-1] = 0
        elif waypoint_ind < 5 * total_steps:
            sim.data.ctrl[:ARM_nJnt] = waypoints5[waypoint_ind - 4 * total_steps]["position"]
            sim.data.ctrl[-1] = 0
        elif waypoint_ind < 6 * total_steps:
            sim.data.ctrl[:ARM_nJnt] = waypoints6[waypoint_ind - 5 * total_steps]["position"]
            sim.data.ctrl[-1] = 0
        elif waypoint_ind < 7 * total_steps:
            sim.data.ctrl[:ARM_nJnt] = waypoints7[waypoint_ind - 6 * total_steps]["position"]
            sim.data.ctrl[-1] = 0.4
            sim.data.ctrl[-2] = 0.4
        else:
            sim.data.ctrl[:ARM_nJnt] = waypoints8[waypoint_ind - 7 * total_steps]["position"]
            sim.data.ctrl[-1] = 0.4

        sim.advance(render=True)

        # After all segments are executed, check error and reset simulation.
        if sim.data.time > 8 * horizon:
            box_pos = sim.data.xpos[box_sid]
            distance = np.linalg.norm(target_pos - box_pos)
            print(f"Error distance is {distance}")
            sim.reset()

if __name__ == "__main__":
    main()
