from pathlib import Path
import math
import numpy as np
import mujoco
import mujoco.viewer
from loop_rate_limiters import RateLimiter

import mink

_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene_table.xml"

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(_XML.as_posix())

    configuration = mink.Configuration(model)
    feet = ["right_foot", "left_foot"]
    hands = ["right_palm", "left_palm"]

    tasks = [
        pelvis_orientation_task := mink.FrameTask(
            frame_name="pelvis",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        torso_orientation_task := mink.FrameTask(
            frame_name="torso_link",
            frame_type="body",
            position_cost=0.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        ),
        posture_task := mink.PostureTask(model, cost=1e-1),
        com_task := mink.ComTask(cost=10.0),
    ]

    # Feet tasks
    feet_tasks = []
    for foot in feet:
        task = mink.FrameTask(
            frame_name=foot,
            frame_type="site",
            position_cost=10.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        feet_tasks.append(task)
    tasks.extend(feet_tasks)

    # Hand tasks
    hand_tasks = []
    for hand in hands:
        task = mink.FrameTask(
            frame_name=hand,
            frame_type="site",
            position_cost=5.0,
            orientation_cost=1.0,
            lm_damping=1.0,
        )
        hand_tasks.append(task)
    tasks.extend(hand_tasks)

    # Collision avoidance
    collision_pairs = [
        (["left_hand_collision", "right_hand_collision"], ["table"]),
        (["left_hand_collision"], ["left_thigh"]),
        (["right_hand_collision"], ["right_thigh"]),
    ]
    collision_avoidance_limit = mink.CollisionAvoidanceLimit(
        model=model,
        geom_pairs=collision_pairs,  # type: ignore
        minimum_distance_from_collisions=0.05,
        collision_detection_distance=0.1,
    )

    limits = [
        mink.ConfigurationLimit(model),
        collision_avoidance_limit,
    ]

    com_mid = model.body("com_target").mocapid[0]
    feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
    hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

    model = configuration.model
    data = configuration.data
    solver = "daqp"

    with mujoco.viewer.launch_passive(
        model=model, data=data, show_left_ui=False, show_right_ui=False
    ) as viewer:
        mujoco.mjv_defaultFreeCamera(model, viewer.cam)

        # Initialize keyframe
        configuration.update_from_keyframe("teleop")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)

        # Initialize mocaps
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        rate = RateLimiter(frequency=200.0, warn=False)
        t = 0.0

        # Print info
        print("\n=== Task Dimensionality Summary (inferred) ===")
        for task in tasks:
            if isinstance(task, mink.FrameTask):
                pos_dim = 3 if getattr(task, "position_cost", 0.0) > 0 else 0
                ori_dim = 3 if getattr(task, "orientation_cost", 0.0) > 0 else 0
                total = pos_dim + ori_dim
                print(f"FrameTask on {getattr(task, 'frame_name', 'N/A')} → {total} DoF (pos={pos_dim}, ori={ori_dim})")
            elif isinstance(task, mink.ComTask):
                print("ComTask → 3 DoF (position only)")
            elif isinstance(task, mink.PostureTask):
                print(f"PostureTask → {model.nq} DoF (all joints)")
            else:
                print(f"{task.__class__.__name__} → Unknown dimension")

        current_com_pos = data.subtree_com[1]
        print(f"Current COM position: x={current_com_pos[0]:.3f}, y={current_com_pos[1]:.3f}, z={current_com_pos[2]:.3f}")

        # --- Set base hand position (down by sides) ---
        base_x = 0.0     # aligned with torso
        base_y = 0.25    # to the sides
        base_z = 0.6     # lower near thighs
        amplitude = 0.02 # small oscillation
        speed = 1.0

        while viewer.is_running():
            com_task.set_target(data.mocap_pos[com_mid])

            # --- Hands naturally hanging down ---
            data.mocap_pos[hands_mid[0]][0] = base_x + amplitude * math.sin(speed * t)
            data.mocap_pos[hands_mid[0]][1] = -base_y
            data.mocap_pos[hands_mid[0]][2] = base_z

            data.mocap_pos[hands_mid[1]][0] = base_x + amplitude * math.sin(speed * t + math.pi)
            data.mocap_pos[hands_mid[1]][1] = base_y
            data.mocap_pos[hands_mid[1]][2] = base_z

            # Update tasks
            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            # Torso gentle rotation
            t += rate.dt
            rotation_speed = 0.8
            rotation_amplitude = 0.4
            yaw_angle = rotation_amplitude * math.sin(rotation_speed * t)
            cos_half = math.cos(yaw_angle / 2)
            sin_half = math.sin(yaw_angle / 2)
            yaw_quat = np.array([cos_half, 0.0, 0.0, sin_half], dtype=np.float64)
            pos_zero = np.zeros(3, dtype=np.float64)

            torso_target = mink.SE3.from_rotation_and_translation(
                rotation=mink.SO3(yaw_quat),
                translation=pos_zero,
            )
            torso_orientation_task.set_target(torso_target)

            if int(t * 100) % 100 == 0:
                print(f"Torso yaw angle: {math.degrees(yaw_angle):.2f}°")

            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)

            mujoco.mj_camlight(model, data)
            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)
            viewer.sync()
            rate.sleep()
