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

    # Enable collision avoidance between the following geoms.
    # left hand - table, right hand - table
    # left hand - left thigh, right hand - right thigh
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

        # Initialize to the home keyframe.
        configuration.update_from_keyframe("teleop")
        posture_task.set_target_from_configuration(configuration)
        pelvis_orientation_task.set_target_from_configuration(configuration)
        torso_orientation_task.set_target_from_configuration(configuration)
        # Initialize mocap bodies at their respective sites.
        for hand, foot in zip(hands, feet):
            mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
            mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
        data.mocap_pos[com_mid] = data.subtree_com[1]

        rate = RateLimiter(frequency=200.0, warn=False)
        t = 0.0

        # print what data I can control
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

        # 🔹 Print current COM base position (world frame)
        current_com_pos = data.subtree_com[1]  # The robot’s COM (body 1 = floating base)
        print(f"Current COM position: x={current_com_pos[0]:.3f}, y={current_com_pos[1]:.3f}, z={current_com_pos[2]:.3f}")

        while viewer.is_running():
            com_task.set_target(data.mocap_pos[com_mid])

            for i, (hand_task, foot_task) in enumerate(zip(hand_tasks, feet_tasks)):
                foot_task.set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))
                hand_task.set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

            # --- 🥊 稳定 Boxing 动作 ---
            t += rate.dt
            amplitude = 0.10       # 出拳距离（小一点）
            base_x = 0.50          # 手初始前后位置
            base_y = 0.15          # 手的左右间距
            base_z = 0.9           # 高度
            speed = 2.5            # 节奏（2.5 比较自然）

            # --- ⚖️ 质心（COM）左右轻微移动 ---
            com_radius = 0.01   # 左右移动幅度（约 3 cm）
            com_speed = 1.5     # 频率
            data.mocap_pos[com_mid][1] = com_radius * math.sin(com_speed * t)

            # 右手
            data.mocap_pos[hands_mid[0]][0] = base_x + amplitude * math.sin(speed * t)
            data.mocap_pos[hands_mid[0]][1] = -base_y
            data.mocap_pos[hands_mid[0]][2] = base_z

            # 左手（反相）
            data.mocap_pos[hands_mid[1]][0] = base_x + amplitude * math.sin(speed * t + math.pi)
            data.mocap_pos[hands_mid[1]][1] = base_y
            data.mocap_pos[hands_mid[1]][2] = base_z

            # --- 手掌朝下固定方向 ---
            # 让手掌朝下：绕 X 轴旋转 180 度（面朝地面）
            down_angle = math.pi  # 180度
            cos_half = math.cos(down_angle / 2)
            sin_half = math.sin(down_angle / 2)
            # 四元数 (w, x, y, z) — 绕 X 轴旋转 180°
            palm_down_quat = np.array([cos_half, sin_half, 0, 0])

            # 应用到两只手
            for h in hands_mid:
                data.mocap_quat[h] = palm_down_quat

            # --- 躯干轻微转动（自然） ---
            pelvis_angle = 0.05 * math.sin(speed * t)
            data.qpos[3:7] = np.array([
                math.cos(pelvis_angle / 2), 0, 0, math.sin(pelvis_angle / 2)
            ])

            # --- 求解并应用 IK ---
            vel = mink.solve_ik(
                configuration, tasks, rate.dt, solver, 1e-1, limits=limits
            )
            configuration.integrate_inplace(vel, rate.dt)
            mujoco.mj_camlight(model, data)

            mujoco.mj_fwdPosition(model, data)
            mujoco.mj_sensorPos(model, data)
            viewer.sync()
            rate.sleep()


