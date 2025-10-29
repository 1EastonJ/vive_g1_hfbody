#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import socket
import orjson   # ⚡ faster JSON
from loop_rate_limiters import RateLimiter
import mink
import time

# --------------------------------------------------------------------------
# ✅ Network setup
# --------------------------------------------------------------------------
UDP_IP = "0.0.0.0"
PORT = 5005
print(f"📡 Listening for VR data on UDP {UDP_IP}:{PORT}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, PORT))
sock.settimeout(0.01)
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1048576)  # ✅ 1 MB buffer

tracker_data = {"left": None, "right": None}

# --------------------------------------------------------------------------#!/usr/bin/env python3
# -*- coding:utf-8 -*-
from pathlib import Path
import numpy as np
import mujoco
import mujoco.viewer
import socket
import orjson
import time
from loop_rate_limiters import RateLimiter
import mink

# --------------------------------------------------------------------------
# ✅ Network setup
# --------------------------------------------------------------------------
UDP_IP = "0.0.0.0"
PORT = 5005
print(f"📡 Listening for VR data on UDP {UDP_IP}:{PORT}")

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, PORT))
sock.setblocking(False)  # 🚀 非阻塞``
sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 8192)  # 小缓冲防堆积

tracker_data = {"left": None, "right": None}

def recv_latest_dual(sock):
    """分别读取左右手最新的包"""
    latest = {"left": None, "right": None}
    while True:
        try:
            msg, _ = sock.recvfrom(512)
            vr_data = orjson.loads(msg)
            role = vr_data.get("role")
            if role in latest:
                latest[role] = vr_data
        except (BlockingIOError, socket.timeout, ValueError):
            break
    return latest

# --------------------------------------------------------------------------
# 🦿 MuJoCo setup
# --------------------------------------------------------------------------
_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene_table.xml"

model = mujoco.MjModel.from_xml_path(_XML.as_posix())
configuration = mink.Configuration(model)

feet = ["right_foot", "left_foot"]
hands = ["right_palm", "left_palm"]

tasks = [
    mink.FrameTask("pelvis", "body", 0.0, 1.0, 1.0),
    mink.FrameTask("torso_link", "body", 0.0, 1.0, 1.0),
    mink.PostureTask(model, cost=1e-1),
    mink.ComTask(cost=10.0),
]

feet_tasks = [mink.FrameTask(f, "site", 10.0, 1.0, 1.0) for f in feet]
hand_tasks = [mink.FrameTask(h, "site", 5.0, 1.0, 1.0) for h in hands]
tasks += feet_tasks + hand_tasks

collision_pairs = [
    (["left_hand_collision", "right_hand_collision"], ["table"]),
    (["left_hand_collision"], ["left_thigh"]),
    (["right_hand_collision"], ["right_thigh"]),
]
collision_avoidance_limit = mink.CollisionAvoidanceLimit(
    model,
    geom_pairs=collision_pairs,
    minimum_distance_from_collisions=0.05,
    collision_detection_distance=0.1,
)
limits = [mink.ConfigurationLimit(model), collision_avoidance_limit]

com_mid = model.body("com_target").mocapid[0]
feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

data = configuration.data
solver = "daqp"

# --------------------------------------------------------------------------
# 📊 Profiling metrics
# --------------------------------------------------------------------------
last_print = time.time()
count = 0
ik_time_accum = 0.0
frame_count = 0

# --------------------------------------------------------------------------
# 🌊 平滑缓冲区
# --------------------------------------------------------------------------
window_size = 6
smooth_buffer = {"left": [], "right": []}

def smooth_update(role, new_data):
    """加权平均平滑（去跳帧 + 稳定轨迹）"""
    pos = np.array(new_data["pos"])
    quat = np.array(new_data["quat"])
    smooth_buffer[role].append((pos, quat))
    if len(smooth_buffer[role]) > window_size:
        smooth_buffer[role].pop(0)

    weights = np.linspace(0.3, 1.0, len(smooth_buffer[role]))
    weights /= weights.sum()

    smoothed_pos = sum(w * p for w, (p, _) in zip(weights, smooth_buffer[role]))
    smoothed_quat = sum(w * q for w, (_, q) in zip(weights, smooth_buffer[role]))
    smoothed_quat /= np.linalg.norm(smoothed_quat)

    return smoothed_pos, smoothed_quat

# --------------------------------------------------------------------------
# 🧠 Main loop
# --------------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)
    configuration.update_from_keyframe("teleop")

    for t in tasks:
        if hasattr(t, "set_target_from_configuration"):
            t.set_target_from_configuration(configuration)

    for hand, foot in zip(hands, feet):
        mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
        mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
    data.mocap_pos[com_mid] = data.subtree_com[1]

    rate = RateLimiter(frequency=200.0, warn=False)
    print("\n✅ VR Teleoperation Mode Active (Dual Stream + Smoothed)\n")

    while viewer.is_running():
        # --- 同时读取左右手最新包 ---
        new_data = recv_latest_dual(sock)
        for role in ["left", "right"]:
            if new_data[role] is not None:
                tracker_data[role] = new_data[role]

        count += 1
        frame_count += 1

        # --- COM固定 ---
        tasks[3].set_target(data.mocap_pos[com_mid])

        # --- 手部目标（平滑）---
        for i, role in enumerate(["right", "left"]):
            if tracker_data[role] is not None:
                smoothed_pos, smoothed_quat = smooth_update(role, tracker_data[role])
                data.mocap_pos[hands_mid[i]] = smoothed_pos
                data.mocap_quat[hands_mid[i]] = smoothed_quat
                hand_tasks[i].set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

        # --- 脚部静态目标 ---
        for i in range(2):
            feet_tasks[i].set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))

        # --- 求解IK ---
        if all(tracker_data.values()):
            t0 = time.time()
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1, limits=limits)
            ik_dt = (time.time() - t0) * 1000
            ik_time_accum += ik_dt
            configuration.integrate_inplace(vel, rate.dt)

        mujoco.mj_camlight(model, data)
        mujoco.mj_fwdPosition(model, data)

        # --- 每2帧渲染一次 (≈60FPS) ---
        if frame_count % 2 == 0:
            viewer.sync()

        rate.sleep()

        # --- FPS显示 ---
        if time.time() - last_print > 1.0:
            avg_ik = ik_time_accum / max(count, 1)
            print(f"FPS: {count:4d} | avg IK: {avg_ik:6.2f} ms/frame")
            count = 0
            ik_time_accum = 0.0
            last_print = time.time()

# 🦿 MuJoCo setup
# --------------------------------------------------------------------------
_HERE = Path(__file__).parent
_XML = _HERE / "unitree_g1" / "scene_table.xml"

model = mujoco.MjModel.from_xml_path(_XML.as_posix())
configuration = mink.Configuration(model)

feet = ["right_foot", "left_foot"]
hands = ["right_palm", "left_palm"]

tasks = [
    mink.FrameTask("pelvis", "body", 0.0, 1.0, 1.0),
    mink.FrameTask("torso_link", "body", 0.0, 1.0, 1.0),
    mink.PostureTask(model, cost=1e-1),
    mink.ComTask(cost=10.0),
]

feet_tasks = [
    mink.FrameTask(f, "site", 10.0, 1.0, 1.0) for f in feet
]
hand_tasks = [
    mink.FrameTask(h, "site", 5.0, 1.0, 1.0) for h in hands
]
tasks += feet_tasks + hand_tasks

# 🧱 collision constraints
collision_pairs = [
    (["left_hand_collision", "right_hand_collision"], ["table"]),
    (["left_hand_collision"], ["left_thigh"]),
    (["right_hand_collision"], ["right_thigh"]),
]
collision_avoidance_limit = mink.CollisionAvoidanceLimit(
    model, geom_pairs=collision_pairs,
    minimum_distance_from_collisions=0.05,
    collision_detection_distance=0.1,
)
limits = [mink.ConfigurationLimit(model), collision_avoidance_limit]

# mocap handles
com_mid = model.body("com_target").mocapid[0]
feet_mid = [model.body(f"{foot}_target").mocapid[0] for foot in feet]
hands_mid = [model.body(f"{hand}_target").mocapid[0] for hand in hands]

data = configuration.data
solver = "daqp"

last_print = time.time()
count = 0

# --------------------------------------------------------------------------
# 🧠 Main loop
# --------------------------------------------------------------------------
with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
    mujoco.mjv_defaultFreeCamera(model, viewer.cam)

    configuration.update_from_keyframe("teleop")
    for t in tasks:
        if hasattr(t, "set_target_from_configuration"):
            t.set_target_from_configuration(configuration)

    # init mocap
    for hand, foot in zip(hands, feet):
        mink.move_mocap_to_frame(model, data, f"{foot}_target", foot, "site")
        mink.move_mocap_to_frame(model, data, f"{hand}_target", hand, "site")
    data.mocap_pos[com_mid] = data.subtree_com[1]

    rate = RateLimiter(frequency=120.0, warn=False)  # 👌 稍低频率更平滑
    print("\n✅ VR Teleoperation Mode Active (Fast UDP)\n")

    while viewer.is_running():
        # --- 📥 Receive UDP (non-blocking with timeout)
        try:
            msg, _ = sock.recvfrom(512)
            vr_data = orjson.loads(msg)
            tracker_data[vr_data["role"]] = vr_data
        except socket.timeout:
            pass

        count += 1
        if time.time() - last_print > 1.0:
            print(f"FPS: {count} frames/sec")
            count = 0
            last_print = time.time()

        # --- Keep COM fixed
        tasks[3].set_target(data.mocap_pos[com_mid])

        # --- Update mocap targets from controllers
        for i, role in enumerate(["right", "left"]):
            if tracker_data[role] is not None:
                data.mocap_pos[hands_mid[i]] = np.array(tracker_data[role]["pos"], dtype=float)
                data.mocap_quat[hands_mid[i]] = np.array(tracker_data[role]["quat"], dtype=float)
                hand_tasks[i].set_target(mink.SE3.from_mocap_id(data, hands_mid[i]))

        # --- Feet: static target
        for i in range(2):
            feet_tasks[i].set_target(mink.SE3.from_mocap_id(data, feet_mid[i]))

        # --- IK solve if both hands valid
        if all(tracker_data.values()):
            vel = mink.solve_ik(configuration, tasks, rate.dt, solver, 1e-1, limits=limits)
            configuration.integrate_inplace(vel, rate.dt)

        mujoco.mj_camlight(model, data)
        mujoco.mj_fwdPosition(model, data)
        viewer.sync()
        rate.sleep()
