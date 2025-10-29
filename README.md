## 🦿 Unitree G1 Half-body VR Teleoperation

This project enables **real-time upper-body teleoperation** of the **Unitree G1 humanoid**
using **HTC Vive / Valve Index controllers**.
Controller poses are streamed via UDP and solved through **Mink’s inverse kinematics (IK)** in **MuJoCo**.

---

### ⚙️ Environment Setup

Before running, activate your virtual environment and run:

```bash
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
```

This prevents version conflicts between **Mink** and **MuJoCo** (`libstdc++`).

---

### 🚀 Run Instructions

#### 🖥 Windows (VR data sender)

```bash
python g1_hfbody.py
```

#### 🧠 Ubuntu (IK and visualization)

```bash
python humanoid_g1_vr.py
```

---

### 🧩 Notes

* `humanoid_g1_vr.py` is the **final and most optimized version**
  → smoothest motion, lowest latency, and best IK stability.
* Powered by **Mink’s QP-based IK solver** with real-time MuJoCo simulation.
* Typical performance: **<30 ms total delay**, **120 Hz update**, **60 Hz rendering**.

---

### 📘 Summary

**Ultra-low-latency teleoperation (~25 ms end-to-end)**
Built with **Mink IK + MuJoCo + Vive/Index controllers**.

---

MIT License © 2025 [Easton J](https://github.com/1EastonJ)
