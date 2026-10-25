"""Petoi Bittle X — Zero-Preset Direct Joint Control + Autonomous Skill Learning via Gemini 4 Argon.

This script extends `bittle_gemini_direct.py` with a persistent JSON-backed
Skill Library (`learned_skills.json`) so Gemini 4 Argon can save newly discovered
postures, keyframe sequences, and CPG gaits and reuse them in 1 shot on future runs.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parent.parent
XML_PATH = ROOT_DIR / "bittle-mujoco" / "bittle.xml"

# Symmetric intuitive degree interface -> MuJoCo actuator radian conversion:
# In the symmetric interface:
#   - shoulder_deg:
#       0 deg = neutral stand shoulder angle (thigh angled 30 deg backward from vertical)
#       +deg  = swing thigh FORWARD (toward head)
#       -deg  = swing thigh BACKWARD (toward tail)
#   - knee_deg:
#       0 deg = neutral stand knee angle (foot ~90 mm below shoulder)
#       +deg  = bend/lift lower leg FORWARD & UP
#       -deg  = swing/fold lower leg BACKWARD
#
# In bittle.xml, the physical "stand" keyframe actuator values are:
#   RF_sh(0) = -0.7854 rad (-45 deg), RF_kn(1) = +1.9199 rad (+110 deg)
#   LF_sh(2) = -0.7854 rad (-45 deg), LF_kn(3) = +1.9199 rad (+110 deg)
#   RR_sh(4) = +0.7854 rad (+45 deg), RR_kn(5) = +1.9199 rad (+110 deg)
#   LR_sh(6) = +0.7854 rad (+45 deg), LR_kn(7) = +1.9199 rad (+110 deg)


def symmetric_deg_to_mjcf_ctrl(
    rf_shoulder_deg: float,
    rf_knee_deg: float,
    lf_shoulder_deg: float,
    lf_knee_deg: float,
    rr_shoulder_deg: float,
    rr_knee_deg: float,
    lr_shoulder_deg: float,
    lr_knee_deg: float,
    stand_ctrl: np.ndarray,
) -> np.ndarray:
    """Convert intuitive symmetric relative joint angles (in degrees) to MJCF ctrl radians."""
    ctrl = stand_ctrl.copy()
    # Decreasing actuator angle swings front shoulders forward (+ axis is backward)
    ctrl[0] = stand_ctrl[0] - math.radians(rf_shoulder_deg)
    ctrl[2] = stand_ctrl[2] - math.radians(lf_shoulder_deg)
    # Increasing actuator angle swings rear shoulders forward (- axis is backward)
    ctrl[4] = stand_ctrl[4] + math.radians(rr_shoulder_deg)
    ctrl[6] = stand_ctrl[6] + math.radians(lr_shoulder_deg)

    # Decreasing actuator angle swings/lifts lower leg (shank) forward & up
    ctrl[1] = stand_ctrl[1] - math.radians(rf_knee_deg)
    ctrl[3] = stand_ctrl[3] - math.radians(lf_knee_deg)
    ctrl[5] = stand_ctrl[5] - math.radians(rr_knee_deg)
    ctrl[7] = stand_ctrl[7] - math.radians(lr_knee_deg)

    return np.clip(ctrl, -2.6, 2.6)


def mjcf_ctrl_to_symmetric_deg(ctrl: np.ndarray, stand_ctrl: np.ndarray) -> dict[str, float]:
    """Convert MJCF ctrl radians back to intuitive symmetric relative degrees."""
    return {
        "rf_shoulder_deg": round(math.degrees(stand_ctrl[0] - ctrl[0]), 1),
        "rf_knee_deg": round(math.degrees(stand_ctrl[1] - ctrl[1]), 1),
        "lf_shoulder_deg": round(math.degrees(stand_ctrl[2] - ctrl[2]), 1),
        "lf_knee_deg": round(math.degrees(stand_ctrl[3] - ctrl[3]), 1),
        "rr_shoulder_deg": round(math.degrees(ctrl[4] - stand_ctrl[4]), 1),
        "rr_knee_deg": round(math.degrees(stand_ctrl[5] - ctrl[5]), 1),
        "lr_shoulder_deg": round(math.degrees(ctrl[6] - stand_ctrl[6]), 1),
        "lr_knee_deg": round(math.degrees(stand_ctrl[7] - ctrl[7]), 1),
    }


class DirectBittleSandbox:
    """MuJoCo physics sandbox for Petoi Bittle X with zero preset gaits/poses."""

    def __init__(
        self,
        xml_path: Path = XML_PATH,
        record_video: bool = True,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        step_callback: Optional[Callable[[], None]] = None,
    ):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

        # Read the hardware calibration neutral standing keyframe directly from bittle.xml
        self.stand_ctrl = self.model.key_ctrl[0].copy()
        self.current_ctrl = self.stand_ctrl.copy()

        self.record_video = record_video
        self.fps = fps
        self.width = width
        self.height = height
        self.step_callback = step_callback

        def _safe_renderer_close(r_self):
            if getattr(r_self, "_mjr_context", None) is not None:
                if getattr(r_self, "_gl_context", None) is not None:
                    r_self._gl_context.make_current()
                r_self._mjr_context.free()
                r_self._mjr_context = None
            if getattr(r_self, "_gl_context", None) is not None:
                r_self._gl_context.free()
                r_self._gl_context = None

        mujoco.Renderer.close = _safe_renderer_close
        prev_renderer = getattr(mujoco, "_active_bittle_renderer", None)
        if prev_renderer is not None:
            try:
                prev_renderer.close()
            except Exception:
                pass

        self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        mujoco._active_bittle_renderer = self.renderer
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.model, self.camera)
        self.camera.distance = 0.45
        self.camera.elevation = -20.0
        self.camera.azimuth = 135.0
        self.camera.lookat[:] = [0.0, 0.0, 0.05]

        self.frames: list[np.ndarray] = []
        self._last_render_time = -1.0

        self.reset()

    def reset(self) -> dict:
        """Reset simulation to initial standing state from XML keyframe 0."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.current_ctrl = self.stand_ctrl.copy()
        self.data.ctrl[:] = self.current_ctrl
        self._last_render_time = -1.0
        self._step_sim(0.3)
        return self.get_state()

    def _step_sim(self, duration_s: float, ctrl_fn: Optional[Callable[[float], np.ndarray]] = None):
        """Step physics for `duration_s` seconds."""
        n_steps = max(1, int(round(duration_s / self.model.opt.timestep)))
        for i in range(n_steps):
            t = i * self.model.opt.timestep
            if ctrl_fn is not None:
                self.current_ctrl = ctrl_fn(t)
            self.data.ctrl[:] = self.current_ctrl
            mujoco.mj_step(self.model, self.data)

            if self.step_callback is not None:
                self.step_callback()

            if self.record_video:
                if self.data.time - self._last_render_time >= (1.0 / self.fps):
                    self._last_render_time = self.data.time
                    self.frames.append(self.render_frame())

    def render_frame(self) -> np.ndarray:
        """Render current camera view tracking the robot's torso."""
        torso_pos = self.data.sensor("torso_pos").data
        self.camera.lookat[0] = float(torso_pos[0])
        self.camera.lookat[1] = float(torso_pos[1])
        self.camera.lookat[2] = 0.05
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def get_snapshot_png(self) -> bytes:
        """Capture a PNG snapshot of the current simulation state."""
        rgb = self.render_frame()
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def get_state(self) -> dict:
        """Return detailed physical state: torso 3D position, orientation, joint angles, and foot heights."""
        pos = self.data.sensor("torso_pos").data.copy()
        quat = self.data.sensor("torso_quat").data.copy()  # [w, x, y, z]
        w, x, y, z = quat

        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll_deg = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        sinp = 2.0 * (w * y - z * x)
        pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))

        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))

        floor_z = float(self.data.geom_xpos[self.model.geom("floor").id][2])
        foot_heights_mm = {}
        for leg, site_name in [
            ("RF", "rf_foot_site"),
            ("LF", "lf_foot_site"),
            ("RR", "rr_foot_site"),
            ("LR", "lr_foot_site"),
        ]:
            sz = float(self.data.site(site_name).xpos[2])
            foot_heights_mm[leg] = round((sz - floor_z) * 1000.0, 1)

        fallen = abs(roll_deg) > 60.0 or abs(pitch_deg) > 60.0

        return {
            "torso_pos_m": {
                "x": round(float(pos[0]), 4),
                "y": round(float(pos[1]), 4),
                "z_above_floor": round(float(pos[2]) - floor_z, 4),
            },
            "orientation_deg": {
                "yaw": round(yaw_deg, 1),
                "pitch": round(pitch_deg, 1),
                "roll": round(roll_deg, 1),
            },
            "fallen": fallen,
            "commanded_joint_angles_deg": mjcf_ctrl_to_symmetric_deg(
                self.current_ctrl, self.stand_ctrl
            ),
            "foot_site_heights_above_floor_mm": foot_heights_mm,
            "sim_time_s": round(float(self.data.time), 2),
        }

    def move_joints_to(
        self,
        rf_shoulder_deg: float = 0.0,
        rf_knee_deg: float = 0.0,
        lf_shoulder_deg: float = 0.0,
        lf_knee_deg: float = 0.0,
        rr_shoulder_deg: float = 0.0,
        rr_knee_deg: float = 0.0,
        lr_shoulder_deg: float = 0.0,
        lr_knee_deg: float = 0.0,
        transition_s: float = 0.5,
        hold_s: float = 0.3,
    ) -> dict:
        """Smoothly interpolate from current joint angles to the specified 8 joint angles and hold."""
        before = self.get_state()
        start_ctrl = self.current_ctrl.copy()
        target_ctrl = symmetric_deg_to_mjcf_ctrl(
            rf_shoulder_deg=rf_shoulder_deg,
            rf_knee_deg=rf_knee_deg,
            lf_shoulder_deg=lf_shoulder_deg,
            lf_knee_deg=lf_knee_deg,
            rr_shoulder_deg=rr_shoulder_deg,
            rr_knee_deg=rr_knee_deg,
            lr_shoulder_deg=lr_shoulder_deg,
            lr_knee_deg=lr_knee_deg,
            stand_ctrl=self.stand_ctrl,
        )

        transition_s = max(0.05, float(transition_s))
        hold_s = max(0.0, float(hold_s))

        def interp_ctrl(t: float) -> np.ndarray:
            alpha = min(1.0, t / transition_s)
            smooth = 0.5 * (1.0 - math.cos(math.pi * alpha))
            return start_ctrl + (target_ctrl - start_ctrl) * smooth

        self._step_sim(transition_s + hold_s, ctrl_fn=interp_ctrl)
        after = self.get_state()
        return {
            "before_pos_m": before["torso_pos_m"],
            "after_state": after,
        }

    def execute_keyframe_trajectory(self, keyframes: list[dict]) -> dict:
        """Execute a custom multi-step sequence of joint angle keyframes synthesized by the model."""
        before = self.get_state()
        for kf in keyframes:
            self.move_joints_to(
                rf_shoulder_deg=float(kf.get("rf_shoulder_deg", 0.0)),
                rf_knee_deg=float(kf.get("rf_knee_deg", 0.0)),
                lf_shoulder_deg=float(kf.get("lf_shoulder_deg", 0.0)),
                lf_knee_deg=float(kf.get("lf_knee_deg", 0.0)),
                rr_shoulder_deg=float(kf.get("rr_shoulder_deg", 0.0)),
                rr_knee_deg=float(kf.get("rr_knee_deg", 0.0)),
                lr_shoulder_deg=float(kf.get("lr_shoulder_deg", 0.0)),
                lr_knee_deg=float(kf.get("lr_knee_deg", 0.0)),
                transition_s=float(kf.get("transition_s", 0.25)),
                hold_s=float(kf.get("hold_s", 0.05)),
            )
        after = self.get_state()
        dx = after["torso_pos_m"]["x"] - before["torso_pos_m"]["x"]
        dy = after["torso_pos_m"]["y"] - before["torso_pos_m"]["y"]
        return {
            "num_keyframes_executed": len(keyframes),
            "displacement_m": {"dx": round(dx, 4), "dy": round(dy, 4)},
            "before_pos_m": before["torso_pos_m"],
            "after_state": after,
        }

    def execute_custom_oscillator(
        self,
        cycles: int,
        cycle_period_s: float,
        rf_phase_deg: float,
        lf_phase_deg: float,
        rr_phase_deg: float,
        lr_phase_deg: float,
        rf_shoulder_amp_deg: float,
        lf_shoulder_amp_deg: float,
        rr_shoulder_amp_deg: float,
        lr_shoulder_amp_deg: float,
        knee_lift_amp_deg: float,
        shoulder_offset_deg: float = 0.0,
        knee_offset_deg: float = 0.0,
    ) -> dict:
        """Synthesize and run a continuous sinusoidal CPG (Central Pattern Generator) trajectory."""
        before = self.get_state()
        # Smoothly align to neutral standing posture if starting from a non-standing pose (e.g. sit/lie_down)
        if np.linalg.norm(self.current_ctrl - self.stand_ctrl) > 0.15:
            pre_start = self.current_ctrl.copy()
            target_stand = symmetric_deg_to_mjcf_ctrl(
                shoulder_offset_deg, knee_offset_deg,
                shoulder_offset_deg, knee_offset_deg,
                shoulder_offset_deg, knee_offset_deg,
                shoulder_offset_deg, knee_offset_deg,
                stand_ctrl=self.stand_ctrl,
            )
            def align_ctrl(t: float) -> np.ndarray:
                alpha = min(1.0, t / 0.35)
                smooth = 0.5 * (1.0 - math.cos(math.pi * alpha))
                return pre_start + (target_stand - pre_start) * smooth
            self._step_sim(0.45, ctrl_fn=align_ctrl)

        cycles = max(1, int(cycles))
        cycle_period_s = max(0.2, float(cycle_period_s))
        total_time = cycles * cycle_period_s
        warmup_s = min(0.3, total_time * 0.25)

        phases = {
            "rf": math.radians(rf_phase_deg),
            "lf": math.radians(lf_phase_deg),
            "rr": math.radians(rr_phase_deg),
            "lr": math.radians(lr_phase_deg),
        }
        sh_amps = {
            "rf": rf_shoulder_amp_deg,
            "lf": lf_shoulder_amp_deg,
            "rr": rr_shoulder_amp_deg,
            "lr": lr_shoulder_amp_deg,
        }

        def cpg_ctrl(t: float) -> np.ndarray:
            env = min(1.0, t / max(warmup_s, 1e-4))
            omega = 2.0 * math.pi / cycle_period_s
            degs = {}
            for leg in ("rf", "lf", "rr", "lr"):
                theta = omega * t + phases[leg]
                # When sin(theta) > 0, leg is in swing phase (lifting knee and moving shoulder from back to front)
                # Shoulder position leads lift by -90 deg (-cos(theta)) so:
                #   at theta=0..pi (swing): knee is lifted (+sin(theta) > 0), shoulder moves from -amp to +amp (-cos(theta))
                #   at theta=pi..2pi (stance): knee is on ground (0), shoulder pushes body forward by moving from +amp to -amp
                sh_deg = shoulder_offset_deg - sh_amps[leg] * math.cos(theta) * env
                kn_deg = knee_offset_deg + knee_lift_amp_deg * max(0.0, math.sin(theta)) * env
                degs[f"{leg}_sh"] = sh_deg
                degs[f"{leg}_kn"] = kn_deg

            return symmetric_deg_to_mjcf_ctrl(
                rf_shoulder_deg=degs["rf_sh"],
                rf_knee_deg=degs["rf_kn"],
                lf_shoulder_deg=degs["lf_sh"],
                lf_knee_deg=degs["lf_kn"],
                rr_shoulder_deg=degs["rr_sh"],
                rr_knee_deg=degs["rr_kn"],
                lr_shoulder_deg=degs["lr_sh"],
                lr_knee_deg=degs["lr_kn"],
                stand_ctrl=self.stand_ctrl,
            )

        self._step_sim(total_time, ctrl_fn=cpg_ctrl)
        after = self.get_state()
        dx = after["torso_pos_m"]["x"] - before["torso_pos_m"]["x"]
        dy = after["torso_pos_m"]["y"] - before["torso_pos_m"]["y"]
        return {
            "cycles": cycles,
            "displacement_m": {"dx": round(dx, 4), "dy": round(dy, 4)},
            "before_pos_m": before["torso_pos_m"],
            "after_state": after,
        }

    def save_video(self, path: Path | str):
        """Save recorded frames to an MP4 or GIF file safely without partial-pipe-write corruption."""
        if not self.frames:
            return
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".gif":
            imageio.mimsave(str(path), self.frames, fps=self.fps)
            return

        h, w, _ = self.frames[0].shape
        raw_bytes = b"".join(np.ascontiguousarray(f, dtype=np.uint8).tobytes() for f in self.frames)
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [
            ffmpeg_exe,
            "-y",
            "-f", "rawvideo",
            "-vcodec", "rawvideo",
            "-s", f"{w}x{h}",
            "-pix_fmt", "rgb24",
            "-r", str(self.fps),
            "-i", "-",
            "-an",
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            "-crf", "18",
            "-loglevel", "error",
            str(path),
        ]
        subprocess.run(cmd, input=raw_bytes, check=True)


# ---------------------------------------------------------------------------
# Persistent Skill Library (JSON-backed Self-Learning Store)
# ---------------------------------------------------------------------------

DEFAULT_SKILLS_PATH = ROOT_DIR / "learned_skills.json"


class SkillLibrary:
    """Manages persistent robot skills learned and synthesized by Gemini."""

    def __init__(self, path: Path = DEFAULT_SKILLS_PATH):
        self.path = Path(path)
        self.skills: dict[str, dict[str, Any]] = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            try:
                self.skills = json.loads(self.path.read_text(encoding="utf-8"))
            except Exception:
                self.skills = {}
        else:
            self.skills = {}

    def save(self) -> None:
        self.path.write_text(
            json.dumps(self.skills, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def register_pose(
        self, name: str, description: str, joint_angles: dict[str, float]
    ) -> dict[str, Any]:
        entry = {
            "name": name,
            "kind": "pose",
            "description": description,
            "joint_angles": joint_angles,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.skills[name] = entry
        self.save()
        return entry

    def register_keyframes(
        self, name: str, description: str, keyframes: list[dict[str, float]]
    ) -> dict[str, Any]:
        entry = {
            "name": name,
            "kind": "keyframes",
            "description": description,
            "keyframes": keyframes,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.skills[name] = entry
        self.save()
        return entry

    def register_cpg(
        self, name: str, description: str, cpg_params: dict[str, float]
    ) -> dict[str, Any]:
        entry = {
            "name": name,
            "kind": "cpg",
            "description": description,
            "cpg_params": cpg_params,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self.skills[name] = entry
        self.save()
        return entry

    def summary_for_prompt(self) -> str:
        """Compact summary of available learned skills to inject into the prompt."""
        if not self.skills:
            return "None yet (Skill library is currently empty)."
        lines = []
        for name, info in self.skills.items():
            lines.append(
                f"- `{name}` (kind: {info['kind']}): {info['description']}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Low-level Joint Tools + Skill Library Tools for Gemini 4 Argon (NO PRESETS)
# ---------------------------------------------------------------------------

def build_direct_tools(sandbox: DirectBittleSandbox, skill_lib: SkillLibrary):
    """Build low-level joint control primitives AND persistent skill library tools."""

    def set_joint_angles(
        rf_shoulder_deg: float = 0.0,
        rf_knee_deg: float = 0.0,
        lf_shoulder_deg: float = 0.0,
        lf_knee_deg: float = 0.0,
        rr_shoulder_deg: float = 0.0,
        rr_knee_deg: float = 0.0,
        lr_shoulder_deg: float = 0.0,
        lr_knee_deg: float = 0.0,
        transition_s: float = 0.6,
        hold_s: float = 0.4,
    ) -> dict:
        """Directly set the 8 joint target angles (in degrees relative to neutral stand) and smoothly transition to them.

        Coordinate convention for ALL 4 LEGS (RF=Right Front, LF=Left Front, RR=Right Rear, LR=Left Rear):
        - shoulder_deg:
            * 0.0 deg = neutral standing thigh angle (thigh angled 30 deg backward from vertical)
            * Positive (+deg, up to +90): swings the thigh FORWARD (toward the head / horizontal forward)
            * Negative (-deg, down to -70): swings the thigh BACKWARD (toward the tail)
        - knee_deg:
            * 0.0 deg = neutral standing knee angle (foot on ground ~90 mm below shoulder)
            * Positive (+deg, up to +90): bends/lifts the lower leg (shank) FORWARD and UP
            * Negative (-deg, down to -70): folds/extends the lower leg BACKWARD

        Examples of physical geometry from first principles:
        - Neutral standing: all shoulders = 0, all knees = 0 (torso z_above_floor ≈ 0.095 m)
        - Lying flat / prone (伏せ): swing thighs forward (+45 to +65 deg) so the belly lowers to the floor, and tuck knees (+40 to +55 deg)
        - Sitting (お座り): keep front legs extended/supporting (front shoulders ~ -10 to +10, front knees ~ 0) while folding rear legs (rear shoulders +45 to +60, rear knees +45 to +60) so the rear drops
        - Lifting one front paw (e.g. RF wave/shake): shift weight slightly to other 3 legs, swing RF shoulder forward (+50 to +75 deg) and lift RF knee (+30 to +60 deg).
        """
        return sandbox.move_joints_to(
            rf_shoulder_deg=rf_shoulder_deg,
            rf_knee_deg=rf_knee_deg,
            lf_shoulder_deg=lf_shoulder_deg,
            lf_knee_deg=lf_knee_deg,
            rr_shoulder_deg=rr_shoulder_deg,
            rr_knee_deg=rr_knee_deg,
            lr_shoulder_deg=lr_shoulder_deg,
            lr_knee_deg=lr_knee_deg,
            transition_s=transition_s,
            hold_s=hold_s,
        )

    def play_custom_keyframes(keyframes: list[dict]) -> dict:
        """Execute a sequence of custom joint-angle keyframes in order.

        Use this to synthesize multi-step custom motions (such as waving a paw back and forth, doing push-ups, dancing, bowing, or custom step-by-step walking).

        Args:
            keyframes: A list of dicts, where each dict can specify:
                - rf_shoulder_deg (float), rf_knee_deg (float)
                - lf_shoulder_deg (float), lf_knee_deg (float)
                - rr_shoulder_deg (float), rr_knee_deg (float)
                - lr_shoulder_deg (float), lr_knee_deg (float)
                - transition_s (float, e.g. 0.25): duration in seconds to interpolate to this keyframe
                - hold_s (float, e.g. 0.05): duration in seconds to hold this keyframe
        """
        return sandbox.execute_keyframe_trajectory(keyframes)

    def run_custom_cpg_oscillator(
        cycles: int = 3,
        cycle_period_s: float = 0.5,
        rf_phase_deg: float = 0.0,
        lf_phase_deg: float = 180.0,
        rr_phase_deg: float = 180.0,
        lr_phase_deg: float = 0.0,
        rf_shoulder_amp_deg: float = 20.0,
        lf_shoulder_amp_deg: float = 20.0,
        rr_shoulder_amp_deg: float = 20.0,
        lr_shoulder_amp_deg: float = 20.0,
        knee_lift_amp_deg: float = 28.0,
        shoulder_offset_deg: float = 0.0,
        knee_offset_deg: float = 0.0,
    ) -> dict:
        """Synthesize and execute a continuous sinusoidal Central Pattern Generator (CPG) gait from mathematical parameters.

        How the oscillator works for each leg i in {rf, lf, rr, lr}:
          theta_i(t) = 2 * pi * t / cycle_period_s + radians(phase_deg_i)
          shoulder_deg_i(t) = shoulder_offset_deg - shoulder_amp_deg_i * cos(theta_i(t))
          knee_deg_i(t)     = knee_offset_deg + knee_lift_amp_deg * max(0, sin(theta_i(t)))

        Physical tips:
        - Diagonal trot forward: pair (RF, LR) at phase 0 deg and (LF, RR) at phase 180 deg, with positive shoulder_amp_deg (~18 to 25 deg) and knee_lift_amp_deg (~25 to 35 deg).
        - Walking backward: use negative shoulder_amp_deg (e.g. -20 deg) so the stance phase pushes backward.
        - Turning left: make right legs have larger amplitude (e.g. +25 deg) and left legs smaller or negative amplitude (e.g. -10 to +5 deg).
        - Turning right: make left legs have larger amplitude (e.g. +25 deg) and right legs smaller or negative amplitude.
        """
        return sandbox.execute_custom_oscillator(
            cycles=cycles,
            cycle_period_s=cycle_period_s,
            rf_phase_deg=rf_phase_deg,
            lf_phase_deg=lf_phase_deg,
            rr_phase_deg=rr_phase_deg,
            lr_phase_deg=lr_phase_deg,
            rf_shoulder_amp_deg=rf_shoulder_amp_deg,
            lf_shoulder_amp_deg=lf_shoulder_amp_deg,
            rr_shoulder_amp_deg=rr_shoulder_amp_deg,
            lr_shoulder_amp_deg=lr_shoulder_amp_deg,
            knee_lift_amp_deg=knee_lift_amp_deg,
            shoulder_offset_deg=shoulder_offset_deg,
            knee_offset_deg=knee_offset_deg,
        )

    def execute_saved_skill(skill_name: str, repeat: int = 1) -> dict:
        """Execute a previously learned and saved skill from `learned_skills.json` by name.

        Args:
            skill_name: Name of the saved skill (e.g. 'trot_forward', 'sit', 'lie_down', 'wave_right_paw').
            repeat: Number of cycles (for 'cpg' skills) or repetitions (for 'keyframes' skills). Ignored for 'pose' skills.
        """
        if skill_name not in skill_lib.skills:
            return {
                "error": f"Skill '{skill_name}' not found.",
                "available_skills": list(skill_lib.skills.keys()),
            }
        entry = skill_lib.skills[skill_name]
        kind = entry["kind"]
        repeat = max(1, int(repeat))

        if kind == "pose":
            ja = entry["joint_angles"]
            res = sandbox.move_joints_to(
                rf_shoulder_deg=float(ja.get("rf_shoulder_deg", 0.0)),
                rf_knee_deg=float(ja.get("rf_knee_deg", 0.0)),
                lf_shoulder_deg=float(ja.get("lf_shoulder_deg", 0.0)),
                lf_knee_deg=float(ja.get("lf_knee_deg", 0.0)),
                rr_shoulder_deg=float(ja.get("rr_shoulder_deg", 0.0)),
                rr_knee_deg=float(ja.get("rr_knee_deg", 0.0)),
                lr_shoulder_deg=float(ja.get("lr_shoulder_deg", 0.0)),
                lr_knee_deg=float(ja.get("lr_knee_deg", 0.0)),
                transition_s=float(ja.get("transition_s", 0.6)),
                hold_s=float(ja.get("hold_s", 0.4)),
            )
            res["executed_skill"] = skill_name
            res["kind"] = "pose"
            return res

        elif kind == "keyframes":
            kfs = entry["keyframes"] * repeat
            res = sandbox.execute_keyframe_trajectory(kfs)
            res["executed_skill"] = skill_name
            res["kind"] = "keyframes"
            res["repeat"] = repeat
            return res

        elif kind == "cpg":
            p = entry["cpg_params"]
            res = sandbox.execute_custom_oscillator(
                cycles=repeat,
                cycle_period_s=float(p.get("cycle_period_s", 0.5)),
                rf_phase_deg=float(p.get("rf_phase_deg", 0.0)),
                lf_phase_deg=float(p.get("lf_phase_deg", 180.0)),
                rr_phase_deg=float(p.get("rr_phase_deg", 180.0)),
                lr_phase_deg=float(p.get("lr_phase_deg", 0.0)),
                rf_shoulder_amp_deg=float(p.get("rf_shoulder_amp_deg", 20.0)),
                lf_shoulder_amp_deg=float(p.get("lf_shoulder_amp_deg", 20.0)),
                rr_shoulder_amp_deg=float(p.get("rr_shoulder_amp_deg", 20.0)),
                lr_shoulder_amp_deg=float(p.get("lr_shoulder_amp_deg", 20.0)),
                knee_lift_amp_deg=float(p.get("knee_lift_amp_deg", 28.0)),
                shoulder_offset_deg=float(p.get("shoulder_offset_deg", 0.0)),
                knee_offset_deg=float(p.get("knee_offset_deg", 0.0)),
            )
            res["executed_skill"] = skill_name
            res["kind"] = "cpg"
            return res

        return {"error": f"Unsupported skill kind: {kind}"}

    def save_pose_skill(
        skill_name: str,
        description: str,
        rf_shoulder_deg: float = 0.0,
        rf_knee_deg: float = 0.0,
        lf_shoulder_deg: float = 0.0,
        lf_knee_deg: float = 0.0,
        rr_shoulder_deg: float = 0.0,
        rr_knee_deg: float = 0.0,
        lr_shoulder_deg: float = 0.0,
        lr_knee_deg: float = 0.0,
        transition_s: float = 0.6,
        hold_s: float = 0.4,
    ) -> dict:
        """Save a verified static posture (pose) as a reusable skill in `learned_skills.json`.

        Args:
            skill_name: Snake_case identifier (e.g. 'stand', 'sit', 'lie_down').
            description: Clear description of what the posture does.
        """
        entry = skill_lib.register_pose(
            name=skill_name,
            description=description,
            joint_angles={
                "rf_shoulder_deg": rf_shoulder_deg,
                "rf_knee_deg": rf_knee_deg,
                "lf_shoulder_deg": lf_shoulder_deg,
                "lf_knee_deg": lf_knee_deg,
                "rr_shoulder_deg": rr_shoulder_deg,
                "rr_knee_deg": rr_knee_deg,
                "lr_shoulder_deg": lr_shoulder_deg,
                "lr_knee_deg": lr_knee_deg,
                "transition_s": transition_s,
                "hold_s": hold_s,
            },
        )
        return {"status": "saved", "skill": entry}

    def save_keyframe_skill(
        skill_name: str,
        description: str,
        keyframes: list[dict],
    ) -> dict:
        """Save a verified multi-step keyframe motion as a reusable skill in `learned_skills.json`.

        Tip: For rhythmic gestures like waving a paw or push-ups, save 1 unit cycle (or the core motion) so `execute_saved_skill(skill_name, repeat=N)` can repeat it N times.

        Args:
            skill_name: Snake_case identifier (e.g. 'wave_right_paw', 'push_up').
            description: Clear description of the motion and whether `repeat` controls repetitions.
            keyframes: List of keyframe dicts with the 8 joint angles, transition_s, and hold_s.
        """
        entry = skill_lib.register_keyframes(
            name=skill_name,
            description=description,
            keyframes=keyframes,
        )
        return {"status": "saved", "skill": entry}

    def save_cpg_skill(
        skill_name: str,
        description: str,
        cycle_period_s: float = 0.45,
        rf_phase_deg: float = 0.0,
        lf_phase_deg: float = 180.0,
        rr_phase_deg: float = 180.0,
        lr_phase_deg: float = 0.0,
        rf_shoulder_amp_deg: float = 22.0,
        lf_shoulder_amp_deg: float = 22.0,
        rr_shoulder_amp_deg: float = 22.0,
        lr_shoulder_amp_deg: float = 22.0,
        knee_lift_amp_deg: float = 30.0,
        shoulder_offset_deg: float = 0.0,
        knee_offset_deg: float = 0.0,
    ) -> dict:
        """Save a verified CPG oscillator gait as a reusable skill in `learned_skills.json`.

        When executed via `execute_saved_skill(skill_name, repeat=N)`, `repeat` will specify the number of gait cycles (steps).

        Args:
            skill_name: Snake_case identifier (e.g. 'trot_forward', 'walk_backward', 'turn_left').
            description: Clear description of the gait (including approx distance per cycle if known).
        """
        entry = skill_lib.register_cpg(
            name=skill_name,
            description=description,
            cpg_params={
                "cycle_period_s": cycle_period_s,
                "rf_phase_deg": rf_phase_deg,
                "lf_phase_deg": lf_phase_deg,
                "rr_phase_deg": rr_phase_deg,
                "lr_phase_deg": lr_phase_deg,
                "rf_shoulder_amp_deg": rf_shoulder_amp_deg,
                "lf_shoulder_amp_deg": lf_shoulder_amp_deg,
                "rr_shoulder_amp_deg": rr_shoulder_amp_deg,
                "lr_shoulder_amp_deg": lr_shoulder_amp_deg,
                "knee_lift_amp_deg": knee_lift_amp_deg,
                "shoulder_offset_deg": shoulder_offset_deg,
                "knee_offset_deg": knee_offset_deg,
            },
        )
        return {"status": "saved", "skill": entry}

    def list_saved_skills() -> dict:
        """Return all currently saved skills in `learned_skills.json`."""
        return {"skills": skill_lib.skills}

    def get_robot_state() -> dict:
        """Return the robot's current 3D position, orientation (yaw, pitch, roll), joint angles, and foot heights."""
        return sandbox.get_state()

    tools = [
        execute_saved_skill,
        save_pose_skill,
        save_keyframe_skill,
        save_cpg_skill,
        list_saved_skills,
        set_joint_angles,
        play_custom_keyframes,
        run_custom_cpg_oscillator,
        get_robot_state,
    ]
    tool_map = {fn.__name__: fn for fn in tools}
    return tools, tool_map


SYSTEM_INSTRUCTION = """You are an embodied robotics controller driving an 8-DOF Petoi Bittle X quadruped robot in MuJoCo, equipped with a persistent **Self-Learning Skill Library** (`learned_skills.json`).

### Workflow Rules:
1. **REUSE EXISTING SKILLS FIRST (Fast & Low Cost)**:
   - Check the "Available Learned Skills" provided in the prompt.
   - If a saved skill matches part or all of the user's instruction, call `execute_saved_skill(skill_name=..., repeat=...)` directly! You can call multiple `execute_saved_skill` calls in sequence in a single turn when all required skills already exist.
2. **SYNTHESIZE NEW MOTIONS FROM FIRST PRINCIPLES**:
   - If a requested motion or posture is NOT yet in the skill library, synthesize it using the low-level tools:
     * `set_joint_angles` for static postures (stand, sit, lie_down, etc.)
     * `play_custom_keyframes` for multi-frame gestures (waving a paw, push-ups, bow, dance, etc.)
     * `run_custom_cpg_oscillator` for rhythmic locomotion (trot forward, backward, turn left/right, etc.)
3. **VERIFY & SAVE NEW SKILLS TO THE LIBRARY**:
   - After executing a newly synthesized motion, inspect the physical state (`torso_pos_m`, `orientation_deg`, `fallen`) and camera image.
   - Once verified that the motion succeeded without falling, **save each distinct reusable skill** using `save_pose_skill`, `save_keyframe_skill`, or `save_cpg_skill` so future runs can execute it in 1 shot without trial-and-error!

### Joint Angle Coordinate System (for all 4 legs: RF, LF, RR, LR):
- `shoulder_deg`: 0° = neutral stand; `+deg` swings the thigh FORWARD toward the head; `-deg` swings the thigh BACKWARD toward the tail.
- `knee_deg`: 0° = neutral stand; `+deg` bends/lifts the lower leg (shank) FORWARD & UP; `-deg` swings/folds the lower leg BACKWARD.
"""


def run_direct_gemini_agent(
    prompt: str,
    sandbox: DirectBittleSandbox,
    skill_lib: SkillLibrary,
    model_name: str = "gemini-4-argon",
    include_vision: bool = True,
    max_turns: int = 10,
) -> str:
    """Run Gemini 4 Argon with direct joint control + persistent skill library."""
    load_dotenv(ROOT_DIR / ".env")
    client = genai.Client()

    tools, tool_map = build_direct_tools(sandbox, skill_lib)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.1,
    )

    skills_summary = skill_lib.summary_for_prompt()
    initial_text = (
        f"=== Available Learned Skills in `{skill_lib.path.name}` ===\n"
        f"{skills_summary}\n\n"
        f"Initial robot state: {sandbox.get_state()}\n"
        f"User instruction: {prompt}"
    )

    initial_parts = [types.Part.from_text(text=initial_text)]
    if include_vision:
        initial_parts.append(
            types.Part.from_bytes(data=sandbox.get_snapshot_png(), mime_type="image/png")
        )

    contents: list[types.Content] = [
        types.Content(role="user", parts=initial_parts)
    ]

    print(f"\n[Zero-Preset + Skill-Library Agent] Model: {model_name}")
    print(f"[Skills Loaded] {len(skill_lib.skills)} skill(s): {list(skill_lib.skills.keys())}")
    print(f"[User Prompt]   {prompt}")
    print(f"[Initial State] {sandbox.get_state()}\n")

    api_calls = 0
    for turn in range(1, max_turns + 1):
        api_calls += 1
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)

        for part in candidate.content.parts:
            if part.text:
                print(f"[Gemini Turn {turn}] {part.text}")

        function_calls = response.function_calls
        if not function_calls:
            print(f"\n[Completed in {api_calls} API call(s) | Saved skills: {list(skill_lib.skills.keys())}]")
            return response.text or ""

        tool_response_parts: list[types.Part] = []
        for fc in function_calls:
            fn_name = fc.name
            fn_args = dict(fc.args) if fc.args else {}
            print(f"  -> Executing tool: {fn_name}({fn_args})")

            if fn_name not in tool_map:
                res = {"error": f"Unknown tool: {fn_name}"}
            else:
                res = tool_map[fn_name](**fn_args)

            print(f"     Result: {res}")
            tool_response_parts.append(
                types.Part.from_function_response(name=fn_name, response={"result": res})
            )

        if include_vision:
            tool_response_parts.append(
                types.Part.from_bytes(
                    data=sandbox.get_snapshot_png(), mime_type="image/png"
                )
            )

        contents.append(types.Content(role="user", parts=tool_response_parts))

    return "Max turns reached."


def main():
    parser = argparse.ArgumentParser(
        description="Zero-Preset Direct Joint Control + Autonomous Skill Learning via Gemini 4 Argon."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="前に3歩進んでから、お座りして、右前足を上げて3回バイバイ（手を振る）して、最後に伏せて",
        help="Natural language command for Bittle.",
    )
    parser.add_argument(
        "--skills-file",
        type=str,
        default=str(DEFAULT_SKILLS_PATH),
        help="Path to the persistent learned skills JSON file.",
    )
    parser.add_argument(
        "--reset-skills",
        action="store_true",
        help="Clear the learned skills JSON file before running.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-4-argon",
        help="Gemini model ID (default: gemini-4-argon).",
    )
    parser.add_argument(
        "--video",
        type=str,
        default=str(ROOT_DIR / "videos" / "bittle_skills_demo.mp4"),
        help="Output MP4/GIF path to record the simulation.",
    )
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Launch interactive MuJoCo passive viewer while executing.",
    )
    parser.add_argument(
        "--no-vision",
        action="store_true",
        help="Disable sending rendered MuJoCo camera images to Gemini.",
    )
    args = parser.parse_args()

    skills_path = Path(args.skills_file)
    if args.reset_skills and skills_path.exists():
        skills_path.unlink()
        print(f"[Reset] Removed existing skills file: {skills_path}")

    skill_lib = SkillLibrary(skills_path)

    if args.viewer:
        import mujoco.viewer

        sandbox = DirectBittleSandbox(record_video=bool(args.video))
        with mujoco.viewer.launch_passive(sandbox.model, sandbox.data) as viewer:
            def sync_cb():
                viewer.sync()
                time.sleep(sandbox.model.opt.timestep * 0.5)

            sandbox.step_callback = sync_cb
            final_text = run_direct_gemini_agent(
                prompt=args.prompt,
                sandbox=sandbox,
                skill_lib=skill_lib,
                model_name=args.model,
                include_vision=not args.no_vision,
            )
            print("\n=== Final Response ===")
            print(final_text)
    else:
        sandbox = DirectBittleSandbox(record_video=bool(args.video))
        final_text = run_direct_gemini_agent(
            prompt=args.prompt,
            sandbox=sandbox,
            skill_lib=skill_lib,
            model_name=args.model,
            include_vision=not args.no_vision,
        )
        print("\n=== Final Response ===")
        print(final_text)

    if args.video:
        out_path = Path(args.video)
        sandbox.save_video(out_path)
        print(f"\n[Video Saved] {out_path.resolve()} ({len(sandbox.frames)} frames)")


if __name__ == "__main__":
    main()
