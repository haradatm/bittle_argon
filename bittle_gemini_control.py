"""Petoi Bittle X — MuJoCo Controller with Gemini 4 Argon Integration.

Provides high-level action primitives (tools) for controlling the Petoi Bittle X
quadruped in MuJoCo, plus multimodal rendering and Gemini 4 Argon agent execution.
"""

from __future__ import annotations

import argparse
import io
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import imageio.v2 as imageio
import imageio_ffmpeg
import mujoco
import numpy as np
from dotenv import load_dotenv
from google import genai
from google.genai import types
from PIL import Image

# Add bittle-mujoco directory to sys.path so we can import gaits
ROOT_DIR = Path(__file__).resolve().parent
BITTLE_DIR = ROOT_DIR / "bittle-mujoco"
if str(BITTLE_DIR) not in sys.path:
    sys.path.insert(0, str(BITTLE_DIR))

from gaits import GAIT_COL_MAP, GAITS, STAND_CTRL, _convert_frame, convert_gait, convert_pose


def convert_gait_right(name: str) -> np.ndarray:
    """Create a right-turning gait by mirroring left/right leg columns of a left gait.

    OpenCat gait table columns:
      0: LF upper, 1: RF upper, 2: RR upper, 3: LR upper,
      4: LF lower, 5: RF lower, 6: RR lower, 7: LR lower
    Swapping LF<->RF (0<->1, 4<->5) and LR<->RR (2<->3, 6<->7) mirrors the gait.
    """
    if name not in GAITS:
        raise KeyError(f"Unknown gait '{name}'")
    frames = GAITS[name]
    mirrored = []
    for f in frames:
        mf = [f[1], f[0], f[3], f[2], f[5], f[4], f[7], f[6]]
        mirrored.append(_convert_frame(mf))
    return np.array(mirrored)


class BittleMuJoCoController:
    """High-level controller for Petoi Bittle X in MuJoCo."""

    FRAME_DT = 0.020  # OpenCat 50 Hz frame rate

    def __init__(
        self,
        xml_path: Path = BITTLE_DIR / "bittle.xml",
        record_video: bool = True,
        fps: int = 30,
        width: int = 640,
        height: int = 480,
        step_callback: Optional[Callable[[], None]] = None,
    ):
        self.xml_path = xml_path
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)

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
        self.camera.elevation = -22.0
        self.camera.azimuth = 135.0
        self.camera.lookat[:] = [0.0, 0.0, 0.05]

        self.frames: list[np.ndarray] = []
        self._last_render_time = -1.0
        self.current_ctrl = STAND_CTRL.copy()
        self.action_log: list[dict] = []

        self.reset()

    def reset(self) -> dict:
        """Reset Bittle to the calibrated standing keyframe and settle."""
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)
        self.current_ctrl = STAND_CTRL.copy()
        self.data.ctrl[:] = self.current_ctrl
        self._last_render_time = -1.0
        # Settle for 0.3 seconds
        self._step_sim(0.3)
        return self.get_state()

    def _step_sim(self, duration_s: float, ctrl_fn: Optional[Callable[[float], np.ndarray]] = None):
        """Step the simulation for `duration_s` seconds."""
        n_steps = int(round(duration_s / self.model.opt.timestep))
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
        """Render current camera view (tracking the torso softly)."""
        torso_pos = self.data.sensor("torso_pos").data
        self.camera.lookat[0] = float(torso_pos[0])
        self.camera.lookat[1] = float(torso_pos[1])
        self.camera.lookat[2] = 0.05
        self.renderer.update_scene(self.data, camera=self.camera)
        return self.renderer.render().copy()

    def get_snapshot_png(self) -> bytes:
        """Return a PNG image of the current scene."""
        rgb = self.render_frame()
        img = Image.fromarray(rgb)
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()

    def get_state(self) -> dict:
        """Get the robot's position, orientation (yaw, pitch, roll), and height."""
        pos = self.data.sensor("torso_pos").data.copy()
        quat = self.data.sensor("torso_quat").data.copy()  # [w, x, y, z]
        w, x, y, z = quat

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (w * x + y * z)
        cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
        roll_deg = math.degrees(math.atan2(sinr_cosp, cosr_cosp))

        # Pitch (y-axis rotation)
        sinp = 2.0 * (w * y - z * x)
        pitch_deg = math.degrees(math.asin(max(-1.0, min(1.0, sinp))))

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_deg = math.degrees(math.atan2(siny_cosp, cosy_cosp))

        return {
            "x_m": round(float(pos[0]), 4),
            "y_m": round(float(pos[1]), 4),
            "z_m": round(float(pos[2]), 4),
            "yaw_deg": round(yaw_deg, 1),
            "pitch_deg": round(pitch_deg, 1),
            "roll_deg": round(roll_deg, 1),
            "sim_time_s": round(float(self.data.time), 2),
        }

    def transition_to_pose(self, pose_name: str, transition_s: float = 0.6, hold_s: float = 0.6) -> dict:
        """Smoothly blend from current joint angles to a target pose and hold."""
        start_state = self.get_state()
        target_ctrl = convert_pose(pose_name)
        start_ctrl = self.current_ctrl.copy()

        def blend_ctrl(t: float) -> np.ndarray:
            alpha = min(1.0, t / max(transition_s, 1e-4))
            # Smooth cosine interpolation
            smooth_alpha = 0.5 * (1.0 - math.cos(math.pi * alpha))
            return start_ctrl + (target_ctrl - start_ctrl) * smooth_alpha

        self._step_sim(transition_s + hold_s, ctrl_fn=blend_ctrl)
        end_state = self.get_state()
        result = {
            "action": f"pose:{pose_name}",
            "before": start_state,
            "after": end_state,
        }
        self.action_log.append(result)
        return result

    def play_gait_cycles(self, gait_seq: np.ndarray, gait_label: str, steps: int = 1) -> dict:
        """Play a gait sequence for `steps` cycles with smooth entry and exit to balance."""
        steps = max(1, int(steps))
        start_state = self.get_state()

        # First smoothly ensure we are near STAND_CTRL before starting gait
        pre_start = self.current_ctrl.copy()
        if np.linalg.norm(pre_start - STAND_CTRL) > 0.15:
            self.transition_to_pose("balance", transition_s=0.4, hold_s=0.2)
            # Remove intermediate log entry to keep action_log clean
            self.action_log.pop()

        n_frames = len(gait_seq)
        total_frames = n_frames * steps
        total_duration = total_frames * self.FRAME_DT
        warmup_time = min(0.4, total_duration * 0.3)

        def gait_ctrl(t: float) -> np.ndarray:
            frame_idx = int(t / self.FRAME_DT) % n_frames
            scale = min(1.0, t / max(warmup_time, 1e-4))
            return STAND_CTRL + (gait_seq[frame_idx] - STAND_CTRL) * scale

        self._step_sim(total_duration, ctrl_fn=gait_ctrl)

        # Return smoothly to balance pose at the end of the gait
        post_start = self.current_ctrl.copy()

        def settle_ctrl(t: float) -> np.ndarray:
            alpha = min(1.0, t / 0.25)
            return post_start + (STAND_CTRL - post_start) * alpha

        self._step_sim(0.35, ctrl_fn=settle_ctrl)

        end_state = self.get_state()
        dx = end_state["x_m"] - start_state["x_m"]
        dy = end_state["y_m"] - start_state["y_m"]
        dist_m = math.hypot(dx, dy)
        result = {
            "action": f"gait:{gait_label}",
            "steps": steps,
            "distance_moved_m": round(dist_m, 4),
            "before": start_state,
            "after": end_state,
        }
        self.action_log.append(result)
        return result

    def save_video(self, path: Path | str):
        """Save recorded frames to an MP4 or GIF file safely without partial-pipe-write corruption."""
        if not self.frames:
            return
        path = Path(path)
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
# Tool Wrapper Factory for Gemini Function Calling
# ---------------------------------------------------------------------------

def build_gemini_tools(controller: BittleMuJoCoController):
    """Build Python functions with docstrings and type annotations for Gemini Tool Use."""

    def walk_forward(steps: int = 1, style: str = "trot") -> dict:
        """Move the Bittle robot forward by the specified number of steps (gait cycles).

        Args:
            steps: Number of gait cycles (steps) to move forward (e.g., 3 for 3 steps).
            style: Locomotion style. One of 'trot' (default, stable and fast), 'walk', 'crawl', or 'gallop'.
        """
        gait_map = {"trot": "trF", "walk": "wkF", "crawl": "crF", "gallop": "gpF"}
        gait_key = gait_map.get(style.lower(), "trF")
        gait_seq = convert_gait(gait_key)
        return controller.play_gait_cycles(gait_seq, f"forward_{style}", steps=steps)

    def walk_backward(steps: int = 1) -> dict:
        """Move the Bittle robot backward by the specified number of steps (gait cycles).

        Args:
            steps: Number of gait cycles (steps) to walk backward.
        """
        gait_seq = convert_gait("bkF")
        return controller.play_gait_cycles(gait_seq, "backward", steps=steps)

    def turn_left(steps: int = 1) -> dict:
        """Turn the Bittle robot to the left by the specified number of steps.

        Args:
            steps: Number of gait cycles to turn left.
        """
        gait_seq = convert_gait("trL")
        return controller.play_gait_cycles(gait_seq, "turn_left", steps=steps)

    def turn_right(steps: int = 1) -> dict:
        """Turn the Bittle robot to the right by the specified number of steps.

        Args:
            steps: Number of gait cycles to turn right.
        """
        gait_seq = convert_gait_right("trL")
        return controller.play_gait_cycles(gait_seq, "turn_right", steps=steps)

    def stand() -> dict:
        """Make the Bittle robot stand up in its balanced default posture."""
        return controller.transition_to_pose("balance", transition_s=0.6, hold_s=0.5)

    def lie_down() -> dict:
        """Make the Bittle robot lie down flat on the ground (rest / prone posture, '伏せ')."""
        return controller.transition_to_pose("rest", transition_s=0.8, hold_s=0.8)

    def sit() -> dict:
        """Make the Bittle robot sit down on its hind legs ('お座り')."""
        return controller.transition_to_pose("sit", transition_s=0.7, hold_s=0.7)

    def stretch() -> dict:
        """Make the Bittle robot perform a body stretch posture ('伸び')."""
        return controller.transition_to_pose("str", transition_s=0.8, hold_s=0.8)

    def butt_up() -> dict:
        """Make the Bittle robot bow its front legs down with its rear up ('プレイバウ / お尻上げ')."""
        return controller.transition_to_pose("buttUp", transition_s=0.7, hold_s=0.7)

    def jump_forward(jumps: int = 1) -> dict:
        """Make the Bittle robot jump forward.

        Args:
            jumps: Number of jumps to perform.
        """
        gait_seq = convert_gait("jpF")
        return controller.play_gait_cycles(gait_seq, "jump_forward", steps=jumps)

    def get_robot_state() -> dict:
        """Return the current 3D position (x, y, z in meters) and orientation (yaw, pitch, roll in degrees) of the Bittle robot."""
        return controller.get_state()

    tools = [
        walk_forward,
        walk_backward,
        turn_left,
        turn_right,
        stand,
        lie_down,
        sit,
        stretch,
        butt_up,
        jump_forward,
        get_robot_state,
    ]
    tool_map = {fn.__name__: fn for fn in tools}
    return tools, tool_map


# ---------------------------------------------------------------------------
# Gemini 4 Argon Robotics Agent Loop
# ---------------------------------------------------------------------------

SYSTEM_INSTRUCTION = """You are an embodied robotics controller driving a Petoi Bittle X quadruped robot in a MuJoCo physics simulator.
When the user gives a natural language command (in Japanese or English), break it down into the appropriate sequence of robot action tool calls and execute them in order.
- After each action (or sequence of actions), inspect the returned physical state (position x, y, z and orientation yaw, pitch, roll) and visual camera feedback to verify that the robot executed the motion properly.
- Once all requested actions are completed, summarize the actions performed and the final physical state clearly in the user's language.
"""


def run_gemini_agent(
    prompt: str,
    controller: BittleMuJoCoController,
    model_name: str = "gemini-4-argon",
    include_vision: bool = True,
    max_turns: int = 10,
) -> str:
    """Run a multi-step tool-use loop with Gemini 4 Argon to control Bittle in MuJoCo."""
    load_dotenv(ROOT_DIR / ".env")
    client = genai.Client()

    tools, tool_map = build_gemini_tools(controller)

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_INSTRUCTION,
        tools=tools,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        temperature=0.1,
    )

    initial_parts = [
        types.Part.from_text(
            text=f"Initial robot state: {controller.get_state()}\nUser instruction: {prompt}"
        )
    ]
    if include_vision:
        initial_parts.append(
            types.Part.from_bytes(data=controller.get_snapshot_png(), mime_type="image/png")
        )

    contents: list[types.Content] = [
        types.Content(role="user", parts=initial_parts)
    ]

    print(f"\n[Gemini Agent] Model: {model_name}")
    print(f"[User Prompt]  {prompt}")
    print(f"[Initial State] {controller.get_state()}\n")

    for turn in range(1, max_turns + 1):
        response = client.models.generate_content(
            model=model_name,
            contents=contents,
            config=config,
        )

        candidate = response.candidates[0]
        contents.append(candidate.content)

        # Print any reasoning/text parts produced in this turn
        for part in candidate.content.parts:
            if part.text:
                print(f"[Gemini Turn {turn}] {part.text}")

        function_calls = response.function_calls
        if not function_calls:
            # Done! Return final text response
            return response.text or ""

        # Execute all function calls requested in this turn in sequence
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
                    data=controller.get_snapshot_png(), mime_type="image/png"
                )
            )

        contents.append(types.Content(role="user", parts=tool_response_parts))

    return "Max turns reached."


def main():
    parser = argparse.ArgumentParser(
        description="Control Petoi Bittle X in MuJoCo with natural language via Gemini 4 Argon."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="前に3歩進んでから伏せて",
        help="Natural language command for Bittle.",
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
        default="bittle_gemini_demo.mp4",
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

    if args.viewer:
        import mujoco.viewer

        controller = BittleMuJoCoController(record_video=bool(args.video))
        with mujoco.viewer.launch_passive(controller.model, controller.data) as viewer:
            def sync_cb():
                viewer.sync()
                time.sleep(controller.model.opt.timestep * 0.5)

            controller.step_callback = sync_cb
            final_text = run_gemini_agent(
                prompt=args.prompt,
                controller=controller,
                model_name=args.model,
                include_vision=not args.no_vision,
            )
            print("\n=== Final Response ===")
            print(final_text)
    else:
        controller = BittleMuJoCoController(record_video=bool(args.video))
        final_text = run_gemini_agent(
            prompt=args.prompt,
            controller=controller,
            model_name=args.model,
            include_vision=not args.no_vision,
        )
        print("\n=== Final Response ===")
        print(final_text)

    if args.video:
        out_path = Path(args.video)
        controller.save_video(out_path)
        print(f"\n[Video Saved] {out_path.resolve()} ({len(controller.frames)} frames)")


if __name__ == "__main__":
    main()

