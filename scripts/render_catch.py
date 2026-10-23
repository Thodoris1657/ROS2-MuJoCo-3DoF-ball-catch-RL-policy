#!/usr/bin/env python
"""Record the full trial -- pick, carry, drop, bounce, catch -- for Earth, Mars and the Moon.

Writes one side-by-side MP4 (three panels advancing in the same simulated time, so
the different timing is visible directly), one MP4 per gravity, and a plot of ball
and cup height over time. Needs an offscreen GL (``MUJOCO_GL=egl`` or ``osmesa``)
and ffmpeg on the PATH.

    MUJOCO_GL=egl python scripts/render_catch.py --policy tuned --out media/
    MUJOCO_GL=egl python scripts/render_catch.py --policy runs/catch_sac/best_model.zip --slowmo 0.5
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from mujoco_arm_rl.catch_eval import make_catch_policy
from mujoco_arm_rl.envs.ballcatch import GRAVITIES, BallCatchEnv

LABEL = {"earth": "EARTH  g = 9.81", "mars": "MARS  g = 3.71", "moon": "MOON  g = 1.62"}


def overlay(frame: np.ndarray, lines: list[str], colour=(255, 255, 255)) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    try:
        big = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
        small = ImageFont.truetype("DejaVuSansMono.ttf", 16)
    except OSError:
        big = small = ImageFont.load_default()
    y = 10
    for i, text in enumerate(lines):
        font = big if i == 0 else small
        draw.text((12, y), text, font=font, fill=colour if i else (255, 220, 120),
                  stroke_width=2, stroke_fill=(0, 0, 0))
        y += 28 if i == 0 else 21
    return np.asarray(img)


def record(gname: str, policy_spec: str, seed: int, width: int, height: int, fps: int,
           drop, restitution: float | None) -> tuple[list[np.ndarray], dict]:
    env = BallCatchEnv(gravity=gname, start="pick", return_ball=False,
                       render_mode="rgb_array", width=width, height=height)
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = (0.17, 0.10, 0.24)
    cam.distance, cam.azimuth, cam.elevation = 1.25, 128.0, -14.0
    env.camera = cam
    policy = make_catch_policy(policy_spec)
    opts = {"drop": drop} if drop is not None else {}
    if restitution is not None:
        opts["restitution"] = restitution
    obs, info = env.reset(seed=seed, options=opts)
    policy.reset()
    every = max(1, int(round(1.0 / (fps * env.dt))))
    frames, trace = [], dict(t=[], ball=[], tip=[], phase=[])
    k, tail = 0, None
    status = ""
    while True:
        obs, _, term, trunc, info = env.step(policy(obs, env))
        k += 1
        trace["t"].append(env.sim_time)
        trace["ball"].append(env.ball_pos[2])
        trace["tip"].append(env.ee_pos[2])
        trace["phase"].append(info["phase"])
        if info["outcome"]:
            status = "CAUGHT" if info["is_success"] else info["outcome"]
        if k % every == 0:
            ph = {"pick": "pick + carry", "fall": "falling", "rise": "rebounding",
                  "hold": "caught!", "failed": "missed"}.get(info["phase"], info["phase"])
            lines = [LABEL[gname], f"t = {env.sim_time:5.2f} s   {ph}",
                     f"e = {env.restitution:.2f}   bounces {info['bounces']}"]
            if status:
                lines.append(status)
            frames.append(overlay(env.render(), lines))
        if (term or trunc) and tail is None:
            tail = int(0.8 * fps)        # keep filming a moment after the outcome
        if tail is not None:
            if k % every == 0:
                tail -= 1
            if tail <= 0:
                break
    trace.update(info=info, release=info.get("t_release"), apex=info.get("apex"))
    env.close()
    return frames, trace


def write_mp4(frames: list[np.ndarray], path: Path, fps: int) -> None:
    h, w, _ = frames[0].shape
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{w}x{h}", "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p",
           "-crf", "20", "-movflags", "+faststart", str(path)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames:
        proc.stdin.write(np.ascontiguousarray(f).tobytes())
    proc.stdin.close()
    proc.wait()


def plot(traces: dict[str, dict], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(traces), figsize=(5 * len(traces), 3.6), sharey=True)
    for ax, (g, tr) in zip(np.atleast_1d(axes), traces.items()):
        t0 = tr["release"] or 0.0
        t = np.asarray(tr["t"]) - t0
        keep = t > -0.3
        ax.plot(t[keep], np.asarray(tr["ball"])[keep], label="ball centre", color="#e8590c", lw=2)
        ax.plot(t[keep], np.asarray(tr["tip"])[keep] - 0.068, label="cup - contact distance",
                color="#1c7ed6", lw=2, ls="--")
        if tr["apex"]:
            ax.axvline(tr["apex"]["t"] - t0, color="grey", lw=0.8, ls=":")
            ax.axhline(tr["apex"]["z"], color="grey", lw=0.8, ls=":")
        ok = "caught" if tr["info"]["is_success"] else tr["info"]["outcome"]
        ax.set_title(f"{g.capitalize()} (g = {GRAVITIES[g]})  -  {ok}")
        ax.set_xlabel("time since release [s]")
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("height [m]")
    np.atleast_1d(axes)[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--policy", default="tuned")
    ap.add_argument("--gravity", nargs="*", default=list(GRAVITIES), choices=list(GRAVITIES))
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--drop", type=float, nargs=3, default=[0.30, 0.0, 0.48])
    ap.add_argument("--restitution", type=float, default=0.75)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--fps", type=int, default=50)
    ap.add_argument("--slowmo", type=float, default=1.0, help="<1 plays slower than real time")
    ap.add_argument("--out", default="media")
    args = ap.parse_args()
    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg not found on PATH")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    play_fps = max(1, int(round(args.fps * args.slowmo)))

    clips, traces = {}, {}
    for g in args.gravity:
        frames, tr = record(g, args.policy, args.seed, args.width, args.height, args.fps,
                            args.drop, args.restitution)
        clips[g], traces[g] = frames, tr
        write_mp4(frames, out / f"catch_{g}.mp4", play_fps)
        print(f"{g}: {tr['info']['outcome']}  ({len(frames)} frames) -> {out / f'catch_{g}.mp4'}")
    if len(clips) > 1:
        n = max(len(c) for c in clips.values())
        padded = [c + [c[-1]] * (n - len(c)) for c in clips.values()]
        sep = np.full((args.height, 4, 3), 20, dtype=np.uint8)
        combo = []
        for i in range(n):
            row = []
            for j, c in enumerate(padded):
                if j:
                    row.append(sep)
                row.append(c[i])
            combo.append(np.concatenate(row, axis=1))
        write_mp4(combo, out / "catch_all_gravities.mp4", play_fps)
        print(f"side by side -> {out / 'catch_all_gravities.mp4'}")
    plot(traces, out / "catch_heights.png")
    print(f"plot -> {out / 'catch_heights.png'}")


if __name__ == "__main__":
    main()
