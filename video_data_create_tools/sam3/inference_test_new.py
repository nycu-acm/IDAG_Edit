import os
import cv2
import numpy as np
import torch
import random
import subprocess
from pathlib import Path
from PIL import Image

from .sam3.model_builder import build_sam3_video_predictor
from .sam3.visualization_utils import prepare_masks_for_visualization

# ──────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────
def _load_video_frames(video_path: str) -> list:
    """Read every frame from *video_path* and return a list of RGB numpy arrays."""
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames

def _save_frames(video_frames: list, frames_dir: Path) -> None:
    """Save all frames as JPEGs at original resolution."""
    frames_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(video_frames):
        Image.fromarray(frame).save(frames_dir / f"{i:05d}.png")

def _save_object_masks(outputs_per_frame: dict, mask_dir: Path, total_frames: int, H: int, W: int,) -> None:
    """
    Save binary masks for one object at original video resolution.
 
    Parameters
    ----------
    outputs_per_frame : per-frame dict from _segment_single_object()
    mask_dir          : destination folder  (e.g. layout_masks/tiger/)
    total_frames      : total number of frames in the source video
    H, W              : original video height and width
    """
    mask_dir.mkdir(parents=True, exist_ok=True)
    blank = np.zeros((H, W), dtype=np.uint8)
 
    for frame_idx in range(total_frames):
        frame_name  = f"{frame_idx + 1:05d}.png"
        frame_masks = outputs_per_frame.get(frame_idx, {})
 
        if frame_masks:
            # Merge all obj entries for this frame (should be just obj_id=0)
            merged = np.zeros((H, W), dtype=np.uint8)
            for mask in frame_masks.values():
                merged[mask > 0] = 255
            Image.fromarray(merged).save(mask_dir / frame_name)
        else:
            Image.fromarray(blank).save(mask_dir / frame_name)

# --- Propagation ---
def _segment_single_object(predictor, video_path: str, prompt_text: str, prompt_frame_index: int,) -> dict:
    """
    Open a fresh SAM3 session for ONE object, propagate, and return per-frame masks.
 
    Returns
    -------
    dict  {frame_index: {obj_id: mask_array}}
    """
    response   = predictor.handle_request(dict(type="start_session", resource_path=video_path))
    session_id = response["session_id"]
 
    predictor.handle_request(dict(
        type="add_prompt",
        session_id=session_id,
        frame_index=prompt_frame_index,
        obj_id=0,               # always 0 — single object per session
        text=prompt_text,
    ))
 
    raw_outputs = {}
    for res in predictor.handle_stream_request(
        dict(type="propagate_in_video", session_id=session_id)
    ):
        raw_outputs[res["frame_index"]] = res["outputs"]
 
    return prepare_masks_for_visualization(raw_outputs)

def _build_videos(video_name: str, video_frames: list, all_object_masks: dict[str, dict], output_dir: Path, 
                  alpha: float = 0.4, fps: int = 8,) -> None:
    """
    Write output.mp4 (plain frames) and visualized.mp4 (all masks overlaid)
    at original video resolution.
 
    Parameters
    ----------
    video_frames     : list of RGB numpy arrays
    all_object_masks : {object_name: outputs_per_frame}
    output_dir       : destination folder
    alpha            : blend factor for mask overlay
    fps              : output video frame-rate
    """
    if not video_frames:
        return
 
    H, W  = video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    tmp_raw = output_dir / "raw_output.mp4"
    tmp_vis = output_dir / "vis_tmp.mp4"
 
    writer_raw = cv2.VideoWriter(str(tmp_raw), fourcc, fps, (W, H))
    writer_vis = cv2.VideoWriter(str(tmp_vis), fourcc, fps, (W, H))
 
    # Assign a fixed colour per object name
    color_map: dict[str, np.ndarray] = {}
    for i, name in enumerate(all_object_masks.keys()):
        random.seed(i)
        color_map[name] = np.array(
            [random.randint(60, 255), random.randint(60, 255), random.randint(60, 255)],
            dtype=np.uint8,
        )
 
    for frame_idx, frame in enumerate(video_frames):
        bgr_frame  = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        vis_overlay = frame.copy()
 
        for name, outputs_per_frame in all_object_masks.items():
            frame_masks = outputs_per_frame.get(frame_idx, {})
            for mask in frame_masks.values():
                vis_overlay[mask > 0] = color_map[name]
 
        vis_blend = (alpha * vis_overlay + (1 - alpha) * frame).astype(np.uint8)
        bgr_vis   = cv2.cvtColor(vis_blend, cv2.COLOR_RGB2BGR)
 
        writer_raw.write(bgr_frame)
        writer_vis.write(bgr_vis)
 
    writer_raw.release()
    writer_vis.release()
    
    # Re-encode to H.264 / yuv420p
    for f_in, f_out in [
        (tmp_raw, output_dir / f"{video_name}.mp4"),
        (tmp_vis, output_dir / "visualized.mp4"),
    ]:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(f_in), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(f_out)],
            capture_output=True,
            check=True,
        )
        f_in.unlink()

# --- Logic: Find Window and Crop ---
# def _process_video_assets(video_frames: list, outputs_per_frame: dict, output_dir: Path, obj_id_to_name: dict,
#                           num_frames: int = 16, target_size: int = 512, alpha: float = 0.4,) -> None:
#     """
#     Crop, resize, and export frames / masks / videos from SAM3 outputs.

#     Parameters
#     ----------
#     video_frames      : list of RGB numpy arrays (all frames of the source video)
#     outputs_per_frame : dict returned by _get_propagation()
#     output_dir        : root directory for all saved assets
#     num_frames        : how many consecutive frames to export (-1 = all)
#     target_size       : output resolution in pixels (square)
#     alpha             : mask overlay blend factor for the visualisation video
#     """
#     frames_dir = output_dir / "frames"
#     frames_dir.mkdir(parents=True, exist_ok=True)
 
#     # One sub-folder per object under layout_masks/
#     masks_root = output_dir / "layout_masks"
#     obj_mask_dirs: dict = {}
#     for obj_id, name in obj_id_to_name.items():
#         d = masks_root / name
#         d.mkdir(parents=True, exist_ok=True)
#         obj_mask_dirs[obj_id] = d
 
#     # ── 1. Find first frame that has at least one mask ──
#     start_frame = 0
#     for idx in sorted(outputs_per_frame.keys()):
#         if outputs_per_frame[idx]:
#             start_frame = idx
#             break
 
#     end_frame = (
#         len(video_frames)
#         if num_frames == -1
#         else min(start_frame + num_frames, len(video_frames))
#     )
#     relevant_indices = list(range(start_frame, end_frame))
 
#     # ── 2. Global bounding box across all objects in the chosen window ──
#     all_x, all_y = [], []
#     for idx in relevant_indices:
#         for obj_id, mask in outputs_per_frame.get(idx, {}).items():
#             ys, xs = np.where(mask > 0)
#             if len(xs):
#                 all_x.extend([xs.min(), xs.max()])
#                 all_y.extend([ys.min(), ys.max()])
 
#     H, W = video_frames[0].shape[:2]
#     if not all_x:
#         x1, y1, x2, y2 = 0, 0, W, H
#     else:
#         x1, x2 = min(all_x), max(all_x)
#         y1, y2 = min(all_y), max(all_y)
 
#     # ── 3. Square crop with 10 % padding, clamped to image boundaries ──
#     center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
#     side       = max(x2 - x1, y2 - y1) * 1.1
#     crop_x1    = max(0, int(center_x - side / 2))
#     crop_y1    = max(0, int(center_y - side / 2))
#     crop_x2    = min(W, int(crop_x1 + side))
#     crop_y2    = min(H, int(crop_y1 + side))
#     final_side = min(crop_x2 - crop_x1, crop_y2 - crop_y1)
 
#     # ── 4. Video writers (temp files, re-encoded later by FFmpeg) ──
#     tmp_raw = output_dir / "raw_crop.mp4"
#     tmp_vis = output_dir / "vis_tmp.mp4"
#     fourcc  = cv2.VideoWriter_fourcc(*"mp4v")
 
#     writer_raw = cv2.VideoWriter(str(tmp_raw), fourcc, 8, (target_size, target_size))
#     writer_vis = cv2.VideoWriter(str(tmp_vis), fourcc, 8, (target_size, target_size))
 
#     color_map: dict = {}
 
#     def _get_color(obj_id):
#         if obj_id not in color_map:
#             random.seed(int(obj_id))
#             color_map[obj_id] = np.array(
#                 [random.randint(60, 255), random.randint(60, 255), random.randint(60, 255)],
#                 dtype=np.uint8,
#             )
#         return color_map[obj_id]
 
#     # ── 5. Per-frame processing ──
#     for i, idx in enumerate(relevant_indices):
#         frame_name  = f"{i + 1:05d}"                            # "00001", "00002", …
#         frame       = video_frames[idx]
#         crop_img    = frame[crop_y1 : crop_y1 + final_side, crop_x1 : crop_x1 + final_side]
#         resized_img = cv2.resize(crop_img, (target_size, target_size))
 
#         # Save cropped frame
#         Image.fromarray(resized_img).save(frames_dir / f"{frame_name}.jpg")
 
#         vis_overlay = resized_img.copy()
 
#         for obj_id, mask in outputs_per_frame.get(idx, {}).items():
#             # Crop + resize this object's mask
#             m_crop    = mask[crop_y1 : crop_y1 + final_side, crop_x1 : crop_x1 + final_side]
#             m_crop    = m_crop.astype(np.uint8) * 255
#             m_resized = cv2.resize(m_crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
 
#             # Save to layout_masks/<object_name>/00001.png
#             if obj_id in obj_mask_dirs:
#                 Image.fromarray(m_resized).save(obj_mask_dirs[obj_id] / f"{frame_name}.png")
 
#             # Accumulate coloured overlay for the visualisation video
#             vis_overlay[m_resized > 0] = _get_color(obj_id)
 
#         vis_overlay = (alpha * vis_overlay + (1 - alpha) * resized_img).astype(np.uint8)
 
#         writer_raw.write(cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
#         writer_vis.write(cv2.cvtColor(vis_overlay, cv2.COLOR_RGB2BGR))
 
#     writer_raw.release()
#     writer_vis.release()
 
#     # ── 6. Re-encode to H.264 / yuv420p for broad compatibility ──
#     for f_in, f_out in [
#         (tmp_raw, output_dir / "output.mp4"),
#         (tmp_vis, output_dir / "visualized.mp4"),
#     ]:
#         subprocess.run(
#             ["ffmpeg", "-y", "-i", str(f_in), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(f_out)],
#             capture_output=True,
#             check=True,
#         )
#         f_in.unlink()

def split_with_ffmpeg(video_path: str, out_root: str, num_clips: int = 3, num_frames: int = 16, fps: int = 8,) -> list[dict]:
    """
    Split *video_path* into *num_clips* non-overlapping clips of *num_frames* frames each.
 
    Returns a list of dicts with keys: start, end, dir, clip.
    """
    video_path = Path(video_path)
    video_name = video_path.stem
 
    cap   = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
 
    max_start = total - num_frames
    assert max_start > 0, "Video too short for the requested clip length."
 
    possible = list(range(max_start))
    random.shuffle(possible)
    starts: list[int] = []
    for s in possible:
        if all(abs(s - p) >= num_frames for p in starts):
            starts.append(s)
        if len(starts) == num_clips:
            break
    starts = sorted(starts)
    print("Selected clip starts:", starts)
 
    base_dir = Path(out_root) / video_name
    base_dir.mkdir(parents=True, exist_ok=True)
    meta = []
 
    for start in starts:
        end      = start + num_frames - 1
        clip_dir = base_dir / f"{video_name}_{start}_{end}"
 
        for sub in ("frames", "layout_masks", "clip", "visualized_clip"):
            (clip_dir / sub).mkdir(parents=True, exist_ok=True)
 
        clip_path = clip_dir / "clip" / "clip.mp4"
 
        subprocess.run(
            [
                "ffmpeg", "-y", "-i", str(video_path),
                "-vf", f"select='between(n,{start},{end})',setpts=PTS-STARTPTS",
                "-r", str(fps), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                str(clip_path),
            ],
            check=True,
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(clip_path), str(clip_dir / "frames" / "frame_%02d.jpg")],
            check=True,
        )
 
        meta.append({"start": start, "end": end, "dir": clip_dir, "clip": clip_path})
        print("Created:", clip_dir)
 
    return meta


# ──────────────────────────────────────────────
# Public entry-point
# ──────────────────────────────────────────────
def sam3_inference(
    video_path: Path,
    text_prompts: dict[str, str],
    prompt_frame_index: int = 0,
    output_dir: str = "sam3_output_assets",
    # num_frames: int = -1,
    # target_size: int = 512,
    fps: int = 8,
    gpu_id: int = 0,
) -> Path:
    """
    Run SAM3 segmentation on *video_path* and export cropped frames, masks, and videos.

    Parameters
    ----------
    video_path          : path to the source video file
    text_prompt         : text description of the object to track
    prompt_frame_index  : frame index used for the initial text prompt
    output_dir          : root folder for all output assets
    num_frames          : number of consecutive frames to export (-1 = all)
    target_size         : output resolution in pixels (square crop)
    gpu_id              : CUDA device index

    Returns
    -------
    Path to the *output_dir* containing all saved assets.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
 
    # video_name = Path(video_path).stem
    # out_path   = Path(output_dir) / video_name
    # out_path.mkdir(parents=True, exist_ok=True)
 
    # ── Load all frames once (original resolution) ──
    # print(f"Loading frames from: {video_path}")
    # video_frames = _load_video_frames(video_path)
    # H, W         = video_frames[0].shape[:2]
    # total_frames = len(video_frames)
    # print(f"  {total_frames} frames loaded  ({W}×{H})")
 
    # # ── Save frames at original resolution ──
    # print("Saving frames …")
    # _save_frames(video_frames, out_path / "frames")
    
    # ── Load all frames once (original resolution) ──
    print(f"Loading frames from: {video_path}")   
    video_frames = []
    image_paths = sorted(Path(video_path).glob("*.png"))  # 或 *.png
    for path in image_paths:
        img = cv2.imread(str(path))  # shape: (H, W, C)，BGR 格式
        video_frames.append(img)
    H, W = video_frames[0].shape[:2]
    total_frames = len(video_frames)    
 
    # ── Build predictor (shared across all object sessions) ──
    gpus_to_use = [torch.cuda.current_device()]
    predictor   = build_sam3_video_predictor(gpus_to_use=gpus_to_use)
 
    # ── Segment each object in its own session ──
    all_object_masks: dict[str, dict] = {}
 
    for name, prompt_text in text_prompts.items():
        print(f"\nSegmenting '{name}'  →  prompt: \"{prompt_text}\"")
 
        outputs_per_frame = _segment_single_object(
            predictor=predictor,
            video_path=video_path,
            prompt_text=prompt_text,
            prompt_frame_index=prompt_frame_index,
        )
 
        # Save per-frame masks at original resolution
        _save_object_masks(
            outputs_per_frame=outputs_per_frame,
            mask_dir=output_dir / "layout_masks" / name,
            total_frames=total_frames,
            H=H,
            W=W,
        )
 
        all_object_masks[name] = outputs_per_frame
        print(f"  Masks saved → layout_masks/{name}/")
 
    # ── Build output videos at original resolution ──
    print("\nBuilding output videos …")
    print(os.path.basename(video_path))
    _build_videos(
        video_name=os.path.basename(video_path),
        video_frames=video_frames,
        all_object_masks=all_object_masks,
        output_dir=output_dir,
        fps=fps,
    )
 
    predictor.shutdown()
    print(f"\nDone. Assets saved to: {output_dir}")
    return output_dir


# ──────────────────────────────────────────────
# CLI convenience
# ──────────────────────────────────────────────
# if __name__ == "__main__":
    # sam3_inference(
    #     video_path="./videos/run_two_man.mp4",
    #     text_prompts={
    #         "left_man":  "Man in red hoodie",
    #         "right_man": "man in gray shirt",
    #     }
    # )