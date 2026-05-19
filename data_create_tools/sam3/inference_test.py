import os
import cv2
import numpy as np
import torch
import random
import subprocess
from pathlib import Path
from PIL import Image, ImageDraw
import sam3
from sam3.model_builder import build_sam3_video_predictor
from sam3.visualization_utils import prepare_masks_for_visualization

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# --- Setup ---
output_dir = Path("sam3_output_assets")
output_dir.mkdir(exist_ok=True)
frames_dir = output_dir / "frames"
masks_dir = output_dir / "layout_masks"
frames_dir.mkdir(exist_ok=True)
masks_dir.mkdir(exist_ok=True)

gpus_to_use = [torch.cuda.current_device()]
predictor = build_sam3_video_predictor(gpus_to_use=gpus_to_use)

video_path = "./videos/two_man_beach.mp4"
cap = cv2.VideoCapture(video_path)
video_frames = []
while True:
    ret, frame = cap.read()
    if not ret: break
    video_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
cap.release()

# --- SAM3 Session ---
response = predictor.handle_request(dict(type="start_session", resource_path=video_path))
session_id = response["session_id"]

# Add Prompt (e.g., "person")
predictor.handle_request(dict(
    type="add_prompt", session_id=session_id, frame_index=0, text="left shirtless man"
))


def split_with_ffmpeg(video_path, out_root, num_clips=3, num_frames=16, fps=8):
    video_path = Path(video_path)
    video_name = video_path.stem

    # --- get metadata ---
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    max_start = total - num_frames
    assert max_start > 0, "Video too short"

    # ---- sample non-overlapping ----
    possible = list(range(max_start))
    random.shuffle(possible)

    starts = []
    for s in possible:
        if all(abs(s - p) >= num_frames for p in starts):
            starts.append(s)
        if len(starts) == num_clips:
            break

    starts = sorted(starts)
    print("Selected starts:", starts)

    # ==================================================
    # Create folders and extract clips
    # ==================================================

    base_dir = Path(out_root) / video_name
    base_dir.mkdir(exist_ok=True, parents=True)

    meta = []

    for start in starts:
        end = start + num_frames - 1

        clip_dir = base_dir / f"{video_name}_{start}_{end}"
        (clip_dir / "frames").mkdir(parents=True, exist_ok=True)
        (clip_dir / "layout_masks").mkdir(exist_ok=True)
        (clip_dir / "clip").mkdir(exist_ok=True)
        (clip_dir / "visualized_clip").mkdir(exist_ok=True)

        clip_path = clip_dir / "clip" / "clip.mp4"

        # ---- FFmpeg frame-accurate cut ----
        cmd = [
            "ffmpeg", "-y",
            "-i", str(video_path),

            # frame-accurate seeking
            "-vf", f"select='between(n,{start},{end})',setpts=PTS-STARTPTS",

            "-r", str(fps),
            "-c:v", "libx264",
            "-pix_fmt", "yuv420p",
            str(clip_path)
        ]

        subprocess.run(cmd, check=True)

        # ---- also dump frames for your loader ----
        frame_pattern = clip_dir / "frames" / "frame_%02d.jpg"

        subprocess.run([
            "ffmpeg", "-y",
            "-i", str(clip_path),
            str(frame_pattern)
        ], check=True)

        meta.append({
            "start": start,
            "end": end,
            "dir": clip_dir,
            "clip": clip_path
        })

        print("Created:", clip_dir)

    return meta

# --- Propagation ---
def get_propagation(predictor, session_id):
    outputs = {}
    for res in predictor.handle_stream_request(dict(type="propagate_in_video", session_id=session_id)):
        outputs[res["frame_index"]] = res["outputs"]
    return prepare_masks_for_visualization(outputs)

outputs_per_frame = get_propagation(predictor, session_id)

# --- Logic: Find Window and Crop ---
def process_video_assets(video_frames, outputs_per_frame, num_frames=16, target_size=512):
    # 1. Find the first frame where at least one object exists (or modify for specific IDs)
    start_frame = 0
    for idx in sorted(outputs_per_frame.keys()):
        if len(outputs_per_frame[idx]) > 0:
            start_frame = idx
            break
    if num_frames == -1:
        end_frame = len(video_frames)
    else:
        end_frame = min(start_frame + num_frames, len(video_frames))
    relevant_indices = list(range(start_frame, end_frame))

    # 2. Find Global Bounding Box for all objects across these 16 frames
    all_x, all_y = [], []
    for idx in relevant_indices:
        frame_masks = outputs_per_frame.get(idx, {})
        for obj_id, mask in frame_masks.items():
            ys, xs = np.where(mask > 0)
            if len(xs) > 0:
                all_x.extend([xs.min(), xs.max()])
                all_y.extend([ys.min(), ys.max()])

    if not all_x: # Fallback if no masks found
        x1, y1, x2, y2 = 0, 0, video_frames[0].shape[1], video_frames[0].shape[0]
    else:
        x1, x2, y1, y2 = min(all_x), max(all_x), min(all_y), max(all_y)

    # 3. Make it 1:1 Aspect Ratio
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    side = max(x2 - x1, y2 - y1) * 1.1 # 10% padding
    alpha = 0.4  # for visualization blending
    # Clamp to image boundaries
    H, W = video_frames[0].shape[:2]
    crop_x1 = max(0, int(center_x - side / 2))
    crop_y1 = max(0, int(center_y - side / 2))
    crop_x2 = min(W, int(crop_x1 + side))
    crop_y2 = min(H, int(crop_y1 + side))

    # Adjust back if out of bounds to maintain 1:1
    final_side = min(crop_x2 - crop_x1, crop_y2 - crop_y1)

    # 4. Generate Assets
    tmp_path = output_dir / "raw_crop.mp4"
    final_path = output_dir / "output_512.mp4"
    vis_path = output_dir / "visualized_512.mp4"

    writer_raw = cv2.VideoWriter(str(tmp_path), cv2.VideoWriter_fourcc(*'mp4v'), 8, (target_size, target_size))
    writer_vis = cv2.VideoWriter(str(output_dir / "vis_tmp.mp4"), cv2.VideoWriter_fourcc(*'mp4v'), 8, (target_size, target_size))
    color_map = {}

    def get_color(obj_id):
        if obj_id not in color_map:
            random.seed(int(obj_id))
            color_map[obj_id] = np.array([
                random.randint(60, 255),
                random.randint(60, 255),
                random.randint(60, 255),
            ], dtype=np.uint8)
        return color_map[obj_id]

    for i, idx in enumerate(relevant_indices):
        frame = video_frames[idx]
        # Crop and Resize Frame
        crop_img = frame[crop_y1:crop_y1+final_side, crop_x1:crop_x1+final_side]
        resized_img = cv2.resize(crop_img, (target_size, target_size))

        # Save Frame JPG
        Image.fromarray(resized_img).save(frames_dir / f"frame_{i:02d}.jpg")

        # Combine all masks for the "layout mask"
        combined_mask = np.zeros((H, W), dtype=np.uint8)
        frame_data = outputs_per_frame.get(idx, {})
        vis_overlay = resized_img.copy()

        for obj_id, mask in frame_data.items():
            combined_mask[mask] = 255
            # For visualization
            m_crop = mask[crop_y1:crop_y1+final_side, crop_x1:crop_x1+final_side]
            m_crop = m_crop.astype(np.uint8) * 255
            m_resized = cv2.resize(m_crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
            vis_overlay[m_resized > 0] = get_color(obj_id)
        vis_overlay = (alpha * vis_overlay + (1 - alpha) * resized_img).astype(np.uint8)
        # Save Layout Mask
        mask_crop = combined_mask[crop_y1:crop_y1+final_side, crop_x1:crop_x1+final_side]
        mask_resized = cv2.resize(mask_crop, (target_size, target_size), interpolation=cv2.INTER_NEAREST)
        Image.fromarray(mask_resized).save(masks_dir / f"mask_{i:02d}.png")

        # Write Videos
        writer_raw.write(cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR))
        writer_vis.write(cv2.cvtColor(vis_overlay, cv2.COLOR_RGB2BGR))

    writer_raw.release()
    writer_vis.release()

    # FFmpeg Re-encode for compatibility
    for f_in, f_out in [(tmp_path, final_path), (output_dir / "vis_tmp.mp4", vis_path)]:
        subprocess.run([
            "ffmpeg", "-y", "-i", str(f_in), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(f_out)
        ], capture_output=True)
        f_in.unlink()

process_video_assets(video_frames, outputs_per_frame, num_frames=-1, target_size=512)
predictor.shutdown()
print(f"Processing complete. Assets saved to: {output_dir}")