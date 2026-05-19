import os
import re
import json
import glob
import base64
import argparse
import numpy as np
from typing import Any
from pathlib import Path
from gpt_instruction import cot_prompt

parent_dir = os.path.dirname(os.path.abspath(__file__))
from image_util.sample_video2frames import extract_frames
from sam3.inference_test_new import sam3_inference
from openai import OpenAI

# os.environ["OPENAI_API_KEY"] = 'your-api-key'
assert "OPENAI_API_KEY" in os.environ, "Please set the OPENAI_API_KEY environment variable."

def encode_image_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")

def gpt_eval(prompt: str, frames: list) -> str:
    # api_key = ""
    # client = OpenAI(api_key = api_key)
    client = OpenAI()
    content =[{"type": "text", "text": prompt}]
    
    for idx, frame_path in enumerate(frames):
        b64 = encode_image_to_base64(frame_path)
        content.append({
            "type": "image_url", 
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })
    
    response = client.chat.completions.create(
    # response = client.beta.chat.completions.parse(
        model="gpt-4.1",
        messages=[{
            "role": "user", 
            "content": content
        }],
        # max_tokens=1000,
        seed=42
    )
    return response.choices[0].message.content

def frame_sampling(all_frame_paths: list, num_frames_to_use: int = None, sampling: str = "uniform") -> list:
    if num_frames_to_use and num_frames_to_use < len(all_frame_paths):
        if sampling == 'uniform':
            indices = np.linspace(0, len(all_frame_paths) - 1, num_frames_to_use, dtype=int)
            frame_paths = [all_frame_paths[i] for i in indices]
        elif sampling == 'first_last':
            first_half = num_frames_to_use // 2
            last_half = num_frames_to_use - first_half
            frame_paths = all_frame_paths[:first_half] + all_frame_paths[-last_half:]
        else:
            raise ValueError(f"Invalid frame sampling method: {sampling}")
    else:
        frame_paths = all_frame_paths
    
    return frame_paths

def read_object_list(txt_file: str) -> dict[str, str]:
    """Read object list from txt file and return as dictionary"""
    object_dict = {}
    
    with open(txt_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:  # 跳過空行
                continue
            
            parts = line.split(',', 1) 
            if len(parts) == 2:
                subdir_name = parts[0].strip()
                object_list = parts[1].strip().strip('"') 
                object_dict[subdir_name] = object_list
    
    return object_dict

def parse_edit_pairs(object_list: str) -> dict[str, str]:
    """Parse an edit-pair string into a {source: target} dict.
    Example
    -------
    Input : "leftman to Ironman, rightman to Hulk"
    Output: {"leftman": "Ironman", "rightman": "Hulk"}
    """
    pairs: dict[str, str] = {}
    for item in object_list.split(','):
        item = item.strip()
        if ' to ' in item:
            src, tgt = item.split(' to ', 1)
            pairs[src.strip()] = tgt.strip()
        else:
            print(f"Warning: could not parse edit pair '{item}', skipping.")
    return pairs

def build_sam_text_prompts(
    edit_pairs: dict[str, str],
    gpt_object_prompts: list[dict],
) -> dict[str, str]:
    """Merge edit pairs with GPT-refined source descriptions.
 
    Parameters
    ----------
    edit_pairs:
        Ordered dict of {src_name: tgt_name}, e.g. {"leftman": "Ironman"}.
    gpt_object_prompts:
        List of dicts from GPT result["object_prompts"], e.g.
        [{"original": "man on the left side", "edited": "Ironman"}, ...].
 
    Returns
    -------
    text_prompts dict ready for sam3_inference:
        {"leftman": "man on the left side", "rightman": "man on the right side"}
    """
    text_prompts: dict[str, str] = {}
 
    for idx, (src_name, tgt_name) in enumerate(edit_pairs.items()):
        if idx < len(gpt_object_prompts):
            # Use GPT-refined description as the SAM query for better grounding
            sam_key = gpt_object_prompts[idx].get("original", tgt_name)
        else:
            sam_key = tgt_name  # fallback to raw name from the txt file
 
        text_prompts[src_name] = sam_key
 
    return text_prompts

def run_gpt_captioning(frame_path: str, object_list: str,num_frames_to_use: int = None, sampling: str = "uniform") -> dict[str, Any]:   
    frame_files = []
    vid_path = os.path.join(parent_dir, frame_path)
    image_extensions = ['.jpg', '.jpeg', '.png']
    for ext in image_extensions:
        frame_files.extend(glob.glob(os.path.join(vid_path, f"*{ext}")))
    
    frame_files.sort(key=lambda x: int(re.search(r'(\d+)', os.path.basename(x)).group()))
    # print(f"Found {len(frame_files)} frames in {frame_path}")
    
    frame_files = frame_sampling(frame_files, num_frames_to_use, sampling)
    # src_prompt = cot_prompt.replace("src", source_prompt)
    formatted_prompt = cot_prompt.replace("[src_o to edit_o]", object_list)
    rsp = gpt_eval(formatted_prompt, frame_files)
    # print(f"Response: {rsp}")
    
    try:
        if not (rsp.strip().startswith('{') and rsp.strip().endswith('}')):
            rsp = "{" + rsp + "}"
        result = json.loads(rsp)
    except json.JSONDecodeError:
        print("Warning: Failed to parse model response as JSON. Using raw text instead.")
        result = {"raw_response": rsp}

    return result

def process_all_subdirs(data_dir: str, object_list_path: str, num_frames_to_use: int = None, sampling: str = "uniform"):
    """Process all subdirectories in data folder"""
    object_dict = read_object_list(object_list_path)
    all_results = {}
    
    for subdir_name, object_list in object_dict.items():
        # ── 1. Validate directories ──────────────────────────────────────────
        subdir_path = os.path.join(data_dir, subdir_name)
        if not os.path.isdir(subdir_path):
            raise ValueError(f"Directory not found: {subdir_path}")
 
        frames_path = os.path.join(subdir_path, subdir_name)
        if not os.path.exists(frames_path):
            raise ValueError(f"No frames folder found in {subdir_name}")
 
        # ── 2. GPT captioning ────────────────────────────────────────────────
        print(f"[GPT] Processing '{subdir_name}' | edits: {object_list}")
        result = run_gpt_captioning(frames_path, object_list, num_frames_to_use, sampling)
        all_results[subdir_name] = result
        
        # ── 3. Extract GPT object_prompts (original descriptions) ────────────
        gpt_object_prompts: list[dict] = []
        if "object_prompts" in result and isinstance(result["object_prompts"], list):
            for match in result["object_prompts"]:
                gpt_object_prompts.append(match)   # keep full entry; "original" used later
 
        # ── 4. Parse edit pairs from the txt file ────────────────────────────
        edit_pairs = parse_edit_pairs(object_list)
 
        # ── 5. Build SAM text_prompts ────────────────────────────────────────
        text_prompts = build_sam_text_prompts(edit_pairs, gpt_object_prompts)
        
        # ── 6. Call SAM3 mask segmentation ───────────────────────────────────
        print(f"[SAM3] video='{frames_path}' | text_prompts={text_prompts}")
        sam3_inference(
            video_path=frames_path,
            text_prompts=text_prompts,
            output_dir=subdir_path
        )
    
    # ── Save all GPT results ─────────────────────────────────────────────────
    output_file = os.path.join(output_path, "results.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    print(f"All GPT results saved to {output_file}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Video editing pipeline of stage_1_gpt captioning")
    parser.add_argument('--videos_dir', type=Path, required=True, help='Directory containing videos')
    parser.add_argument('--output_dir', default='stage1_output', type=str, help="Directory to save module outputs")
    # parser.add_argument('--source_prompt', type=str, required=True, help='Source prompt about the video')
    parser.add_argument('--edit_objects', type=str, help='Comma-separated list of objects to edit')
    parser.add_argument('--edit_obj_txt', type=str, required=True, help='Path to text file containing edit object prompts')
    parser.add_argument('--num_frames', type=int, default=None, help='Number of frames to use (default: all)')
    parser.add_argument('--sampling', type=str, default='uniform', choices=['uniform', 'first_last'], 
                        help='Frame sampling method if using subset of frames')
    
    args = parser.parse_args()
    output_path = os.path.join(parent_dir, args.output_dir)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    mp4_files = sorted(args.videos_dir.rglob("*.mp4"))
    for idx, video in enumerate(mp4_files, 1):
        print(f"Processing video {idx}/{len(mp4_files)}: {video}")
        video_name = Path(video).stem
        output_frame_path = Path(output_path)/video_name/video_name
        extract_frames(video, output_frame_path)
    
    process_all_subdirs(output_path, args.edit_obj_txt, args.num_frames, args.sampling)
        