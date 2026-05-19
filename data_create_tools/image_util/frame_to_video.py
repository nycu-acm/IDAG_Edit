import os
import glob
import numpy as np
import torch
import torchvision.transforms as T
from torchvision.io import write_video
from einops import rearrange
from PIL import Image

def process_frames(frames, h, w):
    fh, fw = frames.shape[-2:]
    h = int(np.floor(h / 64.0)) * 64
    w = int(np.floor(w / 64.0)) * 64

    nw = int(fw / fh * h)
    if nw >= w:
        size = (h, nw)
    else:
        size = (int(fh / fw * w), w)

    assert len(frames.shape) >= 3
    if len(frames.shape) == 3:
        frames = [frames]

    print(f"[INFO] frame size {(fh, fw)} resize to {size} and centercrop to {(h, w)}")

    frame_ls = []
    for frame in frames:
        resized_frame = T.Resize(size, interpolation=T.InterpolationMode.BILINEAR)(frame)
        # cropped_frame = T.CenterCrop([h, w])(resized_frame)
        cropped_frame = T.FiveCrop([h, w])(resized_frame)[0]
        frame_ls.append(cropped_frame)
    return torch.stack(frame_ls)

def load_frames_from_directory(image_path, start_frame, end_frame, sample_rate, h, w):
    filenames = sorted([os.path.join(image_path, image) for image in os.listdir(image_path) if image.endswith(".png") or image.endswith(".jpg")])
    if not filenames:
        raise ValueError("No images found in the specified directory!")

    # If end_frame is not specified, default to the last image
    end_frame += 1
    if end_frame is None or end_frame > len(filenames):
        end_frame = len(filenames)
        
    selected_filenames = filenames[start_frame:end_frame:sample_rate]
    if not selected_filenames:
        raise ValueError("No images selected based on the provided range and sample rate!")
    
    frame_ls = []
    for frame_path in selected_filenames:
        image = Image.open(frame_path).convert('RGB')
        image = T.ToTensor()(image).unsqueeze(0)
        frame_ls.append(image)
    frames = torch.cat(frame_ls)
    
    frames_tensor = process_frames(frames, h, w)
    print(f"[INFO] 成功載入 {frames_tensor.shape[0]} 個frame，尺寸: {frames_tensor.shape}")
    return frames_tensor

def save_video(frames: torch.Tensor, output_path, frame_ids=None, fps=16):
    os.makedirs(output_path, exist_ok=True)
    if frame_ids is None:
        frame_ids = [i for i in range(len(frames))]
    frames = frames[frame_ids]

    proc_frames = (rearrange(frames, "T C H W -> T H W C") * 255).to(torch.uint8).cpu()
    write_video(os.path.join(output_path, "output.mp4"), proc_frames, fps = fps, video_codec="h264")
    print(f"[INFO] save video to {os.path.join(output_path, 'output.mp4')}")

def process_frames_to_video(input_path, output_path, start_frame, end_frame, sample_rate, h, w, fps=16):
    frames = load_frames_from_directory(input_path, start_frame, end_frame, sample_rate, h, w)     
    save_video(frames, output_path, fps=fps)
    return frames

if __name__ == "__main__":
    input_dir = "data"
    output_video = "output"
    
    start_frame = 0
    end_frame = 15
    sample_rate = 1
    
    height = 512
    width = 512 
    fps = 8
    
    frames_tensor = process_frames_to_video(
        input_path=input_dir,
        output_path=output_video,
        start_frame=start_frame,
        end_frame=end_frame,
        sample_rate=sample_rate,
        h=height,
        w=width,
        fps=fps
    )