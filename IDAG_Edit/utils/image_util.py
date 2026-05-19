import os
import math
import textwrap

import imageio
import numpy as np
from typing import Sequence
import requests
import cv2
from PIL import Image, ImageDraw, ImageFont

import torch
from torchvision import transforms
from einops import rearrange
import torchvision
import imageio

import torchvision.transforms.functional as F
import random
from scipy.ndimage import binary_dilation
import sys

IMAGE_EXTENSION = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp", ".JPEG")

FONT_URL = "https://raw.github.com/googlefonts/opensans/main/fonts/ttf/OpenSans-Regular.ttf"
FONT_PATH = "./docs/OpenSans-Regular.ttf"

np.random.seed(200)
_palette = ((np.random.random((3*255))*0.7+0.3)*255).astype(np.uint8).tolist()
_palette = [0,0,0]+_palette

def save_prediction(pred_mask,output_dir,file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir,file_name))
def colorize_mask(pred_mask):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)
def draw_mask(img, mask, alpha=0.5, id_countour=False):
    img_mask = np.zeros_like(img)
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]

        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)

            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]

            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
        
    return img_mask.astype(img.dtype)




def pad(image: Image.Image, top=0, right=0, bottom=0, left=0, color=(255, 255, 255)) -> Image.Image:
    new_image = Image.new(image.mode, (image.width + right + left, image.height + top + bottom), color)
    new_image.paste(image, (left, top))
    return new_image


def download_font_opensans(path=FONT_PATH):
    font_url = FONT_URL
    response = requests.get(font_url)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(response.content)


def annotate_image_with_font(image: Image.Image, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    image_w = image.width
    _, _, text_w, text_h = font.getbbox(text)
    line_size = math.floor(len(text) * image_w / text_w)

    lines = textwrap.wrap(text, width=line_size)
    padding = text_h * len(lines)
    image = pad(image, top=padding + 3)

    ImageDraw.Draw(image).text((0, 0), "\n".join(lines), fill=(0, 0, 0), font=font)
    return image


def annotate_image(image: Image.Image, text: str, font_size: int = 15):
    if not os.path.isfile(FONT_PATH):
        download_font_opensans()
    font = ImageFont.truetype(FONT_PATH, size=font_size)
    return annotate_image_with_font(image=image, text=text, font=font)


def make_grid(images: Sequence[Image.Image], rows=None, cols=None) -> Image.Image:
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(i) for i in images]

    if rows is None:
        assert cols is not None
        rows = math.ceil(len(images) / cols)
    else:
        cols = math.ceil(len(images) / rows)

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))
    for i, image in enumerate(images):
        if image.size != (w, h):
            image = image.resize((w, h))
        grid.paste(image, box=(i % cols * w, i // cols * h))
    return grid


def save_images_as_gif(
    images: Sequence[Image.Image],
    save_path: str,
    loop=0,
    duration=100,
    optimize=False,
) -> None:

    images[0].save(
        save_path,
        save_all=True,
        append_images=images[1:],
        optimize=optimize,
        loop=loop,
        duration=duration,
    )

def save_images_as_mp4(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:

    writer_edit = imageio.get_writer(
        save_path,
        fps=10)
    for i in images:
        init_image = i.convert("RGB")
        writer_edit.append_data(np.array(init_image))
    writer_edit.close()


def save_tensor_images_and_video(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=10):
    os.makedirs(path, exist_ok=True)
    
    # Rearrange video tensor for easier processing
    videos = rearrange(videos, "b c t h w -> t b c h w")

    # Lists to store each frame for saving as images and creating a video
    frame_list_for_images = []

    for i, x in enumerate(videos):
        # Create a grid of images for this frame
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        
        if rescale:
            x = (x + 1.0) / 2.0  # Rescale from [-1, 1] to [0, 1]
        
        x = (x * 255).numpy().astype(np.uint8)

        # Save individual frame as image
        save_path_image = os.path.join(path, f"{i}.jpg")
        imageio.imsave(save_path_image, x)

        # Append to frame lists
        frame_list_for_images.append(x)

    # Save the frames as a video
    save_path_video = os.path.join(path, "control.mp4")
    imageio.mimwrite(save_path_video, frame_list_for_images, fps=fps)

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=4, fps=8):
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    for x in videos:
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    imageio.mimsave(path, outputs, fps=fps)


def save_images_as_folder(
    images: Sequence[Image.Image],
    save_path: str,
) -> None:
    os.makedirs(save_path, exist_ok=True)
    for index, image in enumerate(images):
        init_image = image
        if len(np.array(init_image).shape) == 3:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.jpg"), np.array(init_image)[:, :, ::-1])
        else:
            cv2.imwrite(os.path.join(save_path, f"{index:05d}.jpg"), np.array(init_image))

def log_infer_samples(
    infer_dataloader,
    save_path,
    num_batch: int = 4,
    fps: int = 8,
    save_input=True,
):
    infer_samples = []
    infer_masks = []
    infer_merge_masks = []
    for idx, batch in enumerate(infer_dataloader):
        if idx >= num_batch:
            break
        infer_samples.append(batch["images"])
        infer_masks.append(batch["layouts"])
        infer_merge_masks.append(batch["masks"])

    infer_samples = torch.cat(infer_samples).numpy()
    _,_,frames,height,width = infer_samples.shape
    infer_samples = rearrange(infer_samples, "b c f h w -> b f h w c")
    print('infer_samples',infer_samples.shape)
    infer_samples = (infer_samples * 0.5 + 0.5).clip(0, 1)
    # infer_samples = numpy_batch_seq_to_pil(infer_samples)
    # infer_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(infer_samples))))) for images in zip(*infer_samples)]
    infer_merge_masks = torch.cat(infer_merge_masks).unsqueeze(0)
    infer_masks = torch.cat(infer_masks)
    # f, s, c, h ,w

    infer_masks = rearrange(infer_masks.squeeze(2), "f s h w -> f h w s")

    # 添加一个全为0的mask到第0维
    zero_mask = torch.zeros(infer_masks.shape[0], infer_masks.shape[1], infer_masks.shape[2], 1)
    infer_masks = torch.cat((zero_mask, infer_masks), dim=-1)
    infer_masks = torch.argmax(infer_masks, axis=-1).numpy()

    masked_frames = []
    for frame_idx in range(frames):
        image = np.array(infer_samples[0][frame_idx])
        mask = infer_masks[frame_idx]
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)

        image = (image * 255).astype(np.uint8)
        masked_frame = draw_mask(image, mask, id_countour=False)
        masked_frames.append(masked_frame)
    #infer_samples_save = rearrange(torch.tensor(infer_samples),'b t h w c -> b c t h w')
    if save_input:
        infer_samples = numpy_batch_seq_to_pil(infer_samples)
        infer_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(infer_samples))))) for images in zip(*infer_samples)]
        save_gif_mp4_folder_type(infer_samples, os.path.join(save_path, 'input.gif'))
    imageio.mimsave(os.path.join(save_path, 'masked_video.mp4'),masked_frames,fps=fps)
    save_videos_grid(infer_merge_masks,os.path.join(save_path, 'merged_masks.mp4'), fps=fps)



def log_train_samples(
    train_dataloader,
    save_path,
    num_batch: int = 4,
):
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= num_batch:
            break
        train_samples.append(batch["images"])

    train_samples = torch.cat(train_samples).numpy()
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = (train_samples * 0.5 + 0.5).clip(0, 1)
    train_samples = numpy_batch_seq_to_pil(train_samples)
    train_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(train_samples))))) for images in zip(*train_samples)]
    # save_images_as_gif(train_samples, save_path)
    save_gif_mp4_folder_type(train_samples, save_path)

def log_train_reg_samples(
    train_dataloader,
    save_path,
    num_batch: int = 4,
):
    train_samples = []
    for idx, batch in enumerate(train_dataloader):
        if idx >= num_batch:
            break
        train_samples.append(batch["class_images"])

    train_samples = torch.cat(train_samples).numpy()
    train_samples = rearrange(train_samples, "b c f h w -> b f h w c")
    train_samples = (train_samples * 0.5 + 0.5).clip(0, 1)
    train_samples = numpy_batch_seq_to_pil(train_samples)
    train_samples = [make_grid(images, cols=int(np.ceil(np.sqrt(len(train_samples))))) for images in zip(*train_samples)]
    # save_images_as_gif(train_samples, save_path)
    save_gif_mp4_folder_type(train_samples, save_path)


def save_gif_mp4_folder_type(images, save_path, save_gif=True):
    if isinstance(images[0], np.ndarray):
        images = [Image.fromarray(i) for i in images]
    elif isinstance(images[0], torch.Tensor):
        images = [transforms.ToPILImage()(i.cpu().clone()[0]) for i in images]
    save_path_mp4 = save_path.replace('gif', 'mp4')
    save_path_folder = save_path.replace('.gif', '')
    # os.makedirs(save_path_folder, exist_ok=True)
    # if save_gif: save_images_as_gif(images, save_path)
    # save_images_as_mp4(images, save_path_mp4)
    save_images_as_folder(images, save_path_folder)

# copy from video_diffusion/pipelines/stable_diffusion.py
def numpy_seq_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    if images.shape[-1] == 1:
        # special case for grayscale (single channel) images
        pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
    else:
        pil_images = [Image.fromarray(image) for image in images]

    return pil_images

# copy from diffusers-0.11.1/src/diffusers/pipeline_utils.py
def numpy_batch_seq_to_pil(images):
    pil_images = []
    for sequence in images:
        pil_images.append(numpy_seq_to_pil(sequence))
    return pil_images


def downsample_image(image, target_size):
    image = Image.fromarray(image)
    resized_image = image.resize(target_size, Image.ANTIALIAS)
    return np.array(resized_image)

def visualize_check_downsample_keypoints(images, keypoint_data, target_res=(32, 32),final_res=(512, 512)):
    # 预处理帧列表
    processed_frames = []

    # 遍历每一帧
    for frame_idx, frame_tensor in enumerate(images):
        # 将张量转换为 NumPy 数组
        # print("frame",frame_tensor.shape)
        frame = frame_tensor.cpu().numpy().astype('uint8')

        # 下采样图片
        downsampled_frame = downsample_image(frame, target_res)

        # 绘制关键点
        for keypoint in keypoint_data[frame_idx]:
            h_coord, w_coord = keypoint
            downsampled_frame[h_coord, w_coord] = [255, 0, 0]  # 使用红色标记关键点

        # 将处理过的帧重新调整到最终分辨率
        final_frame = downsample_image(downsampled_frame, final_res)

        # 将处理过的帧添加到列表中
        processed_frames.append(final_frame)

    # 使用 imageio 保存处理过的帧为视频
    output_video_path = "./down_sample_check_hockey.mp4"
    imageio.mimsave(output_video_path, processed_frames, fps=10)


"""optical flow and trajectories sampling"""
def preprocess(img1_batch, img2_batch, transforms, height,width):
    img1_batch = F.resize(img1_batch, size=[height, width], antialias=False)
    img2_batch = F.resize(img2_batch, size=[height, width], antialias=False)
    return transforms(img1_batch, img2_batch)

def keys_with_same_value(dictionary):
    result = {}
    for key, value in dictionary.items():
        if value not in result:
            result[value] = [key]
        else:
            result[value].append(key)

    conflict_points = {}
    for k in result.keys():
        if len(result[k]) > 1:
            conflict_points[k] = result[k]
    return conflict_points

def find_duplicates(input_list):
    seen = set()
    duplicates = set()

    for item in input_list:
        if item in seen:
            duplicates.add(item)
        else:
            seen.add(item)

    return list(duplicates)

def neighbors_index(point, window_size, H, W):
    """return the spatial neighbor indices"""
    t, x, y = point
    neighbors = []
    for i in range(-window_size, window_size + 1):
        for j in range(-window_size, window_size + 1):
            if i == 0 and j == 0:
                continue
            if x + i < 0 or x + i >= H or y + j < 0 or y + j >= W:
                continue
            neighbors.append((t, x + i, y + j))
    return neighbors



@torch.no_grad()
def sample_trajectories(video_path, device,height,width):
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")

    clips = list(range(len(frames)))

    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()

    finished_trajectories = []

    current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], transforms, 512,512)
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    predicted_flows = list_of_flows[-1]
    print('predicted_flows',predicted_flows.shape)
    predicted_flows = predicted_flows/512

    resolutions = [64, 32, 16, 8]
    res = {}
    window_sizes = {64: 2,
                    32: 1,
                    16: 1,
                    8: 1}

    for resolution in resolutions:
        print("="*30)
        trajectories = {}
        predicted_flow_resolu = torch.round(resolution*torch.nn.functional.interpolate(predicted_flows, scale_factor=(resolution/512, resolution/512)))

        T = predicted_flow_resolu.shape[0]+1
        H = predicted_flow_resolu.shape[2]
        W = predicted_flow_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            flow = predicted_flow_resolu[t]
            for h in range(H):
                for w in range(W):

                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        # this point has not been traversed, start new trajectory
                        x = h + int(flow[1, h, w])
                        y = w + int(flow[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            # trajectories.append([(t, h, w), (t+1, x, y)])
                            trajectories[(t, h, w)]= (t+1, x, y)

        conflict_points = keys_with_same_value(trajectories)
        for k in conflict_points:
            index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
            conflict_points[k].pop(index_to_pop)
            for point in conflict_points[k]:
                if point[0] != T-1:
                    trajectories[point]= (-1, -1, -1) # stupid padding with (-1, -1, -1)

        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x//H, x%H):0 for x in range(H*W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj)
            active_traj = new_active_traj
            active_traj+=[[pixel] for pixel in pixel_set if pixel_set[pixel] == 0]
        all_traj += active_traj

        useful_traj = [i for i in all_traj if len(i)>1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1):
                useful_traj[idx] = useful_traj[idx][:-1]
        print("how many points in all trajectories for resolution{}?".format(resolution), sum([len(i) for i in useful_traj]))
        print("how many points in the video for resolution{}?".format(resolution), T*H*W)

        # validate if there are no duplicates in the trajectories
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        assert len(find_duplicates(trajs)) == 0, "There should not be duplicates in the useful trajectories."

        # check if non-appearing points + appearing points = all the points in the video
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        left_points = all_points- set(trajs)
        print("How many points not in the trajectories for resolution{}?".format(resolution), len(left_points))
        for p in list(left_points):
            useful_traj.append([p])
        print("how many points in all trajectories for resolution{} after pending?".format(resolution), sum([len(i) for i in useful_traj]))


        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)
                    sequence = [(t,x,y)]+neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t,x,y)].copy()
                    traj.remove((t,x,y))
                    sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res["traj{}".format(resolution)] = seqs
        res["mask{}".format(resolution)] = masks
    return res


@torch.no_grad()
def sample_trajectories_new(video_path, device,height,width):
    from torchvision.models.optical_flow import Raft_Large_Weights
    from torchvision.models.optical_flow import raft_large

    weights = Raft_Large_Weights.DEFAULT
    transforms = weights.transforms()

    frames, _, _ = torchvision.io.read_video(str(video_path), output_format="TCHW")

    clips = list(range(len(frames)))
    
    #=============== raft-large estimate forward optical flow============#
    model = raft_large(weights=Raft_Large_Weights.DEFAULT, progress=False).to(device)
    model = model.eval()
    finished_trajectories = []

    current_frames, next_frames = preprocess(frames[clips[:-1]], frames[clips[1:]], transforms, height,width)
    list_of_flows = model(current_frames.to(device), next_frames.to(device))
    predicted_flows = list_of_flows[-1]
    #=============== raft-large estimate forward optical flow============#

    predicted_flows = predicted_flows/max(height,width)

    resolutions =[(height//8,width//8),(height//16,width//16),(height//32,width//32),(height//64,width//64)]
    #resolutions = [64, 32, 16, 8]
    res = {}
    window_sizes = {(height//8,width//8): 2,
                    (height//16,width//16): 1,
                    (height//32,width//32): 1,
                    (height//64,width//64): 1}
    
    for resolution in resolutions:
        print("="*30)
        # print(resolution)
        # print('window_sizes[resolution]',window_sizes[resolution])
        trajectories = {}
        height_scale_factor = resolution[0] / height
        width_scale_factor = resolution[1] / width
        predicted_flow_resolu = torch.round(max(resolution[0], resolution[1])*torch.nn.functional.interpolate(predicted_flows, scale_factor=(height_scale_factor, width_scale_factor)))

        T = predicted_flow_resolu.shape[0]+1
        H = predicted_flow_resolu.shape[2]
        W = predicted_flow_resolu.shape[3]

        is_activated = torch.zeros([T, H, W], dtype=torch.bool)

        for t in range(T-1):
            flow = predicted_flow_resolu[t]
            for h in range(H):
                for w in range(W):

                    if not is_activated[t, h, w]:
                        is_activated[t, h, w] = True
                        # this point has not been traversed, start new trajectory
                        x = h + int(flow[1, h, w])
                        y = w + int(flow[0, h, w])
                        if x >= 0 and x < H and y >= 0 and y < W:
                            # trajectories.append([(t, h, w), (t+1, x, y)])
                            trajectories[(t, h, w)]= (t+1, x, y)

        conflict_points = keys_with_same_value(trajectories)
        for k in conflict_points:
            index_to_pop = random.randint(0, len(conflict_points[k]) - 1)
            conflict_points[k].pop(index_to_pop)
            for point in conflict_points[k]:
                if point[0] != T-1:
                    trajectories[point]= (-1, -1, -1) # stupid padding with (-1, -1, -1)

        active_traj = []
        all_traj = []
        for t in range(T):
            pixel_set = {(t, x//H, x%H):0 for x in range(H*W)}
            new_active_traj = []
            for traj in active_traj:
                if traj[-1] in trajectories:
                    v = trajectories[traj[-1]]
                    new_active_traj.append(traj + [v])
                    pixel_set[v] = 1
                else:
                    all_traj.append(traj)
            active_traj = new_active_traj
            active_traj+=[[pixel] for pixel in pixel_set if pixel_set[pixel] == 0]
        all_traj += active_traj

        useful_traj = [i for i in all_traj if len(i)>1]
        for idx in range(len(useful_traj)):
            if useful_traj[idx][-1] == (-1, -1, -1):
                useful_traj[idx] = useful_traj[idx][:-1]
        print("how many points in all trajectories for resolution{}?".format(resolution), sum([len(i) for i in useful_traj]))
        print("how many points in the video for resolution{}?".format(resolution), T*H*W)

        # validate if there are no duplicates in the trajectories
        trajs = []
        for traj in useful_traj:
            trajs = trajs + traj
        assert len(find_duplicates(trajs)) == 0, "There should not be duplicates in the useful trajectories."

        # check if non-appearing points + appearing points = all the points in the video
        all_points = set([(t, x, y) for t in range(T) for x in range(H) for y in range(W)])
        left_points = all_points- set(trajs)
        print("How many points not in the trajectories for resolution{}?".format(resolution), len(left_points))
        for p in list(left_points):
            useful_traj.append([p])
        print("how many points in all trajectories for resolution{} after pending?".format(resolution), sum([len(i) for i in useful_traj]))


        longest_length = max([len(i) for i in useful_traj])
        sequence_length = (window_sizes[resolution]*2+1)**2 + longest_length - 1

        seqs = []
        masks = []

        # create a dictionary to facilitate checking the trajectories to which each point belongs.
        point_to_traj = {}
        for traj in useful_traj:
            for p in traj:
                point_to_traj[p] = traj

        for t in range(T):
            for x in range(H):
                for y in range(W):
                    neighbours = neighbors_index((t,x,y), window_sizes[resolution], H, W)
                    sequence = [(t,x,y)]+neighbours + [(0,0,0) for i in range((window_sizes[resolution]*2+1)**2-1-len(neighbours))]
                    sequence_mask = torch.zeros(sequence_length, dtype=torch.bool)
                    sequence_mask[:len(neighbours)+1] = True

                    traj = point_to_traj[(t,x,y)].copy()
                    traj.remove((t,x,y))
                    sequence = sequence + traj + [(0,0,0) for k in range(longest_length-1-len(traj))]
                    sequence_mask[(window_sizes[resolution]*2+1)**2: (window_sizes[resolution]*2+1)**2 + len(traj)] = True

                    seqs.append(sequence)
                    masks.append(sequence_mask)

        seqs = torch.tensor(seqs)
        masks = torch.stack(masks)
        res["traj{}".format(resolution[0])] = seqs
        res["mask{}".format(resolution[0])] = masks
    return res