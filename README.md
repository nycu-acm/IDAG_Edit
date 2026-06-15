# IDAG_Edit: Multi-Object Video Editing via Instance-Decoupled Attention and Guidance (ICIP 2026)

> Official implementation of **IDAG_Edit**, a *training-free* video editing framework that overcomes semantic leakage in multi-object video editing through instance-aware attention and guidance mechanisms.
>
> Yuan-Zhih Lin, Huu-Thang Nguyen, Huu-Phu Do, Hong-Han Shuai, Ching-Chun Huang
>
> Department of Computer Science, National Yang Ming Chiao Tung University Taiwan

<p align="center">
  <a href="https://louislin0128.github.io/IDAG_Edit_Page//" target="_blank">
    <img src="https://img.shields.io/badge/Project%20Page-IDAG_Edit-blue?style=for-the-badge" />
  </a>
</p>

## 🏗️ Overview
![](assets/Figure2_Overview.jpg)

---

## 🛠️ Installations (python==3.11.14 recommended)

### Setup repository and conda environment

```
git clone https://github.com/nycu-acm/IDAG_Edit.git 
cd IDAG_Edit

conda env create -f environment.yaml
conda activate IDAG_Edit
pip install torch==2.7.0+cu128 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128
```

## 📦 Preprocess
We provides sample videos and masks under `resources/`, and also includes `data_create_tools/` for preprocessing-related utilities.

### Mask prediction
You can prepare instance masks in two ways:
1. Use the provided sample masks under `resources/sam2_mask/`.
2. Generate masks with the preprocessing utilities under `data_create_tools/`.

If you want to build masks from your own videos, use the data creation pipeline in `data_create_tools/` to extract frames and generate object-aware masks before running editing.

## 📥 Download pretrained model
### Download Stable Diffusion v1.5
Download the Stable Diffusion v1.5 backbone from [Hugginface](https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5)
and place it at:
```
models/StableDiffusion/stable-diffusion-v1-5
```

### Prepare Community Models
Manually download the community `.safetensors` checkpoint from [RealisticVision](https://civitai.com/models/4201?modelVersionId=130072)
and place it at:
```
models/DreamBooth_LoRA/realisticVisionV60B1_v51VAE.safetensors
```

### Prepare AnimateDiff Motion Modules
Manually download the AnimateDiff motion module from [AnimateDiff](https://github.com/guoyww/AnimateDiff)
and place it at:
```
models/Motion_Module/v3_sd15_mm.ckpt
```


## 🎬 Edited Video
### Run our method: 
```
bash run_editing.sh
```
Our editing config file is in `editing_config_yaml/run_two_man_editing.yaml`.

## 📂 Project Structure

```text
IDAG_Edit/
├── IDAG_Edit/                  # Core package
│   ├── models/
│   ├── pipelines/
│   └── utils/
│
├── configs/                    # Model-level configuration files
├── data_create_tools/          # Preprocessing and mask-generation tools
├── editing_config_yaml/        # Example editing configurations
│   └── run_two_man_editing.yaml
│
├── models/                     # Pretrained checkpoints
│   ├── StableDiffusion/
│   ├── DreamBooth_LoRA/
│   └── Motion_Module/
│
├── p2p_module/                # Prompt-to-Prompt related modules
├── resources/                 # Example videos and masks
│   ├── *.mp4
│   └── sam2_mask/
│
├── run_IDAG_Edit.py           # Main entry script
├── run_editing.sh             # Quick-start script
├── environment.yaml
├── LICENSE
```


## Acknowledgements

This work was financially supported in part (project number:112UA10019) by the Co-creation Platform of the Industry Academia Innovation School, NYCU, under the framework of the National Key Fields Industry-University Cooperation and Skilled Personnel Training Act, 
from the Ministry of Education (MOE) and industry partners in Taiwan. It also supported in part by the National Science and Technology Council, Taiwan, under Grant NSTC-115-2634-FA49-011-, NSTC-114-2218-E-A49-024-, Grant NSTC-1122221-E-A49-089-MY3, Grant NSTC-115-2425-H-A49-001,
Grant NSTC-114-2622-E-A49-027, Grant NSTC-112-2221E-A49-092-MY3, and in part by the Higher Education Sprout Project of the National Yang Ming Chiao Tung University and the Ministry of Education (MOE), Taiwan. 
It is also partly supported by MediaTek Inc., Hon Hai Research Institute, and Industrial Technology Research Institute.
