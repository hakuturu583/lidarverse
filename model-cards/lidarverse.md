<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# LidarVerse Overview

> **Current checkpoint scope:** the quickstarts use the range-only NVIDIA PAI
> tokenizer and PAI bidirectional, causal, and SF-DMD EMA
> packages, with 128x1024 inputs at 10 Hz and 2.5--100 m support. They do not
> include intensity or the paper's cross-sensor checkpoints. The broader
> architecture and dataset descriptions below describe the research project,
> not a claim that these particular packages support every sensor. Quickstart
> throughput is not a paper benchmark.

## Description:  
LidarVerse is a real-time unified LiDAR world model for sensor simulation across tasks and sensors. LidarVerse bridges both generative and reconstruction-based simulation into a single, unified conditional generative framework. It is composed of a universal, fully attention-based tokenizer maps LiDAR range sequences with different scanning patterns into a shared latent space; a bidirectional diffusion teacher learns a common interface for conditioning on a LiDAR prefix, an HD-Map, a persistent spatial cache, and optionally, text; and a four-step autoregressive causal student and maintain both a rolling temporal K/V cache and a world-aligned spatial cache, enabling persistent long-horizon roll-outs with real-time inference at 11 FPS on a single H100 GPU.  
LidarVerse was developed by NVIDIA.  
_This model is for research and development only._  


### License/Terms of Use:
CC-BY-SA-4.0
### Deployment Geography:  
Global

### Use Case:  
This model is intended for research community to study, ablate, and build upon of.  
  

### Release Date:
**GitHub:** 09/11/2026 via https://github.com/nv-tlabs/lidarverse

  

## Reference(s):
[Cosmos-Drive-Dreams: Scalable Synthetic Driving Data Generation with World Foundation Models](https://research.nvidia.com/labs/toronto-ai/cosmos_drive_dreams/)  

## Model Architecture:   
**Architecture Type:** Transformer   
**Network Architecture:** Diffusion Transformer (DiT)  
**This model was developed based on Cosmos-Predict2-T2V-2B.**  
**Number of model parameters:** 2B  

## Input:  
**Input Type(s):** Video  
**Input Format(s):** Lidar  
**Input Parameters:** Three-Dimensional (3D)  
**Other Properties Related to Input:** Flexible resolution and length  
  

## Output:  
**Output Type(s):** Video  
**Output Format:** Lidar  
**Output Parameters:** Three-Dimensional (3D)  
**Other Properties Related to Output:** Flexible resolution and length, streaming supported  
   

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA's hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions.

## Software Integration:
**Runtime Engine(s):** Not Applicable (N/A)    
**Supported Hardware Microarchitecture Compatibility:**
* NVIDIA Ampere
* NVIDIA Blackwell
* NVIDIA Hopper
* NVIDIA Lovelace

**Supported Operating System(s):** Linux  

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment.  


## Model Version(s): 
LidarVerse v1.0 — 2B-parameter trained research model

## Training, Testing, and Evaluation Datasets:  


## Training Dataset:

**Data Modality:**
* Video  
* Text  
* Image  


**Image Training Data Size:** 1 Million to 1 Billion Images  
**Text Training Data Size:** Less than a Billion Tokens  
**Video Training Data Size:** 10,000 to 1 Million Hours  
**Data Collection Method by dataset:** Automatic/Sensors  
**Labeling Method by dataset:** Hybrid: Automatic/Sensors/Manually-Labeled
**Properties:** 112,803 sensor-generated LiDAR range-sequence entries: NVIDIA PAI (111,988), PandaSet (93), and nuScenes (722). Sensor grids: NVIDIA PAI 128×1200 at 10 Hz, PandaSet 64×1024 at 10 Hz, and nuScenes 32×1024 at 20 Hz. No linguistic content.

### Testing Dataset:

**Data Collection Method by dataset:** Automatic/Sensors  
**Labeling Method by dataset:** Hybrid: Automatic/Sensors/Manually-Labeled
**Properties:** 1,319 sensor-generated LiDAR range-sequence entries: NVIDIA PAI (1,180), PandaSet (10), nuScenes (128), and KITTI (1). Sensor grids: NVIDIA PAI 128×1200 at 10 Hz, PandaSet 64×1024 at 10 Hz, nuScenes 32×1024 at 20 Hz, and KITTI 64×1024 at 10 Hz. No linguistic content.

### Evaluation Dataset:
**Benchmark Score:** Refer to Technical Report

**Data Collection Method by dataset:** Automatic/Sensors  
**Labeling Method by dataset:** Hybrid: Automatic/Sensors/Manually-Labeled
**Properties:** 202 sensor-generated LiDAR range-sequence entries from the Waymo Open Dataset validation split. Sensor grid: 64×1024 at 10 Hz. No linguistic content.



## Inference:
**Acceleration Engine:** PyTorch
**Hardware Requirements (GPU Architecture, Model):**
* "NVIDIA H100"
* "NVIDIA A100 80GB"


## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications.  Developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.  
Please make sure you have proper rights and permissions for all input image and video content; if image or video includes people, personal health information, or intellectual property, the image or video generated will not blur or maintain proportions of image subjects included.  
For more detailed information on ethical considerations for this model, please see the Model Card++ [Bias](lidarverse-subcards/bias.md), [Explainability](lidarverse-subcards/explainability.md), [Safety & Security](lidarverse-subcards/safety_security.md), and [Privacy](lidarverse-subcards/privacy.md) Subcards.
Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).

**Generated by NVIDIA Model Card Generator Toolkit.**
