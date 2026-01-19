---
date: '2026-01-05T10:58:31+09:00'
draft: false
title: 'Generating Synthetic Datasets using Isaac Sim'

tags: ["Isaac sim", "Synthetic Data"]
---
[Codes](https://github.com/Yeongseung/isaacsim-synthetic-data)
# Intro
For vision tasks in ML, we naturally need images and corresponding labels to train a model. Normally, we consider two options. searching and finding datasets we can leverage or creating datasets by taking a photos, labeling manually. The latter is rarely feasible, especially for a student or a small team, given the immense amount of time required for labeling. 

<figure class="figure-center">
  <img src="/posts/Generating_Synthetic_Datasets_using_Isaac_Sim/GUI_setup.png" width="500">
  <figcaption>Figure 1. Isaac Sim GUI</figcaption>
</figure>

Here's a third option, which is to utilize **Isaac Sim**(simulation program from NVIDIA) to generate synthetic datasets. Initially, I was kind of skeptical about whether a vision model trained on simulated images would actually perform well in real environments. However, [This post](https://developer.nvidia.com/blog/how-to-train-autonomous-mobile-robots-to-detect-warehouse-pallet-jacks-using-synthetic-data/) changed my mind. I recommend you to read it.

# Setup
In short, my goal was to develop a weed detection model and demonstrate to korean agricultural experts that synthetic data can effectively address data scarcity in the field.

<figure class="figure-center">
  <img src="/posts/Generating_Synthetic_Datasets_using_Isaac_Sim/abstract.png" width="800">
  <figcaption>Figure 2. Smart Farm Optimization Pipeline</figcaption>
</figure>

This image made by NotebookLM roughly shows the process even though there are few typos.

# 0. Asset Preparation
We curated high-quality assets from online sources, treating them as direct representations of the crops and weeds in our target agricultural site.

<figure class="figure-center">
  <img src="/posts/Generating_Synthetic_Datasets_using_Isaac_Sim/assets.png" width="800">
  <figcaption>Figure 3. Bog Marshcress, potato plants, and greenhouse.</figcaption>
</figure>

[Weed](https://fab.com/s/d0e9dc9f5225), [crop](https://skfb.ly/6ZGsA) and [Greenhouse](https://skfb.ly/oqPAn). With these assets in hand, the direction was clear. I had to build a virtual potato farm being invaded by Bog Marshcress(Weed). I imagined things like this :
<figure class="figure-center">
  <img src="/posts/Generating_Synthetic_Datasets_using_Isaac_Sim/imagined_farm.png" width="500">
  <figcaption>Figure 4. Dreamed farm depicted by Gemini</figcaption>
</figure>

# 1. Virtual Environment Creation
## 1.1. Ridges, Furrows, Potatoes
In order to make ridges and furrows, I used sine graph since they look similar.

<figure class="figure-center">
  <img src="/posts/Generating_Synthetic_Datasets_using_Isaac_Sim/farm1.png" width="500">
  <figcaption>Figure 5. Ridges, Furrows and Potatoes</figcaption>
</figure>

### 1.2. Weeds
Using Isaac Sim's Replicator feature, I automated the placement of Bog Marshcress(weed) across the field. Replicator is one of the main feature, which allows me to shuffle the environment. Instead of manually placing every weed, I can script them to pop up randomly on the slopes of the ridges, down in the furrows, or right next to the potato plants. I also randomized their density, size, and rotation.

<figure class="figure-center">
  <img src="/posts/Generating_Synthetic_Datasets_using_Isaac_Sim/farm2.png" width="700">
  <figcaption>Figure 6. weeds placed In Greenhouse, In outside.</figcaption>
</figure>

# 2. Synthetic Data Generation

## 2.1. Domain Randomization
In addition to randomizing the weed placement, it is best practice to randomize as many variables as possible to ensure model robustness, while I kept the scope manageable. the sun’s position, light color (temperature), and the structural difference between open fields and greenhouses.

<figure class="figure-center">
  <img src="/posts/Generating_Synthetic_Datasets_using_Isaac_Sim/farm3.png" width="700">
  <figcaption>Figure 7. Domain Randomization example 1 and 2.</figcaption>
</figure>

## 2.2. Drone-Perspective Data
I decided to collect images from a drone's perspective rather than moving the camera randomly. Maintaining a consistent top-down viewpoint makes it significantly easier for the vision model to learn and recognize patterns within the field.

# 3. Automated Annotation
Since this part is automated, there was surprisingly little for me to do. The real advantage of using a simulator is that labels for tons of drone-captured images are generated without bordering me. Isaac sim recognizes the unique ID of every object, whether it's potato plant or a weed, to automatically produce labels.

<figure class="figure-center">
  <img src="/posts/Generating_Synthetic_Datasets_using_Isaac_Sim/labels.png" width="700">
  <figcaption>Figure 8. Bounding box and segmentation</figcaption>
</figure>

## 4. Vision Model Training
I simply used YOLOv11s model for this detecting task. Isaac Sim supports custom writers, so even though YOLO requires a specific format for inputs, we can handle this by writing a writer. Of course, writing a writer is time-cunsuming work, but thanks to [this repository](https://github.com/Neubotech-AB/replicator-yolo-writer), I didn't have to make one from scratch.

<div align="center">

| Class | Precision | Recall | mAP50 | mAP50-95 |
| :--- | :---: | :---: | :---: | :---: |
| **All** | 0.94 | 0.833 | 0.919 | 0.724 |
| **Potato** | 0.931 | 0.801 | 0.902 | 0.705 |
| **Weed** | 0.948 | 0.865 | 0.935 | 0.742 |

</div>

<figure class="figure-center">
  <img src="/posts/Generating_Synthetic_Datasets_using_Isaac_Sim/inference.png" width="600">
  <figcaption>Figure 9. Inference result(16 images)</figcaption>
</figure>

# 5.Real World Validation
To test how the model might perform in a real field, I conducted a validation using the virtual farm scenarios I built in Isaac Sim. I kept the core layout—the lighting, soil textures, and the placement of crops and weeds—but used Generative AI to transform these scenes into photorealistic images that mimic actual drone footage. This allowed me to analyze the model's inference performance on images that look like the real world.

<figure class="figure-center">
  <img src="/posts/Generating_Synthetic_Datasets_using_Isaac_Sim/real_farm.png" width="700">
  <figcaption>Figure 10. Inference result(16 images)</figcaption>
</figure>

The images above (Figures 10) show the model's inference results. When compared with manual visual inspection, the model successfully distinguishes between potato plants and Bog Marshcress.

However, there are too many bounding boxes for a single object. For example, in the middle of Figure 10, there are at most five potato plants (based on the flowers), but there are 10 boxes shown. I think this issue is caused by the fact that I placed the potatoes much more densely in the virtual environment than they would be in reality.