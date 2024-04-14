# Atta Leafcutter-Ant Image Classification

## Overview
This project aims to classify Atta leafcutter-ant images from iNaturalist using machine learning. Our goal is to accurately categorize these ants by gender, status, and role, contributing significantly to ecological studies.

## Supervision and Collaboration
- **Supervisor:** Dr. Fatma Tarlaci

## Objectives
- Develop a machine learning model for the classification of Atta leafcutter-ants.
- Classify ant images into categories: Gender (Male/Female), Status (Alive/Dead), Role (Alate/Dealate).

## Methodology
- Employ and fine-tune a pre-trained model like YOLOv8.
- Document model selection, training processes, and optimization techniques comprehensively.

## Environment Setup
To set up the project environment:
1. Create the Conda environment from the provided `environment.yml` file:
   ```bash
   conda env create -f environment.yml
   '''
2. Activate the Conda environment:
   ```bash
   conda activate atta-detection
   '''
3. Upgrade PyTorch and torchvision to the latest versions:
   ```bash
   pip install --upgrade torch torchvision
   '''

4. Install the latest version of Roboflow:
   ```bash
   pip install --upgrade roboflow
   '''

## Tools and Technologies
- Machine Learning Model: YOLOv8
- Gradio Interface: Gradio is utilized to create a user-friendly web interface for the YOLOv8 model. This setup enables image uploads and real-time adjustments of detection settings such as confidence and IoU thresholds, thereby facilitating easy demonstrations and experiments.

## Acknowledgements
- Dr. Fatma Tarlaci for her supervision and expertise in machine learning.
- Dr. Ulrich Mueller for initiating this collaborative project and providing the dataset.