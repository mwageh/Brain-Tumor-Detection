# Brain-Tumor-Detection

## Overview

This project involves detecting brain tumors using MRI images. Follow the steps below to process the data and perform the classification.

## Steps

1. **Download Datasets**
   - **Dataset 1:** [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)
   - **Dataset 2:** [Brain Tumor Detection](https://www.kaggle.com/datasets/ahmedhamada0/brain-tumor-detection)

2. **Pre-processing**
   - Execute the `Pre-processing.py` script to pre-process the MRI images.

3. **Feature Extraction**
   - Run the scripts for the pre-trained models to extract deep features from the pre-processed images.

4. **Feature Concatenation**
   - Execute the `concatenation.py` script to concatenate the extracted deep features.

5. **Feature Selection**
   - Apply genetic selection techniques to choose the most informative features.

6. **Classification**
   - Perform the classification process using the selected features.
