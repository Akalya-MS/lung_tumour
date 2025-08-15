# Lung Tumour Detection in CT Scans

The goal of this project is to detect and localize lung tumour regions in CT scan images. The approach combines segmentation and object detection to achieve accurate identification of tumour regions for medical image analysis.

## Dataset and Preprocessing

* CT scan images collected from publicly available medical imaging datasets, each image having a corresponding mask that highlights the lung tumour regions.
* Resizing - Resizes both images and masks to 256×256 pixels for consistent input size.
* Images normalized to ensure pixel values are on a consistent scale.
* Data augmentation applied to increase robustness, including:
  * Rotation
  * Horizontal and vertical flipping

## Approach

A hybrid deep learning pipeline is implemented to combine the strengths of two models:

1. **UNet** – Used for semantic segmentation to accurately extract the lung regions and possible tumour areas.
2. **Faster R-CNN** – Applied for object detection to localize and draw bounding boxes around the tumour regions.

The pipeline processes images sequentially:

* First, the UNet model segments the lung and tumour regions.
* Then, the Faster R-CNN model detects and localizes the tumours within the segmented regions.

## Pipelining Concept

The models are integrated into a pipeline where the output of the segmentation stage is passed as input to the detection stage. This reduces false positives by focusing the detection network on relevant areas only.

