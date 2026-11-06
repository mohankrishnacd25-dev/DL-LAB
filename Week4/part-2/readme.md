# Object Detection with Selective Search and VGG16

## Overview & Aim

The goal of this project is to construct a custom, two-stage object detection pipeline in a Google Colab environment. Instead of using a single-shot detector, this script mimics the classic R-CNN (Region-based Convolutional Neural Network) architecture. 

It works by first generating potential object locations ("region proposals") and then passing those cropped regions to a powerful image classifier to determine what, if anything, is inside them.

## Key Components & Architecture

* **Region Proposals (OpenCV Selective Search):** The script utilizes OpenCV's `ximgproc` module to run Selective Search (Fast mode). This algorithm groups adjacent pixels by color, texture, and intensity to guess where objects might be. If Selective Search fails, it includes a fallback sliding-window mechanism.
* **Classifier (VGG16):** It uses the classic VGG16 deep learning model, pre-trained on the ImageNet dataset (capable of recognizing 1000 different object categories). 
* **Non-Maximum Suppression (NMS):** Because Selective Search often generates multiple overlapping boxes for the same object, a custom NMS function is implemented. It filters out redundant boxes based on an Intersection over Union (IoU) threshold of 0.3, keeping only the bounding box with the highest confidence score.

## Pipeline & Workflow

1. **Environment Setup:** The script automatically detects if OpenCV's extended modules (`opencv-contrib-python`) are installed. If not, it uninstalls standard OpenCV, installs the contrib version, and restarts the Colab runtime to enable Selective Search.
2. **Image Upload & Downscaling:** The user uploads an image, which is then scaled down to a width of 600 pixels to speed up processing while maintaining the aspect ratio.
3. **Region Extraction:** `get_region_proposals` identifies up to 200 bounding boxes that likely contain objects.
4. **Cropping & Batch Prediction:** 
   * Each proposed bounding box is cropped from the original image and resized to 224x224 (the required input size for VGG16).
   * These crops are stacked into a batch and passed to VGG16 for classification in a single, efficient step.
5. **Filtering & NMS:** Predictions are decoded. Only classifications with a confidence probability higher than 90% are kept. The custom NMS function then clears out overlapping duplicates.
6. **Visualization:** The final bounding boxes, along with their ImageNet labels and confidence scores, are drawn on the image using OpenCV and displayed via Colab's `cv2_imshow`.

## Expected Outcome

Upon successfully running the script and uploading an image, the expected output is the original image displayed directly within the Colab output cell, annotated with:
*   **Green Bounding Boxes:** Drawn accurately around the localized objects.
*   **Text Labels & Scores:** Displaying the predicted ImageNet class name (e.g., `golden_retriever`, `sports_car`) and the confidence score (e.g., `0.94`) located just above each bounding box.

*Note:* Because this script uses a two-stage approach (running a full CNN prediction on up to 200 individual region crops), it will be noticeably slower than modern single-shot detectors (like YOLO or EfficientDet). It serves as an excellent educational demonstration of how early region-based object detection systems (like the original R-CNN) operated.