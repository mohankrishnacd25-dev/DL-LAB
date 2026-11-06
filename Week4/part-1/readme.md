# Object Detection with EfficientDet

## Overview & Aim

The goal of this project is to build and run a modern, single-shot object detection pipeline in a Google Colab environment. Instead of manually proposing regions, this script uses a highly optimized deep learning model to simultaneously identify what objects are in an image and exactly where they are located.

This code demonstrates the practical application of pre-trained models for immediate, out-of-the-box computer vision tasks.

## Key Components & Architecture

* **Model (EfficientDet-D0):** The script loads a pre-trained EfficientDet model directly from TensorFlow Hub. EfficientDet is known for its excellent balance of high accuracy and computational efficiency.
* **Target Classes (COCO):** The model is trained on the COCO (Common Objects in Context) dataset. The script includes a mapping for a subset of common objects, allowing it to name things like people, cars, animals, and everyday items.
* **OpenCV (cv2):** Used for all image manipulation tasks—specifically reading the image, drawing the bounding box rectangles, and overlaying the text labels.

## Pipeline & Workflow

1. **Model Loading:** The `detector` is instantiated by downloading the EfficientDet architecture and pre-trained weights from a TensorFlow Hub URL.
2. **Image Upload:** The script uses Google Colab's native file upload widget to let the user select an image from their local machine.
3. **Preprocessing:** 
   * The image is read as a standard NumPy array of BGR pixels.
   * It is converted into a TensorFlow tensor.
   * A batch dimension is added (`input_tensor[tf.newaxis, ...]`) because neural networks expect to process images in batches, even if it's just a batch of one.
4. **Inference:** The tensor is passed through the detector in a single forward pass, returning bounding box coordinates, class IDs, and confidence scores.
5. **Post-processing & Filtering:** 
   * The script loops through the raw detections and ignores any with a confidence score below 50% (`confidence_threshold = 0.5`).
   * It scales the normalized bounding box coordinates (fractions from 0 to 1) back into actual pixel values based on the image's original dimensions.
6. **Visualization:** OpenCV draws green rectangles around the valid detections and writes the class name and score above the box. The final image is displayed using Colab's safe `cv2_imshow` function.

## Expected Outcome

Upon successfully running the script and uploading an image, the expected output is the original image displayed directly within the Colab output cell, annotated with:
* **Green Bounding Boxes:** Drawn accurately around the localized objects.
* **Text Labels & Scores:** Displaying the predicted COCO class name (e.g., `person`, `car`, `dog`) and the confidence score (e.g., `0.85`) located just above each bounding box.

Because EfficientDet is a single-shot detector, this pipeline will run significantly faster than older, region-proposal-based methods (like R-CNN).