# AI-Based Mini Application: On-Device Face Detection with TensorFlow Lite

## Overview
This Android mini-application performs real-time on-device face detection using **TensorFlow Lite (TFLite)** and Google's pretrained **MediaPipe BlazeFace (`blaze_face_short_range.tflite`)** model. Inspired by the reference repository `Uzair390-Del/NumberGuessingAppTflite`, it demonstrates loading a `.tflite` model from the `assets` directory, preprocessing image tensors, executing inference via the `Interpreter` API, and rendering bounding boxes on Android.

## Prerequisites
- **Android Studio**: Hedgehog (2023.1.1) or newer
- **JDK**: Java 8 / 17
- **Android SDK**: API Level 24 (Android 7.0) or higher

## Setup & Model Placement (IMPORTANT)
1. Download the pretrained **`blaze_face_short_range.tflite`** model from Hugging Face (`litert-community/MediaPipe-FaceDetector-BlazeFace-Short-Range`) or Google MediaPipe Models.
2. Place the downloaded `blaze_face_short_range.tflite` file directly inside:
   `app/src/main/assets/blaze_face_short_range.tflite`
3. Open this project folder in **Android Studio**, allow Gradle to sync dependencies (`tensorflow-lite`, `tensorflow-lite-support`), and click **Run** on an emulator or physical Android device.

## Dependencies Used
- `org.tensorflow:tensorflow-lite:2.16.1`
- `org.tensorflow:tensorflow-lite-support:0.4.4`
- `org.tensorflow:tensorflow-lite-task-vision:0.4.4`
