Project Overview: Sign-Link

Sign-Link is an AI-powered assistive technology designed to bridge the communication gap between the Deaf and hard-of-hearing community and non-signers. By utilizing a standard webcam, the system captures American Sign Language (ASL) gestures and translates them into digital text in real-time.

Unlike older systems that required expensive "data gloves" or depth-sensing cameras (like Xbox Kinect), Sign-Link is software-centric, making it highly accessible and low-cost. It uses a hybrid approach of Computer Vision (OpenCV), Landmark Detection (MediaPipe), and Deep Learning (TensorFlow) to achieve high-speed inference.

Key Features

1. Real-Time Landmark-Based Recognition
Instead of feeding the entire raw image into the neural network, the system uses MediaPipe to identify 21 3D hand landmarks.

Benefit: This filters out "background noise" (like a cluttered room) and focuses the AI only on the shape and orientation of the hand, drastically increasing accuracy and speed.

2. High-Performance CNN Architecture
The core "brain" is a Convolutional Neural Network (CNN) optimized for spatial pattern recognition.

Training: Built on a massive dataset of 29,000 images.

Optimization: Includes Dropout layers to prevent the model from simply "memorizing" the training data, ensuring it works for different users.

3. Low-Latency Inference
With a target inference time of under 100ms, the translation feels instantaneous to the user. This is achieved by using a lightweight model structure that doesn't require a high-end GPU to run smoothly.

4. Robust Preprocessing Pipeline
To handle real-world conditions, the project features a sophisticated pipeline:

Gray-scaling & Normalization: Standardizes input regardless of skin tone or lighting.

Region of Interest (ROI) Extraction: Automatically "crops" the view to the hand’s location.

5. Seamless User Interface
The project integrates a functional GUI (built with Tkinter or Flask) that provides:

A live video feed with a bounding box overlay.

A clear text-display area for the predicted letter/word.

High-visibility labels for better accessibility.
