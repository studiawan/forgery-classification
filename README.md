# forgery-classification
Lightweight deep learning model for detecting real vs. fake images under multiple JPEG compression scenarios.

This repository contains the implementation of a lightweight image forgery classification model based on a modified ShuffleNet V2 architecture. The model is designed to classify images as real or fake under various JPEG compression scenarios, including single compression, double compression, and Facebook recompression to simulate real-world social-media degradation.

To improve feature extraction and computational efficiency, the base ShuffleNet V2 network is improved with:
- Ghost Module: generates additional feature maps through inexpensive linear operations.
- Squeeze-and-Excitation (SE) Block: adaptively recalibrates channel-wise feature responses.
- FReLU Activation Function: improves representation of spatial features.

Key features:
- Multi-scenario training: 1×, 2×, and Facebook compression
- Lightweight and fast inference suitable for forensic image analysis
- Implemented in PyTorch
