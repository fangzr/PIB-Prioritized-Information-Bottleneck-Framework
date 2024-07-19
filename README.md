# PIB: Prioritized Information Bottleneck Framework for Collaborative Edge Video Analytics

## Abstract

Collaborative edge sensing systems, particularly in collaborative perception systems in autonomous driving, can significantly enhance tracking accuracy and reduce blind spots with multi-view sensing capabilities. However, their limited channel capacity and the redundancy in sensory data pose significant challenges, affecting the performance of collaborative inference tasks. To tackle these issues, we introduce a Prioritized Information Bottleneck (PIB) framework for collaborative edge video analytics. We first propose a priority-based inference mechanism that jointly considers the signal-to-noise ratio (SNR) and the camera's coverage area of the region of interest (RoI). To enable efficient inference, PIB reduces video redundancy in both spatial and temporal domains and transmits only the essential information for the downstream inference tasks. This eliminates the need to reconstruct videos on the edge server while maintaining low latency. Specifically, it derives compact, task-relevant features by employing the deterministic information bottleneck (IB) method, which strikes a balance between feature informativeness and communication costs. Given the computational challenges caused by IB-based objectives with high-dimensional data, we resort to variational approximations for feasible optimization. Compared to five coding methods for image and video compression, PIB improves mean object detection accuracy (MODA) by 17.8\% and reduces communication costs by 49.61\% in poor channel conditions.

## Requirements

To replicate the environment and dependencies used in this project, you will need the following packages:

```plaintext
ffmpeg-python==0.2.0
kornia==0.6.1
matplotlib==3.5.3
numpy==1.21.5
pillow==9.4.0
python==3.7.12
pytorch==1.10.0
torchaudio==0.10.0
torchvision==0.11.0
tqdm==4.66.4
