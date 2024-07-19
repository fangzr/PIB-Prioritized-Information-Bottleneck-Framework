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
```

## Framework

![System model](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Figure/system-model.jpg | width=50%)

Our system includes edge cameras positioned across various scenes, each covering a specific field of view. The combined fields of view ensure comprehensive monitoring of each scene. In high-density pedestrian areas, the goal is to enable collaborative perception for predicting pedestrian occupancy despite limited channel capacity and poor conditions. The framework uses edge servers to receive and process video data from the cameras, which is then analyzed by a cloud server connected via fast wired links. This setup ensures efficient surveillance and real-time analytics, prioritizing essential data for transmission and processing.

## Demo

### Single Camera Perception

The following video demonstrates the perception results from a single camera (the 4th edge camera). Notice the limited perception range and the pedestrians that are not detected (dashed circles).

[![Single Camera Perception](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Demo/single-4.gif)](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/raw/main/Demo/single-4.mp4)

### Collaborative Perception

### Two-camera collaboration
The next video shows the improved perception coverage when the 4th and 7th edge cameras collaborate. While collaboration enhances the coverage, there are still some undetected pedestrians compared to the results from seven edge cameras.

[![Two-Camera Collaborative Perception](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Demo/double.gif)](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Demo/double.mp4)

### Seven-camera collaboration
We utilize all cameras (seven edge cameras) to cooperate with each other and improve perception coverage. Although we see rapid growth in streaming data rates, it is noted that this solution provides the best coverage compared to the combinations mentioned above.

[![Seven-Camera Collaborative Perception](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Demo/7-camera_Compression.gif)](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Demo/7-camera_Compression.mp4)

## Acknowledgement

We gratefully acknowledge the contributions of the following projects:

- [MVDet](https://github.com/hou-yz/MVDet) for their invaluable tools and insights into multi-view detection.
- [TOCOM-TEM](https://github.com/shaojiawei07/TOCOM-TEM) for providing task-oriented communication framework for edge video analytics.