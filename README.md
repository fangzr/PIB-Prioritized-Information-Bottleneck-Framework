# PIB: Prioritized Information Bottleneck Theoretic Framework with Distributed Online Learning for Edge Video Analytics (IEEE ToN'25)

This is the open-source repository of the paper published in IEEE ToN [(Paper PDF)](https://www.researchgate.net/publication/387662571_Prioritized_Information_Bottleneck_Theoretic_Framework_with_Distributed_Online_Learning_for_Edge_Video_Analytics).

## Abstract

Collaborative perception systems leverage multiple edge devices, such as surveillance cameras or autonomous cars, to enhance sensing quality and eliminate blind spots. Despite their advantages, challenges such as limited channel capacity and data redundancy impede their effectiveness. To address these issues, we introduce the Prioritized Information Bottleneck (PIB) framework for edge video analytics. This framework prioritizes the shared data based on the signal-to-noise ratio (SNR) and camera coverage of the region of interest (RoI), reducing spatial-temporal data redundancy to transmit only essential information. This strategy avoids the need for video reconstruction at edge servers and maintains low latency. It leverages a deterministic information bottleneck method to extract compact, relevant features, balancing informativeness and communication costs. For high-dimensional data, we apply variational approximations for practical optimization. To reduce communication costs in fluctuating connections, we propose a gate mechanism based on distributed online learning (DOL) to filter out less informative messages and efficiently select edge servers. Moreover, we establish the asymptotic optimality of DOL by proving the sublinearity of its regrets. To validate the effectiveness of the PIB framework, we conduct real-world experiments on three types of edge devices with varied computing capabilities. Compared to five coding methods for image and video compression, PIB improves mean object detection accuracy (MODA) by 17.8\% while reducing communication costs by 82.65\% under poor channel conditions.

## Requirements

To replicate the environment and dependencies used in this project, you will need the following packages:

```plaintext
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

<img src="https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Figure/system-model.jpg" alt="System model" width="80%">

**Figure 1: System model.**

Our system includes edge cameras positioned across various scenes, each covering a specific field of view. The combined fields of view ensure comprehensive monitoring of each scene. In high-density pedestrian areas, the goal is to enable collaborative perception for predicting pedestrian occupancy despite limited channel capacity and poor conditions. The framework uses edge servers to receive and process video data from the cameras, which is then analyzed by a cloud server connected via fast wired links. This setup ensures efficient surveillance and real-time analytics, prioritizing essential data for transmission and processing.

## Experimental Results

### Dataset
Our experiments employ the [Wildtrack dataset](https://www.epfl.ch/labs/cvlab/data/data-wildtrack/) from EPFL. This dataset features high-resolution images captured by seven cameras positioned in a public area, recording unscripted pedestrian movements \[[Chavdarova et al., 2018](https://arxiv.org/abs/1705.03847)\].

### Experimental Parameters
We conduct simulations using the following settings:
- **Operating Frequency**: 2.4 GHz
- **Path Loss Exponent**: 3.5
- **Shadowing Deviation**: 8 dB
- **Interference Power**: Devices emit an interference power of 0.1 Watts.
- **Device Density**: 10 to 100 devices per 100 square meters, testing various congestion levels.
- **Bandwidth**: 2 MHz
- **Camera Placement**: Cameras are located approximately 200 meters from the edge server.


### Baselines
To evaluate the performance of our PIB framework, we compare it against five baselines, including both video coding and image coding approaches:

1. **[TOCOM-TEM](https://github.com/shaojiawei07/TOCOM-TEM)**  
   A task-oriented communication framework that utilizes a temporal entropy model for edge video analytics. It applies the deterministic Information Bottleneck (IB) principle to extract and transmit compact, task-relevant features, integrating spatial-temporal data on the server for enhanced inference accuracy.

2. **[JPEG](https://dl.acm.org/doi/abs/10.1145/103085.103089)**  
   A widely used image compression standard that employs lossy compression algorithms to reduce image data size. JPEG is commonly used to decrease communication loads in networked camera systems.

3. **[H.265 (HEVC)](https://ieeexplore.ieee.org/abstract/document/7100895/?casa_token=1h0USFniCPEAAAAA:xmqN8ev626HwHPwczUkAK5Sw--C02k0E5RVVJ9ayDNgicli412wTNmdKDLIK-WU-DrWidFmDkQ)**  
   Also known as High Efficiency Video Coding, H.265 offers up to 50% better data compression than its predecessor H.264, while maintaining the same video quality. It is crucial for efficient data transmission in high-density camera networks.

4. **[H.264 (AVC)](https://ieeexplore.ieee.org/abstract/document/1218189/?casa_token=MZil42Kz95IAAAAA:ndWTce90S_raoq3D-qrvHNwP5zpjqV7vy4YEibLV1m93H0uRQBPvUanm2GvBTSQkeEmRX5LRQQ)**  
   Known as Advanced Video Coding, H.264 significantly enhances video compression efficiency, allowing high-quality video transmission at lower bit rates.

5. **[AV1](https://ieeexplore.ieee.org/abstract/document/9363937/)**  
   AOMedia Video 1 (AV1) is an open, royalty-free video coding format developed by the Alliance for Open Media (AOMedia). It outperforms existing codecs like H.264 and H.265, making it ideal for online video applications with improved compression efficiency.

### Impact of Communication Bottlenecks and Delayed Cameras on Perception Accuracy

As shown in Figure 2, we demonstrate how communication bottlenecks and delayed cameras affect perception accuracy:

<img src="https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Figure/performance1.png" alt="Impact of communication bottlenecks and delayed cameras on perception accuracy." width="80%">

**Figure 2: Impact of communication bottlenecks and delayed cameras on perception accuracy.**

### Communication Bottleneck vs Latency

Figure 3 illustrates the trade-off between communication bottlenecks and latency in our system:

<img src="https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Figure/latency1.png" alt="Communication bottleneck vs latency." width="80%">

**Figure 3: Communication bottleneck vs latency.**

### Hardware Platform Configuration

As shown in Figure 4, our experimental setup features a practical hardware testbed that includes three distinct edge devices: NVIDIA Jetson™ Orin Nano™ 4GB, NVIDIA Jetson™ Orin NX™ 16GB, and ThinkStation™ P360. The edge devices collaboratively interact with edge servers equipped with RTX 5000 Ada GPUs for efficient video decoding.

<img src="https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Figure/hardware.png" alt="Hardware setup" width="80%">

**Figure 4: Edge device configuration.**

### Jetson™ Orin device Configuration

The Jetson™ Orin NX™ 16GB/ Jetson™ Orin Nano™ devices are configured with a PyTorch deep learning environment. The configuration for Jetson NX differs from x86 architectures, and setting up the environment requires following the official NVIDIA installation guide for PyTorch on the Jetson platform. For detailed instructions, you can refer to the official [PyTorch installation guide for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#install-multiple-versions-pytorch) or this helpful [tutorial](https://www.cnblogs.com/guohaomeng/p/18347870).

### Encoder Latency Across Different Platforms

The encoding latency results of our PIB in different edge devices are presented in **Table 1**. It can be observed that the feature map generation phase dominates the overall encoding latency, while the entropy coding phase contributes a negligible amount of time. Furthermore, edge devices with higher computing capacity exhibit significantly lower encoding latency.

**Table 1: Encoder Latency Across Different Platforms**

| **Phase**                   | **Nano (ms)**    | **Orin NX (ms)**  | **P360 (ms)**    |
|-----------------------------|------------------|-------------------|------------------|
| Feature map generation      | 755.32±69.32     | 227.54±2.65       | 37.49±0.90       |
| Entropy coding              | 10.83±3.51       | 1.79±0.75         | 0.40±0.11        |
| **Total encoder latency**   | **766.15±70.55** | **229.34±2.67**   | **37.80±0.94**   |


## Usage

### Environment Setup

1. Create and activate the Conda environment:
```bash
conda create -n PIB_env python=3.7.12
conda activate PIB_env
```

2. Install the required packages:
```bash
pip install kornia==0.6.1 matplotlib==3.5.3 numpy==1.21.5 pillow==9.4.0
pip install torch==1.10.0 torchaudio==0.10.0 torchvision==0.11.0 tqdm==4.66.4
```

### Training Pipeline

The training process consists of two main stages: feature extraction and coding/inference.

#### Stage 1: Feature Extraction

Run feature extraction using `main_feature_extraction.py`. The script supports various parameters:

```bash
python main_feature_extraction.py \
    --dataset_path "/path/to/your/dataset" \
    --epochs 30 \
    --beta 1e-5 \
    --target_rate 80 \
    --delays "X1 X2 X3 X4 X5 X6 X7"  # Xi represents frame delay for i-th camera, calculated based on channel conditions
```

Key parameters:
- `--dataset_path`: Path to your dataset directory
- `--epochs`: Number of training epochs (default: 30)
- `--beta`: Information bottleneck trade-off parameter (default: 1e-5)
- `--target_rate`: The constraint on the communication cost (KB)
- `--delays`: Frame delays for each camera (space-separated values). Each value X represents the number of frames delayed for that camera, calculated based on network conditions in utils/channel.py

#### Stage 2: Coding and Inference

After feature extraction, run the coding and inference stage using `main_coding_and_inference.py`:

```bash
python main_coding_and_inference.py \
    --dataset_path "/path/to/your/dataset" \
    --model_path "/path/to/trained/model/MultiviewDetector.pth" \
    --epochs 10 \
    --delays "X1 X2 X3 X4 X5 X6 X7"  # Xi represents frame delay for i-th camera, calculated based on channel conditions
```

Key parameters:
- `--dataset_path`: Path to your dataset directory
- `--model_path`: Path to the trained model from Stage 1
- `--epochs`: Number of inference epochs (default: 10)
- `--delays`: Frame delays for each camera (space-separated values). Each value X represents the number of frames delayed for that camera, calculated based on network conditions in utils/channel.py

### Example Training Workflow

1. First, run feature extraction:
```bash
CUDA_VISIBLE_DEVICES=0,1 python main_feature_extraction.py \
    --dataset_path "/data/Wildtrack" \
    --epochs 30 \
    --beta 1e-5 \
    --target_rate 80
```

2. Then, run coding and inference using the trained model:
```bash
CUDA_VISIBLE_DEVICES=0,1 python main_coding_and_inference.py \
    --dataset_path "/data/Wildtrack" \
    --model_path "logs_feature_extraction/YYYY-MM-DD_HH-MM-SS/MultiviewDetector.pth" \
    --epochs 10
```

Note: Replace the model path with your actual trained model path, which will be in the logs directory with a timestamp.

## Demo

### Single Camera Perception

The following video demonstrates the perception results from a single camera (the 4th edge camera). Notice the limited perception range and the pedestrians that are not detected (dashed circles).

<!-- [![Single Camera Perception](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Demo/trimmed_gifs/single-4.gif)](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/raw/main/Demo/single-4.mp4) -->

https://raw.githubusercontent.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/refs/heads/main/Demo/single-4.mp4

### Collaborative Perception

#### Two-camera collaboration
The next video shows the improved perception coverage when the 4th and 7th edge cameras collaborate. While collaboration enhances the coverage, there are still some undetected pedestrians compared to the results from seven edge cameras.

https://raw.githubusercontent.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/refs/heads/main/Demo/double.mp4

<!-- [![Two-Camera Collaborative Perception](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Demo/trimmed_gifs/double.gif)](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Demo/double.mp4) -->

#### Seven-camera collaboration
We utilize all cameras (seven edge cameras) to cooperate with each other and improve perception coverage. Although we see rapid growth in streaming data rates, it is noted that this solution provides the best coverage compared to the combinations mentioned above.

<!-- [![Seven-Camera Collaborative Perception](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Demo/trimmed_gifs/7-camera_Compression.gif)](https://github.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/blob/main/Demo/7-camera_Compression.mp4) -->

https://raw.githubusercontent.com/fangzr/PIB-Prioritized-Information-Bottleneck-Framework/refs/heads/main/Demo/7-camera_Compression.mp4

## Citations

If you find this code useful for your research, please cite our papers:

```bibtex
@article{fang2025ton,
  title={Prioritized Information Bottleneck Theoretic Framework with Distributed Online Learning for Edge Video Analytics},
  author={Fang, Z. and Hu, S. and Wang, J. and Deng, Y. and Chen, X. and Fang, Y.},
  journal={IEEE/ACM Transactions on Networking},
  year={Jan. 2025},
  note={DOI: 10.1109/TON.2025.3526148},
  publisher={IEEE}
}

@inproceedings{fang2024pib,
  author = {Z. Fang and S. Hu and L. Yang and Y. Deng and X. Chen and Y. Fang},
  title = {{PIB: P}rioritized Information Bottleneck Framework for Collaborative Edge Video Analytics},
  booktitle = {IEEE Global Communications Conference (GLOBECOM)},
  year = {Dec. 2024},
  pages = {1--6},
  address = {Cape Town, South Africa}
}
```

## Acknowledgement

We gratefully acknowledge the contributions of the following projects:

- [MVDet](https://github.com/hou-yz/MVDet) for their invaluable tools and insights into multi-view detection.
- [TOCOM-TEM](https://github.com/shaojiawei07/TOCOM-TEM) for providing task-oriented communication framework for edge video analytics.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
