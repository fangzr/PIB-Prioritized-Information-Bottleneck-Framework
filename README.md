# R-ACP: Real-Time Adaptive Collaborative Perception Leveraging Robust Task-Oriented Communications

**Code Coming Soon!**

---

## Research Problem

In collaborative perception systems, multiple unmanned ground vehicles (UGVs) equipped with cameras work together to enhance sensing capabilities and perception accuracy. However, dynamic environments pose significant challenges:

- **Extrinsic Calibration Errors**: Sudden movements, accidents, or terrain changes frequently disrupt camera calibration parameters, causing perception inaccuracies.
- **Timeliness and Data Freshness**: Ensuring real-time updates and fresh data streams is essential, especially in scenarios requiring timely reactions (e.g., pedestrian tracking, emergency alerts).
- **Communication Constraints**: Bandwidth limitations and unreliable communication channels further complicate real-time collaborative perception.

<img src="https://raw.githubusercontent.com/fangzr/R-ACP/refs/heads/main/Challenges.png" alt="Effect of unpredictable accidents involving UGVs on camera extrinsic parameters and perception error rates." width="80%">

Figure 1 illustrates how unpredictable accidents involving UGVs impact extrinsic calibration parameters, significantly increasing perception errors.

---

## Motivation

Accurate extrinsic calibration between cameras on multiple UGVs is crucial for collaborative perception tasks. Traditional calibration methods fail to accommodate rapid changes and non-linear disturbances introduced by dynamic and unpredictable real-world conditions. This motivates a robust, adaptive, and efficient approach to collaborative perception that addresses calibration issues, optimizes communication, and ensures timely, accurate data sharing among UGVs.

---

## Contributions

Our work proposes **Real-time Adaptive Collaborative Perception (R-ACP)**, a robust framework designed specifically for dynamic, communication-constrained environments. Key contributions include:

- **Robust Task-Oriented Communication Strategy**: Optimizes both calibration and feature transmission to maintain perception accuracy while minimizing communication overhead.
- **Age of Perceived Targets (AoPT)**: Formulates a novel metric and corresponding optimization problem for maintaining the freshness and quality of perception data.
- **Channel-aware Self-Calibration via Re-ID**: Introduces adaptive compression of key features, leveraging spatial and temporal cross-camera correlations to significantly improve calibration accuracy.
- **Information Bottleneck (IB)-based Encoding**: Dynamically adjusts video compression rates according to task relevance, achieving an optimal balance between bandwidth consumption and inference accuracy.

Extensive experiments demonstrate substantial improvements in multiple object detection accuracy (MODA) by up to **25.49%** and communication cost reduction by up to **51.36%** under severe channel conditions.

---

## System Model

<img src="https://raw.githubusercontent.com/fangzr/R-ACP/refs/heads/main/system_model.png" alt="Collaborative perception system model." width="80%">

Figure 2 illustrates our collaborative perception system, which consists of multiple UGVs equipped with edge cameras. These UGVs collaboratively track pedestrians, transmitting decoded feature streams to an edge server through wireless channels. The edge server processes these features to:

1. Generate a comprehensive pedestrian occupancy map.
2. Perform pedestrian re-identification (Re-ID).
3. Facilitate real-time calibration without precise initial parameters or additional sensors.

The proposed framework effectively manages calibration issues, data freshness, and unreliable communication, making it highly applicable to diverse real-world scenarios, including smart cities, emergency response, and autonomous navigation.

---

## Reference

- [Full Paper (PDF)](https://arxiv.org/abs/2410.04168)

### BibTeX Citation

```bibtex
@article{fang2024r,
  title={R-ACP: Real-Time Adaptive Collaborative Perception Leveraging Robust Task-Oriented Communications},
  author={Fang, Zhengru and Wang, Jingjing and Ma, Yanan and Tao, Yihang and Deng, Yiqin and Chen, Xianhao and Fang, Yuguang},
  journal={arXiv preprint arXiv:2410.04168},
  year={2024}
}
```
