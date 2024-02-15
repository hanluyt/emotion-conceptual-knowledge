A Brain-Inspired Way of Reducing the Network Complexity via Concept-Regularized Coding for Emotion Recognition
==================
Our manuscript has been accepted by AAAI 2024!
--------
This is a PyTorch implementation of the CRPN model proposed in the paper "A Brain-Inspired Way of Reducing the Network Complexity via Concept-Regularized Coding for Emotion Recognition"

Abstract
-------
The human brain can effortlessly and reliably perceive emotions, whereas existing facial emotion recognition (FER) methods suffer from drawbacks such as complex model structures, high storage requirements, and poor interpretability. Inspired by the role of emotion concepts in visual perception coding within the human brain, we propose a dual-pathway framework emulating the neural computation of emotion recognition. Specifically, these two pathways are designed to model the representation of emotion concepts in the brain and the visual perception process, respectively. For the former, we adopt a disentangled approach to extract emotion concepts from complex facial geometric attributes; for the latter, we employ an emotional confidence evaluation strategy to determine which concept is optimal for regularizing the perceptual coding. The proposed concept-regularized coding strategy endows the framework with flexibility and interpretability, as well as good performances on several benchmarking FER datasets.

Framework of the CRPN model
------
![](https://github.com/hanluyt/gACNN_pytorch/blob/main/framework.png)