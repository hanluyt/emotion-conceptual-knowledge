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
![](https://github.com/hanluyt/emotion-conceptual-knowledge/blob/main/Figure1.png)

Requirements
--------
See requirements.txt for required python libraries.
```
pip install requirements.txt
```

We conduct all experiments with the PyTorch toolbox and four NVIDIA GeForce RTX 3090 GPUs.

Usage
-----
**1. Download Pre-trained models**

We use the ResNet-50 pre-trained on VGGface2 as the backbone for the emotion and non-emotion encoder.
```
wget https://drive.google.com/file/d/1i_aYzKsvnnPcwm3kV0_Y8F9ofoVxSm5x/view?usp=drive_link
```
For the percept encoder, we use ResNet-18 pre-trained on Ms-Celeb-1M as the backbone.
```
wget https://drive.google.com/file/d/10NVjrvhacFlHcdW88Kn_PspYUDjXV4xQ/view?usp=drive_link
```

**2.Train the conceptual pathway**

If you have only one gpu, you can use the following command:
```
python main_concept.py
```
Otherwise, you can use (take 4 gpus as an example)
```
OMP_NUM_THREADS=1 nohup torchrun --nproc_per_node=4 main_concept.py >gpu_more.log 2>&1 &
```

**3. The perceptual coding is regularized by emotion concepts.**

* Use the trained conceptual pathway to arrange the confidence scores of samples for each emotion label in the training set in ascending order.
* Select the 20th percentile of confidence scores as the threshold for each label.
*  If confidence score is less than the threshold, we randomly select and aggregate 8 emotion features from the high-score pool that belong to the same label as the emotion feature and then use them to regularize the perceptual feature.

See ```dataset_percept.py``` and ```high_confidence.py``` for more detail.

**4. Train the concept-regularized perceptual network (CRPN)**

After the conceptual pathway is well trained, the parameters of it are frozen to provide guidance for the perceptual pathway.
```
OMP_NUM_THREADS=1 nohup torchrun --nproc_per_node=4 main_percept.py >gpu_more_percept.log 2>&1 &
```

