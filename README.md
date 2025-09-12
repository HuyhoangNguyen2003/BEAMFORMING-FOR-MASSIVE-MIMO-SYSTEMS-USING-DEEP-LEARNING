**Beamforming for Massive MIMO Systems using Deep Learning**

Short Description: Graduation thesis that proposes a deep learning-based solution to optimize analog beamforming in Massive MIMO systems, with the goal of enhancing spectral efficiency for next-generation wireless networks.

**Table of Contents**
1. Introduction

2. Project Structure

3. Technologies Used

4. Installation and Usage

**1. Introduction*
Problem Statement: Massive MIMO systems require powerful signal processors to perform digital beamforming, which leads to high costs and power consumption.

Proposed Solution: This project proposes a deep learning-based solution to optimize analog beamforming, aiming to maximize spectral efficiency and reduce system costs.

**2. Project Structure*
MATLAB Files (.m): These files are used for channel data and sample generation.

channel_gen_LOS.m: Generates channel data with a Line-of-Sight (LoS) path.

gen_samples.m: Generates data samples for the training process.

HybridPrecoding.m: Contains the source code for the hybrid precoding algorithm.

power_allocation.m: Manages power allocation within the system.

Jupyter Notebooks (.ipynb): These notebooks are used for training and testing the deep learning model.

Train_1x64.ipynb: Trains the model with a 1x64 antenna configuration.

Train_2x32_4x64_8x64.ipynb: Trains the model with various antenna configurations.

Python Files (.py):

utils2.py: A file containing utility functions used in the project.

Requirements File (.txt):

requirements.txt: Lists all necessary Python libraries for the project.

**3. Technologies Used*
Languages: Python, MATLAB, C/C++

Libraries & Frameworks: TensorFlow, Keras, Pandas, Matplotlib

Other Tools: GitHub

**4. Installation and Usage*
Prerequisites:

MATLAB

Python 3.x

A list of Python libraries (refer to requirements.txt)

Instructions: Provide a step-by-step guide on how to install and run the project.
