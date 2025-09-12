**Beamforming for Massive MIMO Systems using Deep Learning**

Short Description: Graduation thesis that proposes a deep learning-based solution to optimize analog beamforming in Massive MIMO systems, with the goal of enhancing spectral efficiency for next-generation wireless networks.

**Table of Contents**
1. Introduction

2. Project Structure

3. Technologies Used

4. Installation and Usage

**1. Introduction*
This thesis proposes a Deep Learning-based solution to optimize analog beamforming in Massive MIMO systems. The model learns to directly map from imperfect channel state information (CSI) to optimal beamforming weights, aiming to maximize spectral efficiency for next-generation wireless networks.

**2. Project Structure*
- MATLAB Files (.m): These files are used for channel data and sample generation.

  + channel_gen_LOS.m: Generates channel data with a Line-of-Sight (LoS) path.

  + gen_samples.m: Generates data samples for the training process.

  + HybridPrecoding.m: Contains the source code for the hybrid precoding algorithm.

  + power_allocation.m: Manages power allocation within the system.

- Jupyter Notebooks (.ipynb): These notebooks are used for training and testing the deep learning model.

  + Train_1x64.ipynb: Trains the model with a 1x64 antenna configuration.

  + Train_2x32_4x64_8x64.ipynb: Trains the model with various antenna configurations.

- Python Files (.py):

  + utils2.py: A file containing utility functions used in the project.

- Requirements File (.txt):

  + requirements.txt: Lists all necessary Python libraries for the project.

**3. Technologies Used*
- Languages: Python, MATLAB, C/C++

- Libraries & Frameworks: TensorFlow, Keras, Pandas, Matplotlib

- Other Tools: GitHub

**4. Installation and Usage*
- Prerequisites:

  + MATLAB

  + Python 3.x

  + A list of Python libraries (refer to requirements.txt)
----------------------------------------------------------------
**Installation and Usage**
To run this project, you will need both MATLAB and a Python environment with the specified libraries.

1. Set Up the Environment
- Install Python Libraries: Open your terminal or command prompt and install the necessary Python libraries using the requirements.txt file.

 ```
    pip install -r requirements.txt
 ```

- Ensure MATLAB is Installed: Confirm that you have MATLAB installed on your system, as it's required to generate the project's data.

2. Generate Data
Run MATLAB Scripts: Open MATLAB and run gen_samples.m to generate the channel and data samples:

This process will create the data files needed for the deep learning model.

3. Train and Test the Model
Launch Jupyter: In your terminal, navigate to the project directory and start the Jupyter Notebook server.
```
    jupyter notebook
```
Open and Run Notebooks: In your web browser, open the Jupyter Notebook interface and run the .ipynb files sequentially to train and test the model.
