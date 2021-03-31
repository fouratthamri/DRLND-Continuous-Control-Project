# DRLND-Continuous-Control-Project
The github submission of the third project of Udacity's nanodegree program entitled "Continuous Control"

## Requirements

To set up your python environment to run the code in this repository, follow the instructions below.

1. Create (and activate) a new environment with Python 3.6.

    - __Linux__ or __Mac__:

    ```bash
    conda create --name drlnd python=3.6
    source activate drlnd
    ```

    - __Windows__:

    ```bash
    conda create --name drlnd python=3.6 
    activate drlnd
    ```

2. Clone the repository (if you haven't already!), and navigate to the `python/` folder.  Then, install several dependencies.

    ```bash
    cd python
    pip install .
    ```

3. On Anaconda

    ```bash
    conda install pytorch=1.7.0 -c pytorch
    ```

4. Then create an [IPython kernel](http://ipython.readthedocs.io/en/stable/install/kernel_install.html) for the `drlnd` environment.

    ```bash
    python -m ipykernel install --user --name drlnd --display-name "drlnd"
    ```

5. Change the kernel to match the `drlnd` environment by using the dropdown kernel menu

6. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
S
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

7. Place the file in the repository, in the `Reacher/` folder, and unzip (or decompress) the file.

## Training the model

To train the model, run all the cells in the file `Continuous_Control.ipynb`. The code will automatically save model checkpoints when training finishes by achieveing an average score of 30 over the last 100 episodes.

Use of GPU is recommended to make the training faster.

## Deep Deterministic Gradient Policy (DDPG)

![Image 1](doublejointcontrol.gif?style=centerme)

The goal of this project is to train a double-jointed robot arm to move to target locations in a 3D unity environment.

### States

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm.

### Actions

 Each action is a vector with four real numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Reward

 A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The task is continuous, and in order to solve the environment, the agent must get an average score of +30 over 100 consecutive episodes.

### Report

A full description of the Deep RL model used can be found in the file report.pdf.
