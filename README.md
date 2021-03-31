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