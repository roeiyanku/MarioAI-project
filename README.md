# CIS530_MarioML
This project is to teach an agent to play a level of Super Mario Bros for the NES with a reinforcement learning algorithm based on https://github.com/Sourish07/Super-Mario-Bros-RL . Using that algorithm to train the agent, The agent will be observed at different checkpoints in its learning process and will be evaluated on how effective it is.

Video Explanation of the Reinforcement learning algorithm:https://youtu.be/_gmQZToTMac?si=4miiqBSBWpwLfcOk

Resource Papers:
    Human-level control through deep reinforcement learning: https://doi.org/10.1038/nature14236
    Deep Reinforcement Learning with Double Q-learning: https://arxiv.org/pdf/1509.06461.pdf

## Set up
### Requirements
There are a few prerequistes to running the RL cycle and running pretrained models
- Anaconda software is installed and added to your terminal path
- Your machine can run Nvidia CUDA software (this is mandatory for running the pretrained models)

### Installation
#### Clone this repository into your working directory
    git clone https://github.com/Ia-in03/CIS530_MarioML.git
#### Run conda env from enviornment.yml
    conda env create -f environment.yml
#### And then activate environment
    conda activate smbrl

## Run python scripts
- **main.py** will run the normal script
- **csv_plotter** will run the same as main but all episode information (epsilon value, total=reward, etc.) will be dumped to a csv file
- **eval-models.py** will run through the existing pretrained models and print data (max reward, avg reward) to console
#### Each script has constant values that can be adjusted based on if you want to see a display or not, want to train the agent or not, number of episode cycles, etc.
