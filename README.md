# TDprop

```We based our code off of [ikostrikov's pytorch-rl repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr). ```

## Installation

### PyTorch

```conda install pytorch torchvision cudatoolkit=10.1 -c pytorch``` 

### Gym
```pip install gym[atari] ```

### Baselines for Atari preprocessing
``` 
git clone https://github.com/openai/baselines.git 
cd baselines 
pip install -e . 
```

### Backpack for TDprop
``` pip install backpack-for-pytorch  ``` 

## Replicating results

To replicate our atari experiments run the slurm array found in sarsa/array_slurm.sh
