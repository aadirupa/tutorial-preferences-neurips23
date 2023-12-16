# Preference-based Reinforcement Learning demo

Credits: APReL project and codebase at https://github.com/Stanford-ILIAD/APReL 

## Instructions for running the demo

1. pip3 install -r requirements.txt (this also installs the Stable Baselines package for RL policy training).

2. Install the APReL package by following the install instructions at https://github.com/Stanford-ILIAD/APReL

3. Run gather_prefs.py to interactively gather human preferences for 10 trajectory pairs from the Mountain-car Gym environment.

4. Note down the 3-element weight vector learned from the previous step, and encode it in the 'weights' parameter of trajectory_env in the file train_policy.py

5. run train_policy.py to train an RL agent using the learnt reward function and the PPO algorithm. 

6. run test_policy.py to render the learnt policy. 
