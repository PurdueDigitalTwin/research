# How to run RL algos:

### Navigate to the root directory

cd ~/Desktop/research

### Add the current directory to PYTHONPATH

export PYTHONPATH=$PYTHONPATH:.

### Run the DQN script with the required work_dir flag and other unrequired flags

python src/projects/rl/run_dqn.py --work_dir=./logs

### To run AC script:

python src/projects/rl/run_ac.py --work_dir=./logs

### To run PPO script:

python src/projects/rl/run_ppo.py --work_dir=./logs
