# How to run DQN:

### Navigate to the root directory

cd ~/Desktop/research

### Add the current directory to PYTHONPATH

export PYTHONPATH=$PYTHONPATH:.

### Run the DQN script with the required work_dir flag and other unrequired flags

python src/projects/rl/main_dqn.py --work_dir=./logs
