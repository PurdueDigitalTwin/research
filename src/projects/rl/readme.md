# How to run dqn.py:
# Navigate to the root directory (research)
cd ~/Desktop/research

# Add the current directory to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:.

# Run the DQN script with the required work_dir flag
python src/projects/rl/dqn.py --work_dir=./logs --num_episodes=10000 --batch_size=512