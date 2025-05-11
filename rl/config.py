# config.py
# Hyperparameters for TD3 and training


SEED = 0                   # Random seed for reproducibility

# Network architecture
HIDDEN_SIZES = [256, 256]  # Hidden layer sizes for Actor and Critic

# Replay buffer
BUFFER_SIZE = int(1e6)
BATCH_SIZE = 128

# Discount factor and soft update rate
GAMMA = 0.99
TAU = 0.005

# TD3-specific noise parameters
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_DELAY = 2

# Learning rate for both Actor and Critic optimizers
LR = 1e-4

# Training parameters
MAX_EPISODES = 1000         # Total training episodes
MAX_STEPS = 500           # Maximum steps per episode
EXPL_NOISE = 0.1          # Std of Gaussian exploration noise added to actions
SAVE_INTERVAL = 50        # Episodes between saving model checkpoints
ECAL_FREQ = 50