# init_model.py
from model.policy_value_net import ResNetPolicyValueNet
import os

CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Initialize model and store empty weight
model = ResNetPolicyValueNet()
model.save_weights(MODEL_PATH)

print(f"Initial model is stored at: {MODEL_PATH}")