'''
Author: Zhongke Sun
Updated: 27 March 2025
train.py - Training policy-value network
'''

import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.models import load_model

from model.policy_value_net import ResNetPolicyValueNet
from tools.self_play_data_loader import load_all_games, create_training_dataset

# ----------- Hyperparameters ------------
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.001
CHECKPOINT_DIR = 'checkpoints'
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ----------- Load & Prepare Data ----------
print("Loading self-play data...")
data = load_all_games('self play game results')
x_train, policy_targets, value_targets = create_training_dataset(data)

print(f"x_train shape: {x_train.shape}")
print(f"policy_targets shape: {policy_targets.shape}")
print(f"value_targets shape: {value_targets.shape}")

# ----------- Build or Load Model ----------
model_path = os.path.join(CHECKPOINT_DIR, 'best_model.h5')
if os.path.exists(model_path):
    print("Loading existing model...")
    model = load_model(model_path)
else:
    print("Building new model...")
    model = ResNetPolicyValueNet()

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss=[CategoricalCrossentropy(), MeanSquaredError()],
    metrics=[CategoricalAccuracy(), 'mse']
)

# ----------- Train ------------------------
checkpoint_cb = ModelCheckpoint(
    filepath=os.path.join(CHECKPOINT_DIR, 'best_model'),
    monitor='val_loss',
    save_best_only=True,
    save_format='tf',  # SavedModel
    verbose=1
)

print("Start training...")
model.fit(
    x=x_train,
    y=[policy_targets, value_targets],
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[checkpoint_cb],
    shuffle=True
)

print("Training completed. Best model saved to:", model_path)