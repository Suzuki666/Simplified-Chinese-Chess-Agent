'''
Author: Zhongke Sun
train_loop.py - Continuous self-play & training loop
'''

import os
import time
from tools.self_play_data_loader import load_all_games, create_training_dataset
from model.policy_value_net import ResNetPolicyValueNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy, MeanSquaredError
from tensorflow.keras.metrics import CategoricalAccuracy
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import save_model
import numpy as np
import subprocess

# ---------- Configurations ----------
NUM_ITERATIONS = 5
NUM_SELFPLAY_GAMES = 15
SELF_PLAY_SCRIPT = "self_play_with_model.py"
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model")
LEARNING_RATE = 0.001
EPOCHS = 10
BATCH_SIZE = 64

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ---------- Train Loop ------------
from tqdm import tqdm
for iter in tqdm(range(1, NUM_ITERATIONS + 1), desc="Training Loop", unit="iter"):
    print(f"\n==== Iteration {iter} ====")

    # Step 1: Self-play with current model
    print(f"Step 1: Generating self-play games using model...")
    subprocess.run(["python", SELF_PLAY_SCRIPT])

    # Step 2: Load self-play dataset
    print(f"\nStep 2: Loading training data...")
    data = load_all_games("self play game results")
    x_train, policy_targets, value_targets = create_training_dataset(data)
    print(f"Loaded {len(x_train)} samples.")

    # Step 3: Build or load model
    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        model = ResNetPolicyValueNet()
        model.load_weights(MODEL_PATH)
    else:
        print("Creating new model...")
        model = ResNetPolicyValueNet()

    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss=[CategoricalCrossentropy(), MeanSquaredError()],
        metrics=[CategoricalAccuracy(), 'mse']
    )

    # Step 4: Train model
    print(f"\nStep 3: Training model on self-play data...")
    checkpoint_cb = ModelCheckpoint(
        MODEL_PATH, save_weights_only=True, save_best_only=True, monitor="val_loss", verbose=1
    )
    model.fit(
        x_train, [policy_targets, value_targets],
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.1,
        callbacks=[checkpoint_cb],
        shuffle=True
    )

    print(f"\nIteration {iter} completed. Model saved.\n")
    time.sleep(1)