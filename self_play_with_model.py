'''
Author: Zhongke Sun
self_play_with_model.py - Self-play using trained model
'''

# ---------- Start Single Thread Self-Play ----------

import os
import shutil
import random
from datetime import datetime
from environment.light_env.chessboard import L_Chessboard
from tools.PolicyValueMCTS import PolicyValueMCTS
from model.policy_value_net import ResNetPolicyValueNet
import tensorflow as tf
from environment.lookup_tables import Winner
        
# ---------- Config ----------
NUM_GAMES = 10
NUM_SIMULATIONS = 100
SAVE_DIR = 'self play game results'
MODEL_PATH = 'checkpoints/best_model'

# ---------- Opening Book (Optional Opening Moves) ----------
# OPENING_MOVES = [
#     'H2+3',  # 马八进七 Knight 2 + 3
#     'C2.5',  # 炮二平五 Cannon 2 shift 5
#     'H8+7',  # 马二进三 Knight 8 + 7
#     'C8.5',  # 炮八平五 Cannon 8 shift 5
# ]
OPENING_MOVES = []

# ---------- Clean old self-play Files ----------
if os.path.exists(SAVE_DIR):
    shutil.rmtree(SAVE_DIR)
os.makedirs(SAVE_DIR, exist_ok=True)

# ---------- Load Trained Model ----------
model = ResNetPolicyValueNet()
model.load_weights(MODEL_PATH)
print("Model loaded from:", MODEL_PATH)

# ---------- Self Play Main ----------
def self_play_game_with_model(model, game_id=0, verbose=False):
    env = L_Chessboard()
    env.reset()
    game_data = []

    # Optional: Use one of the opening moves
    if OPENING_MOVES:
        opening_move = random.choice(OPENING_MOVES)
        try:
            uci = env.parse_WXF_move(opening_move)
            fen = env.FENboard()
            env.step(uci)
            game_data.append(f"{fen} -> {uci}")
            if verbose:
                print(f"[Opening] Used move: {opening_move} -> {uci}")
        except Exception as e:
            print(f"[Opening Error] Failed to apply {opening_move}: {e}")
            

    while True:
        fen = env.FENboard()
        if verbose:
            print(f"\nTurn {env.steps}: {'Red' if env.is_red_turn else 'Black'} to move")
            for row in env.board[::-1]:
                print(row)

        mcts = PolicyValueMCTS(env, model, num_simulations=NUM_SIMULATIONS)
        action = mcts.select_action()
        _, reward, done, _ = env.step(action)
        game_data.append(f"{fen} -> {action}")
        
        if done:
            break
        
    if verbose:
        print("\n[Final Board]")
        for row in env.board[::-1]:
            print(row)
    
    
    # ---------- Add game result ----------
    if env.winner == Winner.red:
        print("Winner detected: Red")
        game_data.append("Winner: Red")
    elif env.winner == Winner.black:
        print("Winner detected: Black")
        game_data.append("Winner: Black")
    else:
        print(f"Winner still None or Draw: {env.winner}")
        game_data.append("Winner: Draw")

    # ---------- Apply penalty for draw ----------
    if env.winner == Winner.draw:
        result_score = 0  # Draw
    elif env.winner == Winner.red:
        result_score = 1  # Red wins
    else:
        result_score = -1  # Black wins

    # Optional penalty to draws
    DRAW_PENALTY = 0.5
    if result_score == 0:
        result_score = -DRAW_PENALTY if env.steps % 2 == 0 else DRAW_PENALTY  # Alternate draw punishment

    # ---------- Save Game ----------
    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(SAVE_DIR, f"self_play_game_{game_id}_{timestamp}.txt")
    with open(save_path, 'w') as f:
        f.write("\n".join(game_data + [f"# Result Score: {result_score}"]))
    print(f"[✓] Game {game_id} saved to {save_path}")
    return game_data, result_score



# ---------- Start Self-Play ----------
if __name__ == '__main__':
    for i in range(NUM_GAMES):
        self_play_game_with_model(model, game_id=i, verbose=True)
        






# ---------- Start Multiple Thread Self-Play ----------

# '''
# Author: Zhongke Sun
# self_play_with_model.py - Parallel Self-play using trained model
# '''

# import os
# import shutil
# import random
# from datetime import datetime
# from multiprocessing import Pool, cpu_count
# from environment.light_env.chessboard import L_Chessboard
# from tools.PolicyValueMCTS import PolicyValueMCTS
# from model.policy_value_net import ResNetPolicyValueNet
# import tensorflow as tf

# # ---------- Config ----------
# NUM_GAMES = 10
# NUM_SIMULATIONS = 100
# SAVE_DIR = 'self play game results'
# MODEL_PATH = 'checkpoints/best_model'
# DRAW_PENALTY = 0.5

# # ---------- Opening Book ----------
# OPENING_MOVES = [
#     'H2+3',
#     'C2.5',
#     'H8+7',
#     'C8.5',
# ]

# # ---------- Clean old self-play Files ----------
# if os.path.exists(SAVE_DIR):
#     shutil.rmtree(SAVE_DIR)
# os.makedirs(SAVE_DIR, exist_ok=True)

# # ---------- Load model per process ----------
# def load_model():
#     model = ResNetPolicyValueNet()
#     model.load_weights(MODEL_PATH)
#     return model

# # ---------- Self-play for one game ----------
# def self_play_game(game_id):
#     model = load_model()
#     env = L_Chessboard()
#     game_data = []

#     # Opening move
#     if OPENING_MOVES:
#         opening_move = random.choice(OPENING_MOVES)
#         try:
#             uci = env.parse_WXF_move(opening_move)
#             fen = env.FENboard()
#             env.step(uci)
#             game_data.append(f"{fen} -> {uci}")
#         except Exception as e:
#             print(f"[Opening Error] {opening_move}: {e}")

#     # Game loop
#     while not env.is_end():
#         fen = env.FENboard()
#         mcts = PolicyValueMCTS(env, model, num_simulations=NUM_SIMULATIONS)
#         action = mcts.select_action()
#         env.step(action)
#         game_data.append(f"{fen} -> {action}")

#     # Game result
#     if env.winner == 1:
#         result_score = 1
#         game_data.append("Winner: Red")
#     elif env.winner == -1:
#         result_score = -1
#         game_data.append("Winner: Black")
#     else:
#         result_score = -DRAW_PENALTY if env.steps % 2 == 0 else DRAW_PENALTY
#         game_data.append("Winner: Draw")

#     # Save result
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     filename = os.path.join(SAVE_DIR, f"self_play_game_{game_id}_{timestamp}.txt")
#     with open(filename, 'w') as f:
#         f.write("\n".join(game_data + [f"# Result Score: {result_score}"]))
#     print(f"[✓] Game {game_id} saved to {filename}")
#     return filename

# # ---------- Run all games in parallel ----------
# if __name__ == '__main__':
#     print(f"Generating {NUM_GAMES} games using {min(cpu_count(), NUM_GAMES)} processes...")
#     with Pool(processes=min(cpu_count(), NUM_GAMES)) as pool:
#         pool.map(self_play_game, range(NUM_GAMES))