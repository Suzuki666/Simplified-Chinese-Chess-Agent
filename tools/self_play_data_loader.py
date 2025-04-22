'''
Author: Zhongke Sun
Updated: 26 March 2025
Self-Play Dataset Loader + Encoding
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from environment.lookup_tables import piece_to_index, ActionLabelsRed, ActionLabelsBlack, flip_policy
from environment.light_env.chessboard import L_Chessboard

def parse_self_play_file(filepath):
    data = []
    with open(filepath, 'r') as f:
        lines = f.readlines()
        result_line = lines[-1]
        result = 0
        if "red" in result_line.lower():
            result = 1
        elif "black" in result_line.lower():
            result = -1

        for line in lines[:-1]:
            if '->' not in line:
                continue
            fen, move = line.strip().split(' -> ')
            data.append((fen, move, result))
    return data

def load_all_games(games_dir='self play game results'):
    dataset = []
    for fname in os.listdir(games_dir):
        if fname.endswith('.txt'):
            fpath = os.path.join(games_dir, fname)
            dataset.extend(parse_self_play_file(fpath))
    print(f"Loaded {len(dataset)} samples from {games_dir}")
    return dataset

def move_to_index(fen, move):
    board = L_Chessboard()
    board.assign_fen(fen)
    is_red = board.is_red_turn

    labels = ActionLabelsRed if is_red else ActionLabelsBlack
    label_dict = {m: i for i, m in enumerate(labels)}

    return label_dict.get(move, -1)

def fen_to_tensor(fen):
    '''
    Transform FEN to (15, 10, 9) tensor:
    - First 14 channels: each piece corresponds one channel (red 7 + black 7)
    - The 15 th channel: Red->1, Black->0
    '''
    board = L_Chessboard()
    board.assign_fen(fen)
    tensor = np.zeros((15, 10, 9), dtype=np.float32)

    for y in range(10):
        for x in range(9):
            ch = board.board[y][x]
            if ch == '.':
                continue
            idx = piece_to_index.get(ch, -1)
            if idx != -1:
                tensor[idx, y, x] = 1.0

    tensor[14, :, :] = 1.0 if board.is_red_turn else 0.0
    return tensor

def create_training_dataset(data):
    x_list, policy_list, value_list = [], [], []

    for fen, move, result in data:
        board_tensor = fen_to_tensor(fen)
        action_index = move_to_index(fen, move)
        if action_index == -1:
            continue  # Skip unknown move

        x_list.append(board_tensor)
        policy = np.zeros(2086)
        policy[action_index] = 1
        policy_list.append(policy)
        value_list.append(result)

    return np.array(x_list), np.array(policy_list), np.array(value_list)