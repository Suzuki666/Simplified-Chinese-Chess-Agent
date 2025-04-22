'''
Author: Zhongke Sun
Updated: 00:15 26 March, 2025
Mycode/self_play_mcts.py
'''

import os
import random
import sys
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from environment.light_env.chessboard import L_Chessboard
from tools.MonteCarloTreeSearch import MCTS

def self_play_single_game(game_id = 0, num_simulations=50, verbose=True):
    env = L_Chessboard()
    data = []
    move_count = 0

    while not env.is_end():
        if verbose:
            print(f"\nTurn {move_count + 1}: {'Red' if env.is_red_turn else 'Black'} to move")
            env.print_to_cl()

        # MCTS selects action
        mcts = MCTS(env, num_simulations=num_simulations)
        action = mcts.select_action()

        # Save the state and selected move
        board_state = env.FENboard()
        data.append((board_state, action))

        # Execute the move
        env.move_action_str(action)
        move_count += 1

    winner = env.winner.name if env.winner else "Unknown"
    print(f"\nGame over in {move_count} moves. Winner: {winner}")
    return data, winner

#-------------------- One-time Self Play-----------------
# def save_game_data(game_data, winner, filename="self_play_game.txt"):
#     with open(filename, 'w') as f:
#         for board, action in game_data:
#             f.write(f"{board} -> {action}\n")
#         f.write(f"Result: {winner}\n")
#     print(f"Game data saved to {filename}")

# if __name__ == "__main__":
#     data, winner = self_play_single_game(num_simulations=30, verbose=True)
#     save_game_data(data, winner)


# ---------------------  Multiple Self-Play  --------------------
def save_game_data(game_data, winner, game_id):
    os.makedirs("self play game results", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"self play game results/self_play_game_{game_id}_{timestamp}.txt"
    with open(filename, 'w') as f:
        for board, action in game_data:
            f.write(f"{board} -> {action}\n")
        f.write(f"Result: {winner}\n")
    print(f"[Game {game_id}] Saved to {filename}")
    
def self_play_multiple_games(num_games=10, num_simulations=50, verbose=False):
    for i in range(1, num_games + 1):
        data, winner = self_play_single_game(game_id=i, num_simulations=num_simulations, verbose=verbose)
        save_game_data(data, winner, game_id=i)

if __name__ == "__main__":
    # Run multiple self-play
    self_play_multiple_games(num_games=10, num_simulations=30, verbose=True)