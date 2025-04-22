'''
Author: Zhongke Sun
evaluate_model.py - Evaluate trained policy-value model with visualization
'''
# evaluate_vs_random.py

from environment.light_env.chessboard import L_Chessboard, RED, BLACK, Winner
from tools.PolicyValueMCTS import PolicyValueMCTS
from model.policy_value_net import ResNetPolicyValueNet
import random

NUM_GAMES = 5
MODEL_PATH = "checkpoints/best_model"

def random_move(env):
    return random.choice(env.legal_moves())

def evaluate_model_vs_random(model_path, num_games=20):
    model = ResNetPolicyValueNet()
    model.load_weights(model_path)
    print(f"Loaded model from: {model_path}")

    win, lose, draw = 0, 0, 0

    for i in range(num_games):
        env = L_Chessboard()
        env.reset()
        mcts = PolicyValueMCTS(env, model)

        while True:
            if env.turn == RED:
                action = mcts.select_action()
            else:
                action = random_move(env)

            _, _, done, _ = env.step(action)

            if done:
                break

        if env.winner == Winner.red:
            win += 1
        elif env.winner == Winner.black:
            lose += 1
        else:
            draw += 1

        print(f"[Game {i}] Winner: {env.winner.name}")

    print("\nEvaluation Results vs Random Player:")
    print(f"Model Wins:   {win}")
    print(f"Model Loses:  {lose}")
    print(f"Draws:         {draw}")
    print(f"Win Rate:     {win / num_games * 100:.2f}%")

if __name__ == "__main__":
    evaluate_model_vs_random(MODEL_PATH, NUM_GAMES)