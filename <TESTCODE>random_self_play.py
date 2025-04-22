import random
import time

from environment.env import CChessEnv

def play_random_game(max_steps=200):
    env = CChessEnv()
    env.reset()

    print("=== Initialize Chessboard ===")
    env.render()

    for step in range(max_steps):
        if env.done:
            print(f"\nGame over! Winner: {'Red' if env.red_won else 'Black'}")
            break

        legal_moves = env.board.legal_moves()
        if not legal_moves:
            print("No legal move, tie or terminate")
            break

        move = random.choice(legal_moves)
        print(f"\n[Step {step+1}] {'Red' if env.red_to_move else 'Black'} move {move}")
        env.step(move)
        env.render()
        time.sleep(0.3)

    else:
        print("\nExceed maximum steps, tie or terminate")

if __name__ == '__main__':
    play_random_game()