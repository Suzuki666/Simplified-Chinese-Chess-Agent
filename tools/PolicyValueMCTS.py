'''
Author: Zhongke Sun
Updated: 27 March 2025
MCTS with Policy-Value Network
'''

import copy
import numpy as np
import math
import tensorflow as tf
from collections import defaultdict

from environment.lookup_tables import Winner, ActionLabelsRed, flip_policy
from environment.light_env.chessboard import L_Chessboard
from model.policy_value_net import ResNetPolicyValueNet


class PV_MCTSNode:
    def __init__(self, env, parent=None, action=None, prior=1.0):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior = prior
        self.history = defaultdict(int)  # Used to record repeated situation counting number

    def is_fully_expanded(self):
        return len(self.children) == len(self.env.legal_moves())

    def expand(self, policy_probs):
        for action in self.env.legal_moves():
            if action not in self.children:
                next_env = copy.deepcopy(self.env)
                next_env.step(action)
                next_fen = next_env.FENboard()
                self.children[action] = PV_MCTSNode(
                    env=next_env,
                    parent=self,
                    action=action,
                    prior=policy_probs.get(action, 1e-8)
                )
                self.children[action].history = self.history.copy()
                self.children[action].history[next_fen] += 1

    def best_child(self, c_puct=1.4):
        total_visits = sum(child.visits for child in self.children.values()) + 1e-8
        scores = {}
        for action, child in self.children.items():
            q_value = 0 if child.visits == 0 else child.value_sum / child.visits
            u_value = c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visits)
            scores[action] = q_value + u_value
        return max(scores.items(), key=lambda x: x[1])[0]

    def backup(self, value):
        self.visits += 1
        self.value_sum += value
        if self.parent:
            self.parent.backup(-value)


class PolicyValueMCTS:
    def __init__(self, env, model, num_simulations=100, c_puct=1.4):
        self.root = PV_MCTSNode(env)
        self.model = model
        self.num_simulations = num_simulations
        self.c_puct = c_puct

    def evaluate(self, node):
        fen = node.env.FENboard()
        board_tensor = self.fen_to_tensor(fen)[np.newaxis, ...]
        policy_logits, value = self.model(board_tensor)
        value = float(value[0][0])
        policy_probs = tf.nn.softmax(policy_logits).numpy()[0]

        is_red = node.env.is_red_turn
        labels = ActionLabelsRed if is_red else flip_policy(ActionLabelsRed)
        move_probs = {labels[i]: policy_probs[i] for i in range(len(labels))}
        return move_probs, value

    def fen_to_tensor(self, fen):
        board = L_Chessboard()
        board.assign_fen(fen)
        tensor = np.zeros((15, 10, 9), dtype=np.float32)
        from environment.lookup_tables import piece_to_index
        for y in range(10):
            for x in range(9):
                ch = board.board[y][x]
                if ch != '.':
                    idx = piece_to_index.get(ch, -1)
                    if idx != -1:
                        tensor[idx, y, x] = 1.0
        tensor[14, :, :] = 1.0 if board.is_red_turn else 0.0
        return tensor

    def select_action(self):
        for sim in range(self.num_simulations):
            node = self.root
            path = []

            while node.children:
                action = node.best_child(self.c_puct)
                node = node.children[action]
                path.append(node)

            if not node.env.is_end():
                policy_probs, value = self.evaluate(node)
                node.expand(policy_probs)

                # Test if repetitive situation over 10
                cur_fen = node.env.FENboard()
                if node.history[cur_fen] >= 5:
                    value = 0  # Draw
                elif self._is_repetitive_action(path):
                    value -= 0.5  # Penaltize repetitive actions
            else:
                # End state
                if node.env.winner == Winner.red:
                    value = 1
                elif node.env.winner == Winner.black:
                    value = -1
                else:
                    value = 0

            node.backup(value)

        # Add exploration mechanism at the root node
        visit_counts = np.array([child.visits for child in self.root.children.values()])
        actions = list(self.root.children.keys())
        probs = visit_counts ** (1 / self.c_puct)
        probs = probs / np.sum(probs)

        # Use weighted sampling instead of greedy choosing
        chosen_action = np.random.choice(actions, p=probs)
        return chosen_action

    def _is_repetitive_action(self, path):
        """Check repetitive action in the path"""
        if len(path) < 4:
            return False
        actions = [node.action for node in path if node.action is not None]
        if len(actions) < 4:
            return False
        return actions[-1] == actions[-3] and actions[-2] == actions[-4]