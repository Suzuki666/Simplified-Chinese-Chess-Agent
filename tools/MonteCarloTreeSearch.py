'''
Author: Zhongke Sun
00:14 26 March 2025
'''

import math
import random
import copy
from collections import defaultdict
from environment.lookup_tables import Winner

class MCTSNode:
    def __init__(self, env, parent=None, action=None):
        self.env = copy.deepcopy(env)
        self.parent = parent
        self.action = action
        self.children = {}
        self._untried_actions = env.legal_moves()
        self.visits = 0
        self.value = 0.0

    def is_fully_expanded(self):
        return len(self._untried_actions) == 0

    def best_child(self, c_param=1.4):
        choices = [
            (child.value / child.visits + c_param * math.sqrt(math.log(self.visits) / child.visits), child)
            for child in self.children.values()
        ]
        return max(choices, key=lambda x: x[0])[1]

    def expand(self):
        action = self._untried_actions.pop()
        next_env = copy.deepcopy(self.env)
        next_env.step(action)
        child_node = MCTSNode(next_env, parent=self, action=action)
        self.children[action] = child_node
        return child_node

    def backpropagate(self, result):
        self.visits += 1
        self.value += result
        if self.parent:
            self.parent.backpropagate(-result)  # Attention: The next step is the counterpart, flip win/loss

    def is_terminal_node(self):
        return self.env.done

    def rollout(self):
        rollout_env = copy.deepcopy(self.env)
        while not rollout_env.is_end():
            actions = rollout_env.legal_moves()
            if not actions:
                break
            action = random.choice(actions)
            rollout_env.step(action)

        if rollout_env.winner == Winner.red:
            return 1
        elif rollout_env.winner == Winner.black:
            return -1
        else:
            return 0  # Tie

class MCTS:
    def __init__(self, env, num_simulations=100):
        self.root = MCTSNode(env)
        self.num_simulations = num_simulations

    def select_action(self):
        for _ in range(self.num_simulations):
            node = self.root
            while not node.is_terminal_node() and node.is_fully_expanded():
                node = node.best_child()

            if not node.is_terminal_node():
                node = node.expand()

            result = node.rollout()
            node.backpropagate(result)

        return self.root.best_child(c_param=0).action  # select best without exploration