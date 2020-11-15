import numpy as np
import random
import algo
import time
import matrix_modified
import os
import algo

class efficency_class():
    def __init__(self,game):
        self.game = game
    def calculate_100_times(self):
        for i in range(0,100):
            self.game.start_one_round()
            can_drop = 0
            while(1):
                algorithm = algo.algo_class(self.game)
                action, drop_action, nodes_number = algorithm.AI_chose_node(can_drop)
                end = self.game.ai_input_action(action, drop_action)
                if action == "drop":
                    can_drop = 1
                if end == 1:
                    break



def __main__():
    size=2      # 5 * 100
    game = matrix_modified.game_class(size)  # size of the cards
    efficency = efficency_class(game)

if __name__ == "__main__":
    __main__()
