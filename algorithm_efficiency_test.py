import numpy as np
import random
import algo
import time
import matrix_no_print
import os
import algo_no_print as algo_file

class efficiency_class():
    def __init__(self,game):
        self.game = game
    def calculate_100_times(self):
        list_score=[0,0,0]
        total_nodes=0
        time_start = time.time()

        for i in range(0,5):
            print("i:"+str(i))
            self.game.start_one_round()
            can_drop = 0
            win = 0
            while(1):
                algorithm = algo_file.algo_class(self.game)
                action, drop_action, nodes_number = algorithm.AI_chose_node(can_drop)
                end,win = self.game.ai_input_action(action, drop_action)
                if action == "drop":
                    can_drop = 1
                if end == 1:
                    break
            self.game.end_one_round()
            # win
            if win==1:
                list_score[0] = list_score[0]+1
            # lose
            if win==2:
                list_score[1] = list_score[1]+1
            # tie
            if win==0:
                list_score[2] = list_score[2]+1
            total_nodes = total_nodes + nodes_number
        time_end = time.time()
        time_use = time_end-time_start
        print(list_score)
        print(total_nodes)
        print(time_use)
        return list_score,total_nodes,time_use


def __main__():
    all_list_score = [0, 0, 0]
    all_total_nodes = 0
    all_time_use = 0
    for i in range(0,20):  # 5 * 100
        size=1
        print(size)
        game = matrix_no_print.game_class(size)  # size of the cards
        efficiency = efficiency_class(game)
        list_score, total_nodes, time_use=efficiency.calculate_100_times()

        all_list_score[0] += list_score[0]
        all_list_score[1] += list_score[1]
        all_list_score[2] += list_score[2]

        all_total_nodes += total_nodes

        all_time_use += time_use

    print("*********************************************************************************")
    print(all_list_score)
    print(all_total_nodes)
    print(all_time_use)
    print("*********************************************************************************")
if __name__ == "__main__":
    __main__()
