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

        for i in range(0,10):
            print("i:"+str(i))
            self.game.start_one_round()
            can_drop = 0
            win = 0
            while(1):
                ran_01 = random.randint(0, 2)
                end = 0
                if ran_01==0:
                    print("Chose stay!!!")
                    end, win = self.game.ai_input_action("stay",0)
                    break
                if ran_01==1:
                    print("Chose hit!!!")
                    end,win = self.game.ai_input_action("hit", 0)
                if ran_01 == 2 and can_drop == 1:
                    continue
                if ran_01==2 and can_drop==0:
                    print("Chose drop!!!")
                    can_drop = 1
                    ran_02 = random.randint(0, len(self.game.list_player_cards)-1)
                    end,win = self.game.ai_input_action("drop", ran_02)

                if end == 1:
                    break
            self.game.end_one_round()
            print(win)
            # win
            if win==1:
                list_score[0] = list_score[0]+1
            # lose
            if win==2:
                list_score[1] = list_score[1]+1
            # tie
            if win==0:
                list_score[2] = list_score[2]+1
            total_nodes = total_nodes
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
    for i in range(0,10):  # 5 * 100
        size=2
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
