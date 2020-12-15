

import numpy as np
import torch as tr
from torch.nn import Sequential, Conv2d, Linear, Flatten, LeakyReLU, Tanh
import torch.nn.functional
import matrix_modified
import algo_for_nn as algo
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import time as time

def judge_chose(result,can_drop):
    actions=["stay","hit","drop"]
    chose_action = ""
    if can_drop==0:
        if (result[0] >= result[1] and result[0] >= result[2]): chose_action = actions[0]
        elif (result[1] >= result[0] and result[1] >= result[2]): chose_action = actions[1]
        elif (result[2] >= result[0] and result[2] >= result[1]): chose_action = actions[2]
    else:
        if (result[0] >= result[1]): chose_action = actions[0]
        elif (result[1] >= result[0]): chose_action = actions[1]
    return chose_action

def best_drop(game):
    list = game.list_player_cards
    sum = game.player_sum
    min_val=11
    chose = 0
    for i in range(len(list)):
        if abs(int(sum)-int(list[i][1])-11)<min_val: chose= i
    return chose
def data_processing(game):
    matrix_desk = np.zeros((1,10))
    matrix_player = np.zeros((1, 10))
    matrix_dealer = np.zeros((1, 10))

    for i in range(game.cards.matrix.shape[0]):
        for j in range(game.cards.matrix.shape[1]):
            if game.cards.matrix[i][j][0] == 1:
                number = j%13
                if number>=9: matrix_desk[0,9]+=1
                else: matrix_desk[0,number]+=1
            if game.cards.matrix[i][j][1] == 1:
                number = j % 13
                if number>=9: matrix_player[0,9]+=1
                else: matrix_player[0,number]+=1
            if game.cards.matrix[i][j][2] == 1:
                number = j % 13
                if number>=9: matrix_dealer[0,9]+=1
                else: matrix_dealer[0,number]+=1
    #print(matrix_desk)
    #print(matrix_player)
    #print(matrix_dealer)

    matrix_ans = np.r_[matrix_desk,matrix_player]
    matrix_ans = np.r_[matrix_ans,matrix_dealer]
    matrix_ans=matrix_ans.reshape((1,30))
    #print(matrix_ans.shape)
    return matrix_ans


path = ['imple_3_size_1.pkl','imple_3_size_2.pkl','imple_3_size_3.pkl','imple_3_size_4.pkl','imple_3_size_5.pkl']
class efficiency_class():
    def __init__(self,game,load_net,size):
        self.game = game
        self.load_net=load_net
        self.size=size
    def calculate(self):
        list_score=[0,0,0]
        for i in range(0,5):
            self.game.start_one_round()
            can_drop = 0
            drop_action = 0

            while(1):
                test_item = data_processing(self.game)
                test_item = torch.tensor(torch.from_numpy(test_item)).float()
                result = self.load_net(test_item)
                action = judge_chose(result[0], can_drop)
                if action == "drop":
                    can_drop = 1
                    drop_action = best_drop(self.game)
                end, win = self.game.ai_input_action(action, drop_action)
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
        return list_score


def __main__():
    all_list_score = [0, 0, 0]
    size = 5
    load_net = tr.load(path[size - 1])
    for i in range(0,20):  # 5 * 100
        game = matrix_modified.game_class(size)  # size of the cards
        efficiency = efficiency_class(game,load_net,size)
        list_score= efficiency.calculate()

        all_list_score[0] += list_score[0]
        all_list_score[1] += list_score[1]
        all_list_score[2] += list_score[2]
        print(all_list_score)

    print("*********************************************************************************")
    print("size:",size)
    print(all_list_score)
    print("*********************************************************************************")
if __name__ == "__main__":
    __main__()
