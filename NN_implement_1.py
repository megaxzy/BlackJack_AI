
import torch as tr
from torch.nn import Sequential, Conv2d, Linear, Flatten, LeakyReLU, Tanh
import torch.nn.functional
import matrix_modified
import algo_for_nn as algo
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#import scipy

def __main__():
    path = ['imple_1_size_1.pkl', 'imple_1_size_2.pkl', 'imple_1_size_3.pkl', 'imple_1_size_4.pkl','imple_1_size_5.pkl']

    size=2
    myNet = nn.Sequential(
        nn.Linear(39, 20),
        nn.Tanh(),
        nn.Linear(20, 10),
        nn.Tanh(),
        nn.Linear(10, 1),
        nn.Sigmoid()
    )
    optimzer = torch.optim.SGD(myNet.parameters(), lr=0.001)
    loss_func = nn.MSELoss()

    train_loss= []
    output_true,output_pred=[],[]
    it_num = 100
    for i in range(it_num):
        print(i)
        train_data = np.zeros((1, 39))
        target_data = np.zeros((1, 1))
        for k in range(5):
            game = matrix_modified.game_class(size)  # size of the cards
            for j in range(5):
                game.start_one_round()
                #action=""
                #if game.player_sum >= 17: action ="stay"
                #elif game.player_sum <= 11: action = "hit"
                #else: action = "drop"
                algorithm = algo.algo_class(game)
                action, drop_action, nodes_number,win_rates= algorithm.AI_chose_node_2(0)
                target=0
                if(action == "stay"): target=[[0.5]]
                elif(action == "hit"): target=[[1]]
                elif(action == "drop"): target=[[0]]
                output_true.append(target)
                train_item=np.zeros((1,39))
                for x in range(0,game.cards.matrix.shape[0]):
                    for y in range(0,game.cards.matrix.shape[1]):
                        if game.cards.matrix[x][y][0] == 1: train_item[0][0+ y % 13]+=1
                        if game.cards.matrix[x][y][1] == 1: train_item[0][13+y % 13]+=1
                        if game.cards.matrix[x][y][2] == 1: train_item[0][26+y % 13]+=1
                train_data=np.r_[train_data, train_item]
                target_data=np.r_[target_data,target]
                game.end_one_round()

        train_data_torch = torch.tensor(torch.from_numpy(train_data[1:train_data.shape[0]])).float()
        target_data_torch = torch.tensor(torch.from_numpy(target_data[1:target_data.shape[0]])).float()

        for epoch in range(5000):
            out = myNet(train_data_torch)
            loss = loss_func(out, target_data_torch)
            train_loss.append(loss)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
        out=myNet(train_data_torch)
        for i in range(len(out)):
            output_pred.append(float(out[i][0]))

    # loss function figure
    ran = np.arange(0, it_num, 1/5000)
    plt.plot(ran,train_loss)
    plt.show()

    # error figure
    plt.scatter(output_true, output_pred, marker='o', color='red', s=100, label='First')
    plt.show()
    tr.save(myNet,path[size-1])



if __name__ == "__main__":
    __main__()
















