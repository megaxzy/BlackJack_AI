import numpy as np
import torch as tr
from torch.nn import Sequential, Conv2d, Linear, Flatten, LeakyReLU, Tanh
import torch.nn.functional
import matrix_modified
import algo_for_nn as algo
import torch
import torch.nn as nn
import matplotlib.pyplot as plt




def train_target(result):
    if (result[0] >= result[1] and result[0] >= result[2]): return float(result[0])
    elif (result[1] >= result[0] and result[1] >= result[2]): return float(1+result[1])
    elif (result[2] >= result[0] and result[2] >= result[1]): return float(2+result[2])


def __main__():
    size=1
    path = ['imple_4_size_1.pkl', 'imple_4_size_2.pkl', 'imple_4_size_3.pkl', 'imple_4_size_4.pkl','imple_4_size_5.pkl']

    myNet = nn.Sequential(
        nn.Linear(size*52*5, 300),
        nn.Tanh(),
        nn.Linear(300, 9),
        nn.Tanh(),
        nn.Linear(9, 3),
        nn.Sigmoid()
    )
    print(myNet)

    # set optimzer
    optimzer = torch.optim.SGD(myNet.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    print("training initial *****************************************************")
    it_number=100
    it_epo_num = 5000
    train_loss= []
    output_true,output_pred=[],[]
    for i in range(it_number):
        # initial game
        target_num = 1
        sum_hit,sum_stay, sum_drop= 0,0,0
        train_data = np.zeros((1, size * 52 * 5))
        target_data = np.zeros((1, 3))
        while(1):
            if sum_hit==target_num and sum_stay==target_num and sum_drop==target_num:
                break
            game = matrix_modified.game_class(size)  # size of the cards
            for j in range(25):
                game.start_one_round()
                algorithm = algo.algo_class(game)
                action, drop_action, nodes_number = algorithm.AI_chose_node(0)

                if action=="hit":
                    if sum_hit==target_num: continue
                    else: sum_hit+=1
                if action=="stay":
                    if sum_stay==target_num: continue
                    else: sum_stay+=1
                if action=="drop":
                    if sum_drop==target_num: continue
                    else: sum_drop+=1
                print("Chose Action:",action)

                targets=[]
                if(action == "stay"):
                    targets=[[1,0,0]]
                    output_true.append(1)
                if(action == "hit"):
                    targets=[[0,1,0]]
                    output_true.append(2)
                if(action == "drop"):
                    targets=[[0,0,1]]
                    output_true.append(3)

                train_item=game.cards.matrix.reshape((1,size * 52 * 5))
                train_data=np.r_[train_data, train_item]
                target_data=np.r_[target_data,targets]
                game.end_one_round()

        train_data_torch = torch.tensor(torch.from_numpy(train_data[1:train_data.shape[0]])).float()
        target_data_torch = torch.tensor(torch.from_numpy(target_data[1:target_data.shape[0]])).float()

        for epoch in range(it_epo_num):
            out = myNet(train_data_torch)
            loss = loss_func(out, target_data_torch)  #loss
            train_loss.append(loss)
            optimzer.zero_grad()  #gra
            loss.backward()
            optimzer.step()
        out=myNet(train_data_torch)
        print(out)
        for i in range(len(out)):
            output_pred.append(train_target(out[i]))
    # loss function figure
    x = np.arange(0, it_number, 1/it_epo_num)
    plt.plot(x,train_loss)
    plt.show()
    # item figure
    #print(output_true)
    #print(output_pred)
    plt.scatter(output_true,output_pred,marker = 'x',color = 'red', s = 50 ,label = 'First')
    plt.show()
    # save the file
    print("save")
    tr.save(myNet,path[size-1])


if __name__ == "__main__":
    __main__()