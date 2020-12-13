import numpy as np
import random
import matrix_modified
import time
from scipy import special
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import scipy
import numpy
import random
import matrix_modified
import time
from scipy import special
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import algo_for_nn as algo

import numpy as np
import torch as tr
from torch.nn import Sequential, Conv2d, Linear, Flatten, LeakyReLU, Tanh
import torch.nn.functional
import matrix_modified
import algo_for_nn as algo
import torch
import torch.nn as nn
import numpy as np


def NN(board_size):
    class Nets(tr.nn.Module):
        # torch modified
        def __init__(self, board_size):
            super(Nets, self).__init__()
            self.flatten = Flatten(start_dim=1)
            self.linear = Linear(board_size*5*52, 3,bias=True)
        # torch forward value
        def forward(self, input_value):
            return self.linear(self.flatten(input_value))
    return Nets(board_size)


def calculate_loss(net, x, y_targ):
    return net(x), tr.sum((net(x) - y_targ) * (net(x) - y_targ))


def optimization_step(optimizer, net, x, y_targ):
    optimizer.zero_grad()
    y, e = calculate_loss(net, x, y_targ)
    e.backward()
    optimizer.step()
    return y, e
def judge_right(right_action,result):
    actions=["stay","hit","drop"]
    chose_action = ""
    if (result[0] >= result[1] and result[0] >= result[2]): chose_action = actions[0]
    elif (result[1] >= result[0] and result[1] >= result[2]): chose_action = actions[1]
    elif (result[2] >= result[0] and result[2] >= result[1]): chose_action = actions[2]
    print("chose_action:",chose_action)
    print("right_action:",right_action)
    if chose_action==right_action:
        print("good job !!!!!")
        return 1
    print("bad !!!!!")
    return 0

def __main__():

    # set
    '''
    size = 1
    net = NN(board_size=size)
    print(net)
    '''


    size=1
    myNet = nn.Sequential(
        nn.Linear(size*52*5, 300),
        nn.Sigmoid(),
        nn.Linear(300, 3),
        nn.Sigmoid()
    )
    print(myNet)

    # set optimzer
    optimzer = torch.optim.SGD(myNet.parameters(), lr=0.05)
    loss_func = nn.MSELoss()

    print("training initial *****************************************************")
    train_data=np.zeros((1,size*52*5))
    target_data = np.zeros((1,3))
    for i in range(100):
        # initial game
        game = matrix_modified.game_class(size)  # size of the cards
        for j in range(5):
            game.start_one_round()
            algorithm = algo.algo_class(game)
            action, drop_action, nodes_number = algorithm.AI_chose_node(0)
            print("Chose Action:",action)

            targets=[]
            if(action == "stay"):targets=[[1,0,0]]
            if(action == "hit"):targets=[[0,1,0]]
            if(action == "drop"):targets=[[0,0,1]]

            train_item=game.cards.matrix.reshape((1,size * 52 * 5))
            #print(train_item)
            train_data=np.r_[train_data, train_item]
            #print(train_data)
            target_data=np.r_[target_data,targets]
            game.end_one_round()

    train_data_torch = torch.tensor(torch.from_numpy(train_data)).float()
    target_data_torch = torch.tensor(torch.from_numpy(target_data)).float()

    for epoch in range(1000):
        out = myNet(train_data_torch)
        loss = loss_func(out, target_data_torch)  #loss
        optimzer.zero_grad()  #gra
        loss.backward()
        optimzer.step()

    print("test initial *****************************************************")
    sum_right_count=0
    for i in range(100):
        # initial game
        game = matrix_modified.game_class(size)  # size of the cards
        for j in range(5):
            game.start_one_round()
            algorithm = algo.algo_class(game)
            action, drop_action, nodes_number = algorithm.AI_chose_node(0)
            test_item = game.cards.matrix.reshape((1, size * 52 * 5))
            test_item = torch.tensor(torch.from_numpy(test_item)).float()
            result = myNet(test_item)
            print(result[0])
            sum_right_count += judge_right(action, result[0])

            game.end_one_round()
    print("sum_right_count:", sum_right_count)


if __name__ == "__main__":
    __main__()






'''
test_np=np.random.normal(0.0, pow(3,-0.1), (3,11))
print(test_np)
input_list=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,3,18,18,18,126,136,175,26,166,255,247,127,0,0,0,0,0,0,0,0,0,0,0,0,30,36,94,154,170,253,253,253,253,253,225,172,253,242,195,64,0,0,0,0,0,0,0,0,0,0,0,49,238,253,253,253,253,253,253,253,253,251,93,82,82,56,39,0,0,0,0,0,0,0,0,0,0,0,0,18,219,253,253,253,253,253,198,182,247,241,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,80,156,107,253,253,205,11,0,43,154,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,14,1,154,253,90,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,139,253,190,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,11,190,253,70,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,35,241,225,160,108,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,81,240,253,253,119,25,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,45,186,253,253,150,27,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,16,93,252,253,187,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,249,253,249,64,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,46,130,183,253,253,207,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,39,148,229,253,253,253,250,182,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,24,114,221,253,253,253,253,201,78,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,23,66,213,253,253,253,253,198,81,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,18,171,219,253,253,253,253,195,80,9,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,55,172,226,253,253,253,253,244,133,11,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,136,253,253,253,212,135,132,16,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
inputs = np.array(input_list, ndmin=2).T
print(inputs.shape)
'''
'''
x = np.zeros((2,2))
x[0,0]=15
x[1,1]=-5659
z=expit(x)
print(x)
print(z)
print(expit([-np.inf, -1.5, 0, 1.5, np.inf]))
x = np.linspace(-6, 6, 121)
y = expit(x)
plt.plot(x, y)
plt.grid()
plt.xlim(-6, 6)
plt.xlabel('x')
plt.title('expit(x)')
plt.show()
'''