
import torch as tr
import torch.nn.functional
import matrix_modified
import algo_for_nn as algo
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


def __main__():
    size=5
    path = ['imple_2_size_1.pkl', 'imple_2_size_2.pkl', 'imple_2_size_3.pkl', 'imple_2_size_4.pkl','imple_2_size_5.pkl']

    myNet = nn.Sequential(
        nn.Linear(size*52*5, 30),
        nn.Softmax(),
        nn.Linear(30, 3),
        nn.Sigmoid()
    )
    opti_function = torch.optim.SGD(myNet.parameters(), lr=0.01)
    loss_func = nn.MSELoss()

    train_loss= []
    output_true,output_pred=[],[]
    for iteration in range(100):
        train_data = np.zeros((1, size * 52 * 5))
        target_data = np.zeros((1, 3))
        for i in range(5):
            game = matrix_modified.game_class(size)  # size of the cards
            for j in range(5):
                game.start_one_round()
                algorithm = algo.algo_class(game)
                action, drop_action, nodes_number,win_rates= algorithm.AI_chose_node_2(0)
                targets=[win_rates]
                for item in win_rates:
                    output_true.append(item)
                train_item=game.cards.matrix.reshape((1,size * 52 * 5))
                train_data=np.r_[train_data, train_item]
                target_data=np.r_[target_data,targets]
                game.end_one_round()

        train_data_torch = torch.tensor(torch.from_numpy(train_data[1:train_data.shape[0]])).float()
        target_data_torch = torch.tensor(torch.from_numpy(target_data[1:target_data.shape[0]])).float()

        for epoch in range(2000):
            out = myNet(train_data_torch)
            loss = loss_func(out, target_data_torch)  #loss
            train_loss.append(loss)
            opti_function.zero_grad()  #gra
            loss.backward()
            opti_function.step()
        out=myNet(train_data_torch)
        print(out)
        for i in range(len(out)):
            for j in range(3):
                output_pred.append(float(out[i][j]))
    # loss function figure
    x = np.arange(0, 100, 1/2000)
    plt.plot(x,train_loss)
    plt.show()
    # item figure
    plt.scatter(output_true,output_pred,color = 'red', s = 100 )
    plt.show()
    tr.save(myNet,path[size-1])

if __name__ == "__main__":
    __main__()