
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

'''
def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))

def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)
'''

# NN
'''
class NeuralNetwork():
    # initial
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        # input nodes
        self.input_nodes = inputnodes
        self.hidden_nodes = hiddennodes
        self.output_nodes = outputnodes

        #learning rate
        self.lr = learningrate

        # weight and bias
        self.wih = np.random.normal(0.0, 0.1, (self.hidden_nodes, self.input_nodes))
        self.who = np.random.normal(0.0, 0.1, (self.output_nodes, self.hidden_nodes))
        self.bias_1 = np.random.normal(0.0, 0.1,(self.hidden_nodes,1))
        self.bias_2 = np.random.normal(0.0, 0.1,(self.output_nodes,1))

        # sigmod() function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.derived_function = lambda x:self.activation_function(x)*(1-self.activation_function(x))

    # Train function
    def train(self,inputs_ori,target_list):
        # input
        inputs = inputs_ori
        targets = np.array(target_list, ndmin=2).T
        # hidden
        hidden_inputs = np.dot(self.wih, inputs)+self.bias_1
        hidden_outputs = self.activation_function(hidden_inputs)
        # output
        final_inputs = np.dot(self.who, hidden_outputs)+self.bias_2
        final_outputs = self.activation_function(final_inputs)

        # output
        d_l_d_pre = -2 * (targets - final_outputs)
        d_pre_d_w2 = hidden_outputs.T * self.derived_function(final_inputs)
        d_pre_d_b2 = self.derived_function(final_inputs)
        print( d_l_d_pre.shape)
        print(d_pre_d_w2.shape)
        print(self.who.shape)
        # output update
        self.who -= self.lr * np.dot(d_l_d_pre * d_pre_d_w2)
        self.bias_2 -= self.lr * d_l_d_pre * d_pre_d_b2

        # hidden
        d_h_d_w1 = inputs.T * self.derived_function(hidden_inputs)
        d_pre_d_h = self.who * self.derived_function(final_inputs)
        d_h_d_b1 = self.derived_function(hidden_inputs)
        # hidden update
        print(self.wih.shape)
        print(d_l_d_pre.shape)
        print(d_pre_d_h.shape)
        print(d_h_d_w1.shape)
        self.wih -= self.lr * d_l_d_pre * d_pre_d_h * d_h_d_w1
        self.bias_1 -= self.lr * d_l_d_pre * d_pre_d_h * d_h_d_b1

    # test
    def query(self, input_ori):
        inputs = input_ori
        hidden_inputs = np.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = np.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs
'''

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


def __main__():
    path = ['imple_3_size_1.pkl', 'imple_3_size_2.pkl', 'imple_3_size_3.pkl', 'imple_3_size_4.pkl',
            'imple_3_size_5.pkl']

    size=1
    myNet = nn.Sequential(
        nn.Linear(30, 9),
        nn.Tanh(),
        nn.Linear(9, 3),
        nn.Sigmoid()
    )
    print(myNet)

    # set optimzer
    optimzer = torch.optim.SGD(myNet.parameters(), lr=0.001)
    loss_func = nn.MSELoss()



    print("training initial *****************************************************")


    train_loss= []
    output_true,output_pred=[],[]
    it = 100
    for i in range(it):
        train_data = np.zeros((1, 30))
        target_data = np.zeros((1, 3))
        for k in range(5):
            game = matrix_modified.game_class(size)  # size of the cards

            for j in range(5):
                game.start_one_round()
                algorithm = algo.algo_class(game)
                action, drop_action, nodes_number,win_rates= algorithm.AI_chose_node_2(0)
                targets=[win_rates]
                train_item=data_processing(game)
                output_true.append(win_rates[0])
                output_true.append(win_rates[1])
                output_true.append(win_rates[2])
                #print(train_data)
                train_data=np.r_[train_data, train_item]
                target_data=np.r_[target_data,targets]
                game.end_one_round()

        train_data_torch = torch.tensor(torch.from_numpy(train_data[1:train_data.shape[0]])).float()
        target_data_torch = torch.tensor(torch.from_numpy(target_data[1:target_data.shape[0]])).float()

        for epoch in range(2000):
            out = myNet(train_data_torch)
            loss = loss_func(out, target_data_torch)
            train_loss.append(loss)
            optimzer.zero_grad()
            loss.backward()
            optimzer.step()
        out=myNet(train_data_torch)
        for i in range(len(out)):
            output_pred.append(float(out[i][0]))
            output_pred.append(float(out[i][1]))
            output_pred.append(float(out[i][2]))

    # loss function figure
    r = np.arange(0, it, 1/2000)
    plt.plot(r,train_loss)
    plt.show()

    # error figure
    plt.scatter(output_true,output_pred, s=50)
    plt.show()
    tr.save(myNet,path[size-1])



if __name__ == "__main__":
    __main__()
















