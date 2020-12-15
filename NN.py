import numpy as np
import torch as tr
from torch.nn import Sequential, Conv2d, Linear, Flatten, LeakyReLU, Tanh
import torch.nn.functional
import matrix_modified
import algo_for_nn as algo
import torch
import torch.nn as nn
import numpy as np
import time
import os

path = ['imple_4_size_1.pkl','imple_4_size_2.pkl','imple_4_size_3.pkl','imple_4_size_4.pkl','imple_4_size_5.pkl']
#path_test = ['imple_4_size_1.pkl','imple_4_size_2.pkl','imple_4_size_3.pkl','imple_4_size_4.pkl','imple_4_size_5.pkl']
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

def NN_play():

    print("Hello I'm NN AI ")
    while(1):
        size = int(input("Please input the size you want(must be int and must more than zero): "))
        if size>0:
            break
    load_net = tr.load(path[size-1])
    game = matrix_modified.game_class(size)  #size of the cards

    while(1):
        game.start_one_round()
        game.show_game_condition_for_play()
        can_drop=0
        drop_action=0
        os.system("pause")
        print()
        while(1):
            test_item = game.cards.matrix.reshape((1, size * 52 * 5))
            test_item = torch.tensor(torch.from_numpy(test_item)).float()
            result = load_net(test_item)
            action=judge_chose(result[0],can_drop)
            if action=="drop":
                can_drop=1
            end, win = game.ai_input_action(action, drop_action)
            if end==1:
                break
            os.system("pause")
        game.end_one_round()
        val = input("Do you want NN start one new around(0 is yes; 1 is no)")
        if(val=="1"):break

def __main__():
    NN_play()

if __name__ == "__main__":
    __main__()




























'''
import scipy
import numpy
import random
import matrix_modified
import time
from scipy import special
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import algo_for_nn as algo
class NeuralNetwork():
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        # input nodes
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        #learning rate
        self.lr = learningrate

        # weight and bias
        #self.wih = numpy.zeros((self.hnodes, self.inodes))
        #self.who = numpy.zeros((self.onodes, self.hnodes))
        self.wih = numpy.random.normal(0.0, 0.1, (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, 0.1, (self.onodes, self.hnodes))
        self.bias_1 = numpy.zeros(())
        self.bias_2 = numpy.zeros((self.onodes,1))

        #self.wih = numpy.zeros((self.hnodes, self.inodes))
        #self.who = numpy.zeros((self.onodes, self.hnodes))

        # sigmod() function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.derived_function = lambda x:self.activation_function(x)*(1-self.activation_function(x))

    # Train function
    def train(self,inputs_ori,target_list):
        # input
        inputs = inputs_ori
        targets = numpy.array(target_list, ndmin=2).T
        # hidden
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        # output
        final_inputs = numpy.dot(self.who, hidden_outputs)+self.bias_2
        final_outputs = self.activation_function(final_inputs)

        # calculate errors
        output_errors = -2 * (targets - final_outputs)
        hidden_errors = numpy.dot(self.who.T,output_errors) #loss

        gra_who = numpy.dot(self.derived_function(output_errors),self.activation_function(numpy.transpose(hidden_inputs)))
        self.who -= self.lr * gra_who
        gra_whi = numpy.dot(self.derived_function(hidden_errors,),self.activation_function(numpy.transpose(inputs)))
        self.wih -= self.lr * gra_whi
        #self.bias_2 -= self.lr*output_errors*0.01

        #renew
        #gra_who = numpy.dot((output_errors * final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        #self.who += self.lr * numpy.dot((output_errors * final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        #renew
        #self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0-hidden_outputs)),numpy.transpose(inputs))


    # 查询神经网络
    def query(self, input_ori):
        # 转换输入列表到二维数组
        inputs = input_ori
        # 计算到隐藏层的信号
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算隐藏层输出的信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算到输出层的信号
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

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
    # 设置每层节点个数
    size = 1
    input_nodes = 52 *5 * size
    hidden_nodes = 30
    output_nodes = 3

    # 设置学习率为0.01
    learning_rate = 0.3
    # 创建神经网络
    neural = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    sum_s=0
    sum_h=0
    sum_d=0
    print("training initial *****************************************************")
    for i in range(10):
        # initial game
        game = matrix_modified.game_class(size)  # size of the cards
        for j in range(5):
            game.start_one_round()
            algorithm = algo.algo_class(game)
            action, drop_action, nodes_number = algorithm.AI_chose_node(0)
            print("Chose Action:",action)

            targets=[]
            if(action == "stay"):
                targets=[1,0,0]
                sum_s+=1
            if(action == "hit"):
                targets=[0,1,0]
                sum_h+=1
            if(action == "drop"):
                targets=[0,0,1]
                sum_d+=1

            neural.train(game.cards.matrix.reshape((52*size*5,1)), targets)
            game.end_one_round()



    print("test initial *****************************************************")
    num=0
    sum_right_count=0
    for i in range(10):
        # initial game
        game = matrix_modified.game_class(size)  # size of the cards
        for j in range(5):
            num+=1
            print("number:", num)
            game.start_one_round()
            algorithm = algo.algo_class(game)
            action, drop_action, nodes_number = algorithm.AI_chose_node(0)
            result=neural.query(game.cards.matrix.reshape((52*size*5,1)))
            print(result)
            sum_right_count+=judge_right(action,result)
            print("sum_right_count:",sum_right_count)
            game.end_one_round()

    print(neural.who)
    print(neural.wih)
    print(sum_s)
    print(sum_d)
    print(sum_h)




if __name__ == "__main__":
    __main__()

'''





