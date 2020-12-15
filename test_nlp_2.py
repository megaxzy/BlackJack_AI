import scipy
import numpy
import random
import matrix_modified
import time
from scipy import special
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import algo


# 神经网络类定义
class NeuralNetwork():
    # 初始化神经网络
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # 设置输入层节点，隐藏层节点和输出层节点的数量
        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes
        # 学习率设置
        self.lr = learningrate
        # 权重矩阵设置 正态分布

        self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

        # 激活函数设置，sigmod()函数
        self.activation_function = lambda x: scipy.special.expit(x)

    # 训练神经网络
    def train(self,inputs_ori,target_list):
        # 转换输入输出列表到二维数组
        inputs = inputs_ori
        targets = numpy.array(target_list, ndmin=2).T
        # 计算到隐藏层的信号
        hidden_inputs = numpy.dot(self.wih, inputs)
        # 计算隐藏层输出的信号
        hidden_outputs = self.activation_function(hidden_inputs)
        # 计算到输出层的信号
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)


        print("output:",targets)
        print("output:",final_outputs)
        output_errors = targets - final_outputs
        hidden_errors = numpy.dot(self.who.T,output_errors)

        #隐藏层和输出层权重更新
        self.who += self.lr * numpy.dot((output_errors*final_outputs*(1.0-final_outputs)),numpy.transpose(hidden_outputs))
        #输入层和隐藏层权重更新
        self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))

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
    elif (result[1] >= result[0] and result[0] >= result[2]): chose_action = actions[1]
    elif (result[2] >= result[0] and result[2] >= result[1]): chose_action = actions[2]
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

    # 设置学习率为0.3
    learning_rate = 0.3
    # 创建神经网络
    neural = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

    print("training initial *****************************************************")
    for i in range(100):
        # initial game
        game = matrix_modified.game_class(size)  # size of the cards
        for j in range(5):
            print("number:",i*j+j)
            game.start_one_round()
            algorithm = algo.algo_class(game)
            action, drop_action, nodes_number = algorithm.AI_chose_node(0)
            print("Chose Action:",action)

            targets=[]
            if(action == "stay"): targets=[1,0,0]
            if(action == "hit"): targets=[0,1,0]
            if(action == "drop"): targets=[0,0,1]

            neural.train(game.cards.matrix.reshape((52*size*5,1)), targets)
            game.end_one_round()

    print("test initial *****************************************************")
    sum_right_count=0
    for i in range(100):
        # initial game
        game = matrix_modified.game_class(size)  # size of the cards
        for j in range(5):
            print("number:",i*j+j)
            game.start_one_round()
            algorithm = algo.algo_class(game)
            action, drop_action, nodes_number = algorithm.AI_chose_node(0)
            print(action)
            result=neural.query(game.cards.matrix.reshape((52*size*5,1)))
            print(result)
            sum_right_count+=judge_right(action,result)
            print("sum_right_count:",sum_right_count)
            game.end_one_round()






if __name__ == "__main__":
    __main__()








import scipy
import numpy
import random
import matrix_modified
import time
from scipy import special
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import algo_for_nn as algo

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
class NeuralNetwork():
    # initial
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # input nodes
        self.input_nodes = inputnodes
        self.hidden_nodes = hiddennodes
        self.output_nodes = outputnodes

        # learning rate
        self.lr = learningrate

        # weight and bias
        self.wih = numpy.random.normal(0.0, 0.1, (self.hidden_nodes, self.input_nodes))
        self.who = numpy.random.normal(0.0, 0.1, (self.output_nodes, self.hidden_nodes))
        self.bias_1 = numpy.random.normal(0.0, 0.1, (self.hidden_nodes, 1))
        self.bias_2 = numpy.random.normal(0.0, 0.1, (self.output_nodes, 1))

        # sigmod() function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.derived_function = lambda x: self.activation_function(x) * (1 - self.activation_function(x))

    # Train function
    def train(self, inputs_ori, target_list):
        # input
        inputs = inputs_ori
        targets = numpy.array(target_list, ndmin=2).T
        # hidden
        hidden_inputs = numpy.dot(self.wih, inputs) + self.bias_1
        hidden_outputs = self.activation_function(hidden_inputs)
        # output
        final_inputs = numpy.dot(self.who, hidden_outputs) + self.bias_2
        final_outputs = self.activation_function(final_inputs)

        # output
        d_l_d_pre = -2 * (targets - final_outputs)
        d_pre_d_w2 = hidden_outputs * self.derived_function(final_inputs)
        d_pre_d_b2 = self.derived_function(final_inputs)
        d_pre_d_h = self.who * self.derived_function(final_inputs)
        # output update
        self.who -= self.lr * d_l_d_pre * d_pre_d_w2
        self.bias_2 -= self.lr * d_l_d_pre * d_pre_d_b2

        # hidden
        d_h_d_w1 = inputs * self.derived_function(hidden_inputs)
        d_h_d_b1 = self.derived_function(hidden_inputs)
        # hidden update
        self.wih -= self.lr * d_l_d_pre * d_pre_d_h * d_h_d_w1
        self.bias_1 -= self.lr * d_l_d_pre * d_pre_d_h * d_h_d_b1

    # test
    def query(self, input_ori):
        inputs = input_ori
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


def judge_right(right_action, result):
    actions = ["stay", "hit", "drop"]
    chose_action = ""
    if (result[0] >= result[1] and result[0] >= result[2]):
        chose_action = actions[0]
    elif (result[1] >= result[0] and result[1] >= result[2]):
        chose_action = actions[1]
    elif (result[2] >= result[0] and result[2] >= result[1]):
        chose_action = actions[2]
    print("chose_action:", chose_action)
    print("right_action:", right_action)
    if chose_action == right_action:
        print("good job !!!!!")
        return 1
    print("bad !!!!!")
    return 0


def __main__():
    # set
    size = 1
    game = matrix_modified.game_class(size)  # size of the cards
    game.start_one_round()
    algorithm = algo.algo_class(game)
    action, drop_action, nodes_number = algorithm.AI_chose_node(0)

    matrix_desk = numpy.zeros((1, 10))
    matrix_player = numpy.zeros((1, 10))
    matrix_dealer = numpy.zeros((1, 10))

    for i in range(game.cards.matrix.shape[0]):
        for j in range(game.cards.matrix.shape[1]):
            if game.cards.matrix[i][j][0] == 1:
                number = j % 13
                if number >= 9:
                    matrix_desk[0, 9] += 1
                else:
                    matrix_desk[0, number] += 1
            if game.cards.matrix[i][j][1] == 1:
                number = j % 13
                if number >= 9:
                    matrix_player[0, 9] += 1
                else:
                    matrix_player[0, number] += 1
            if game.cards.matrix[i][j][2] == 1:
                number = j % 13
                if number >= 9:
                    matrix_dealer[0, 9] += 1
                else:
                    matrix_dealer[0, number] += 1
    print(game.cards.matrix)
    print(matrix_desk)
    print(matrix_player)
    print(matrix_dealer)


if __name__ == "__main__":
    __main__()

import scipy
import numpy
import random
import matrix_modified
import time
from scipy import special
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import algo_for_nn as algo

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
class NeuralNetwork():
    # initial
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
        # input nodes
        self.input_nodes = inputnodes
        self.hidden_nodes = hiddennodes
        self.output_nodes = outputnodes

        # learning rate
        self.lr = learningrate

        # weight and bias
        self.wih = numpy.random.normal(0.0, 0.1, (self.hidden_nodes, self.input_nodes))
        self.who = numpy.random.normal(0.0, 0.1, (self.output_nodes, self.hidden_nodes))
        self.bias_1 = numpy.random.normal(0.0, 0.1, (self.hidden_nodes, 1))
        self.bias_2 = numpy.random.normal(0.0, 0.1, (self.output_nodes, 1))

        # sigmod() function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.derived_function = lambda x: self.activation_function(x) * (1 - self.activation_function(x))

    # Train function
    def train(self, inputs_ori, target_list):
        # input
        inputs = inputs_ori
        targets = numpy.array(target_list, ndmin=2).T
        # hidden
        hidden_inputs = numpy.dot(self.wih, inputs) + self.bias_1
        hidden_outputs = self.activation_function(hidden_inputs)
        # output
        final_inputs = numpy.dot(self.who, hidden_outputs) + self.bias_2
        final_outputs = self.activation_function(final_inputs)

        # output
        d_l_d_pre = -2 * (targets - final_outputs)
        d_pre_d_w2 = hidden_outputs * self.derived_function(final_inputs)
        d_pre_d_b2 = self.derived_function(final_inputs)
        d_pre_d_h = self.who * self.derived_function(final_inputs)
        # output update
        self.who -= self.lr * d_l_d_pre * d_pre_d_w2
        self.bias_2 -= self.lr * d_l_d_pre * d_pre_d_b2

        # hidden
        d_h_d_w1 = inputs * self.derived_function(hidden_inputs)
        d_h_d_b1 = self.derived_function(hidden_inputs)
        # hidden update
        self.wih -= self.lr * d_l_d_pre * d_pre_d_h * d_h_d_w1
        self.bias_1 -= self.lr * d_l_d_pre * d_pre_d_h * d_h_d_b1

    # test
    def query(self, input_ori):
        inputs = input_ori
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        return final_outputs


def judge_right(right_action, result):
    actions = ["stay", "hit", "drop"]
    chose_action = ""
    if (result[0] >= result[1] and result[0] >= result[2]):
        chose_action = actions[0]
    elif (result[1] >= result[0] and result[1] >= result[2]):
        chose_action = actions[1]
    elif (result[2] >= result[0] and result[2] >= result[1]):
        chose_action = actions[2]
    print("chose_action:", chose_action)
    print("right_action:", right_action)
    if chose_action == right_action:
        print("good job !!!!!")
        return 1
    print("bad !!!!!")
    return 0


def __main__():
    # set
    size = 1

    game = matrix_modified.game_class(size)  # size of the cards
    game.start_one_round()
    algorithm = algo.algo_class(game)
    action, drop_action, nodes_number = algorithm.AI_chose_node(0)

    matrix_desk = numpy.zeros((1, 10))
    matrix_player = numpy.zeros((1, 10))
    matrix_dealer = numpy.zeros((1, 10))

    for i in range(game.cards.matrix.shape[0]):
        for j in range(game.cards.matrix.shape[1]):
            if game.cards.matrix[i][j][0] == 1:
                number = j % 13
                if number >= 9:
                    matrix_desk[0, 9] += 1
                else:
                    matrix_desk[0, number] += 1
            if game.cards.matrix[i][j][1] == 1:
                number = j % 13
                if number >= 9:
                    matrix_player[0, 9] += 1
                else:
                    matrix_player[0, number] += 1
            if game.cards.matrix[i][j][2] == 1:
                number = j % 13
                if number >= 9:
                    matrix_dealer[0, 9] += 1
                else:
                    matrix_dealer[0, number] += 1
    print(game.cards.matrix)
    print(matrix_desk)
    print(matrix_player)
    print(matrix_dealer)


def sss():
    # 设置每层节点个数
    size = 1
    input_nodes = 52 * 5 * size
    hidden_nodes = 30
    output_nodes = 3

    # 设置学习率为0.01
    learning_rate = 0.3
    # 创建神经网络
    neural = NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    sum_s = 0
    sum_h = 0
    sum_d = 0
    print("training initial *****************************************************")
    for i in range(10):
        # initial game
        game = matrix_modified.game_class(size)  # size of the cards
        for j in range(5):
            game.start_one_round()
            algorithm = algo.algo_class(game)
            action, drop_action, nodes_number = algorithm.AI_chose_node(0)
            print("Chose Action:", action)

            targets = []
            if (action == "stay"):
                targets = [1, 0, 0]
                sum_s += 1
            if (action == "hit"):
                targets = [0, 1, 0]
                sum_h += 1
            if (action == "drop"):
                targets = [0, 0, 1]
                sum_d += 1

            neural.train(game.cards.matrix.reshape((52 * size * 5, 1)), targets)
            game.end_one_round()

    print("test initial *****************************************************")
    num = 0
    sum_right_count = 0
    for i in range(10):
        # initial game
        game = matrix_modified.game_class(size)  # size of the cards
        for j in range(5):
            num += 1
            print("number:", num)
            game.start_one_round()
            algorithm = algo.algo_class(game)
            action, drop_action, nodes_number = algorithm.AI_chose_node(0)
            result = neural.query(game.cards.matrix.reshape((52 * size * 5, 1)))
            print(result)
            sum_right_count += judge_right(action, result)
            print("sum_right_count:", sum_right_count)
            game.end_one_round()

    print(neural.who)
    print(neural.wih)
    print(sum_s)
    print(sum_d)
    print(sum_h)


sss()

if __name__ == "__main__":
    __main__()




import scipy
import numpy
import random
import matrix_modified
import time
from scipy import special
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import algo_for_nn as algo


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
        self.wih = numpy.random.normal(0.0, 0.1, (self.hidden_nodes, self.input_nodes))
        self.who = numpy.random.normal(0.0, 0.1, (self.output_nodes, self.hidden_nodes))
        self.bias_1 = numpy.random.normal(0.0, 0.1,(self.hidden_nodes,1))
        self.bias_2 = numpy.random.normal(0.0, 0.1,(self.output_nodes,1))

        # sigmod() function
        self.activation_function = lambda x: scipy.special.expit(x)
        self.derived_function = lambda x:self.activation_function(x)*(1-self.activation_function(x))

    # Train function
    def train(self,inputs_ori,target_list):
        # input
        inputs = inputs_ori
        targets = numpy.array(target_list, ndmin=2).T
        # hidden
        hidden_inputs = numpy.dot(self.wih, inputs)+self.bias_1
        hidden_outputs = self.activation_function(hidden_inputs)
        # output
        final_inputs = numpy.dot(self.who, hidden_outputs)+self.bias_2
        final_outputs = self.activation_function(final_inputs)

        # output
        d_l_d_pre = -2 * (targets - final_outputs)
        d_pre_d_w2 = hidden_outputs.T * self.derived_function(final_inputs)
        d_pre_d_b2 = self.derived_function(final_inputs)
        d_pre_d_h = self.who * self.derived_function(final_inputs)
        # output update
        self.who -= self.lr * d_l_d_pre * d_pre_d_w2
        self.bias_2 -= self.lr * d_l_d_pre * d_pre_d_b2

        # hidden
        d_h_d_w1 = inputs * self.derived_function(hidden_inputs)
        d_h_d_b1 = self.derived_function(hidden_inputs)
        # hidden update
        self.wih -= self.lr * d_l_d_pre * d_pre_d_h * d_h_d_w1
        self.bias_1 -= self.lr * d_l_d_pre * d_pre_d_h * d_h_d_b1

    # test
    def query(self, input_ori):
        inputs = input_ori
        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
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

    # set
    size = 1

    game = matrix_modified.game_class(size)  # size of the cards
    game.start_one_round()
    algorithm = algo.algo_class(game)
    action, drop_action, nodes_number = algorithm.AI_chose_node(0)

    matrix_desk = numpy.zeros((1,10))
    matrix_player = numpy.zeros((1, 10))
    matrix_dealer = numpy.zeros((1, 10))

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
    print(matrix_desk)
    print(matrix_player)
    print(matrix_dealer)





if __name__ == "__main__":
    __main__()

import scipy
import numpy as np
import random
import matrix_modified
import time
from scipy import special
from scipy.special import expit, logit
import matplotlib.pyplot as plt
import algo_for_nn as algo
import NN


def sigmoid(x):
    # Sigmoid activation function: f(x) = 1 / (1 + e^(-x))
    return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
    # Derivative of sigmoid: f'(x) = f(x) * (1 - f(x))
    fx = sigmoid(x)
    return fx * (1 - fx)


def mse_loss(y_true, y_pred):
    # y_true and y_pred are numpy arrays of the same length
    return ((y_true - y_pred) ** 2).mean()


class OurNeuralNetwork():
    """
    A neural network with:
      - 2 inputs
      - a hidden layer with 2 neurons (h1, h2)
      - an output layer with 1 neuron (o1)

    *** DISCLAIMER ***
    The code below is intend to be simple and educational, NOT optimal.
    Real neural net code looks nothing like this. Do NOT use this code.
    Instead, read/run it to understand how this specific network works.
    """

    def __init__(self):
        # weights
        self.w1 = np.random.normal()
        self.w2 = np.random.normal()
        self.w3 = np.random.normal()
        self.w4 = np.random.normal()
        self.w5 = np.random.normal()
        self.w6 = np.random.normal()
        # biases
        self.b1 = np.random.normal()
        self.b2 = np.random.normal()
        self.b3 = np.random.normal()

    def feedforward(self, x):
        # x is a numpy array with 2 elements, for example [input1, input2]
        h1 = sigmoid(self.w1 * x[0] + self.w2 * x[1] + self.b1)
        h2 = sigmoid(self.w3 * x[0] + self.w4 * x[1] + self.b2)
        o1 = sigmoid(self.w5 * h1 + self.w6 * h2 + self.b3)
        return o1

    def train(self, data, all_y_trues):
        """
        - data is a (n x 2) numpy array, n = # samples in the dataset.
        - all_y_trues is a numpy array with n elements.
        Elements in all_y_trues correspond to those in data.
        """
        learn_rate = 0.1
        epochs = 1000  # number of times to loop through the entire dataset

        for epoch in range(epochs):
            for x, y_true in zip(data, all_y_trues):
                # - - - Do a feedforward (we'll need these values later)
                sum_h1 = self.w1 * x[0] + self.w2 * x[1] + self.b1
                h1 = sigmoid(sum_h1)

                sum_h2 = self.w3 * x[0] + self.w4 * x[1] + self.b2
                h2 = sigmoid(sum_h2)

                sum_o1 = self.w5 * sum_h1 + self.w6 * sum_h2 + self.b3
                o1 = sigmoid(sum_o1)
                y_pred = o1

                # - - - Calculate partial derivatives.
                # - - - Naming: d_L_d_w1 represents "partial L / partial w1"
                d_L_d_ypred = -2 * (y_true - y_pred)

                # Neuron o1
                d_ypred_d_w5 = h1 * deriv_sigmoid(sum_o1)
                d_ypred_d_w6 = h2 * deriv_sigmoid(sum_o1)
                d_ypred_d_b3 = deriv_sigmoid(sum_o1)

                d_ypred_d_h1 = self.w5 * deriv_sigmoid(sum_o1)
                d_ypred_d_h2 = self.w6 * deriv_sigmoid(sum_o1)

                # Neuron h1
                d_h1_d_w1 = x[0] * deriv_sigmoid(sum_h1)
                d_h1_d_w2 = x[1] * deriv_sigmoid(sum_h1)
                d_h1_d_b1 = deriv_sigmoid(sum_h1)

                # Neuron h2
                d_h2_d_w3 = x[0] * deriv_sigmoid(sum_h2)
                d_h2_d_w4 = x[1] * deriv_sigmoid(sum_h2)
                d_h2_d_b2 = deriv_sigmoid(sum_h2)

                # - - - update weights and biases
                # Neuron o1
                self.w5 -= learn_rate * d_L_d_ypred * d_ypred_d_w5
                self.w6 -= learn_rate * d_L_d_ypred * d_ypred_d_w6
                self.b3 -= learn_rate * d_L_d_ypred * d_ypred_d_b3

                # Neuron h1
                self.w1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w1
                self.w2 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_w2
                self.b1 -= learn_rate * d_L_d_ypred * d_ypred_d_h1 * d_h1_d_b1

                # Neuron h2
                self.w3 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w3
                self.w4 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_w4
                self.b2 -= learn_rate * d_L_d_ypred * d_ypred_d_h2 * d_h2_d_b2

            # - - - Calculate total loss at the end of each epoch
            if epoch % 10 == 0:
                y_preds = np.apply_along_axis(self.feedforward, 1, data)
                loss = mse_loss(all_y_trues, y_preds)
                print("Epoch %d loss: %.3f", (epoch, loss))


# Define dataset
data = np.array([
    [-2, -1],  # Alice
    [25, 6],  # Bob
    [17, 4],  # Charlie
    [-15, -6]  # diana
])
all_y_trues = np.array([
    1,  # Alice
    0,  # Bob
    0,  # Charlie
    1  # diana
])

# Train our neural network!
network = OurNeuralNetwork()
for i in range(3):
    network.train(data, all_y_trues)

# Make some predictions
emily = np.array([-7, -3])  # 128 pounds, 63 inches
frank = np.array([20, 2])  # 155 pounds, 68 inches
print("Emily: %.3f" % network.feedforward(emily))  # 0.951 - F
print("Frank: %.3f" % network.feedforward(frank))  # 0.039 - M

'''
def __main__():
    # 设置每层节点个数
    size = 1
    input_nodes = 52 *5 * size
    hidden_nodes = 30
    output_nodes = 3

    # 设置学习率为0.3
    learning_rate = 0.1
    # 创建神经网络
    neural = NN.NeuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)
    sum_s=0
    sum_h=0
    sum_d=0
    print("training initial *****************************************************")
    for i in range(100):
        # initial game
        game = matrix_modified.game_class(size)  # size of the cards
        for j in range(5):
            # print("number:",i*j+j)
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
    for i in range(20):
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
            sum_right_count+= NN.judge_right(action,result)
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











