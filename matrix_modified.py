import numpy as np
import random



class cards_class:

    size=0
    def __init__(self,size):
        self.size=size
        self.matrix = np.zeros((size, 52, 5))  # 4 * 52 * 5 pokers
        self.matrix[0:size, 0:52, 0:1] = 1  # 0 1 2 3 4 desk0,player1,dealer_show2,dealer_hide3,drop4   1 exist 0 not exist

    def hand_to_player(self,x,y):
        if self.matrix[x][y][0] == 1 :
            self.matrix[x][y][0] = 0
            self.matrix[x][y][1] = 1
            return True
        else:
            return False


    def hand_to_dealer(self,x,y,is_hide):  # is_hide = 0 show is  ;_hide=1 hide
        if self.matrix[x][y][0] == 1 :
            self.matrix[x][y][0] = 0
            if is_hide==0:
                self.matrix[x][y][2] = 1
            if is_hide==1:
                self.matrix[x][y][3] = 1
            return True
        else:
            return False


    def show_one_position(self,y):  # spade heart diamond clubs
        return [y//13, y % 13+1]
    def show_whole_position(self):
        print(self.matrix)

class player_class:

    def __init__(self,start_value):
        self.start_value=start_value

    def change_money(self,is_win,values):  #0 tie 1 win -1 lose
        self.start_value= self.start_value+is_win * values


class game_class:
    size=0
    def __init__(self,size):
        self.size=size
        self.cards = cards_class(self.size)
        self.player = player_class(100)
        self.list_player_cards = []
        self.list_dealer_cards = []
        #TODO

        # J Q K = 10 A =1 / 11
        self.player_sum=0
        self.dealer_sum_show=0
        self.dealer_sum_all = 0
        print("Game create ok")
        print("Playing time")

    def start_one_round(self):
        while(1):
            ran_01 = random.randint(0, self.size-1)
            ran_02 = random.randint(0, 51)
            if self.cards.hand_to_player(ran_01,ran_02):
                self.list_player_cards.append(self.cards.show_one_position(ran_02))
                if(ran_02 % 13 +1 >=10):
                    self.player_sum = self.player_sum + 10
                else:
                    self.player_sum = self.player_sum+ran_02 % 13+1
                break
        while(1):
            ran_01 = random.randint(0, self.size-1)
            ran_02 = random.randint(0, 51)
            if self.cards.hand_to_player(ran_01,ran_02):
                self.list_player_cards.append(self.cards.show_one_position(ran_02))
                if(ran_02 % 13 +1 >=10):
                    self.player_sum = self.player_sum + 10
                else:
                    self.player_sum = self.player_sum+ran_02 % 13+1
                break
        while(1):
            ran_01 = random.randint(0, self.size-1)
            ran_02 = random.randint(0, 51)
            if self.cards.hand_to_dealer(ran_01,ran_02,0):
                self.list_dealer_cards.append(self.cards.show_one_position(ran_02))
                if(ran_02 % 13 +1 >=10):
                    self.dealer_sum_show = self.dealer_sum_show + 10
                    self.dealer_sum_all = self.dealer_sum_all + 10
                else:
                    self.dealer_sum_show = self.dealer_sum_show + ran_02 % 13 + 1
                    self.dealer_sum_all = self.dealer_sum_all + ran_02 % 13 + 1
                break
        while(1):
            ran_01 = random.randint(0, self.size-1)
            ran_02 = random.randint(0, 51)
            if self.cards.hand_to_dealer(ran_01,ran_02,1):
                self.list_dealer_cards.append(self.cards.show_one_position(ran_02))
                if(ran_02 % 13 +1 >=10):
                    self.dealer_sum_all = self.dealer_sum_all + 10
                else:
                    self.dealer_sum_all = self.dealer_sum_all + ran_02 % 13 + 1
                break
        print("start one game")

    def show_game_condition_for_test(self):
        print ("******************************************************************************************************")
        print("For test:")
        print ("dealer:" + str(self.dealer_sum_all))
        print ("      " + str(self.list_dealer_cards))
        print ("player:" + str(self.player_sum))
        print ("      " + str(self.list_player_cards))
        print("******************************************************************************************************")
        '''
        for i in range(0,4):
            for j in range(0,52):
                if(self.cards.matrix[i][j][0]==0):
                    print(i,j)
        '''
    def show_game_condition_for_play(self):
        print ("******************************************************************************************************")
        print("For play:")
        print ("dealer:" + str(self.dealer_sum_all))
        list_show_dealer=[]
        for item in self.list_dealer_cards:
            if (item[1] == 1):
                list_show_dealer.append("A")
            else:
                if (item[1] == 11):
                    list_show_dealer.append("J")
                else:
                    if (item[1] == 12):
                        list_show_dealer.append("Q")
                    else:
                        if (item[1] == 13):
                            list_show_dealer .append("K")
                        else:
                            list_show_dealer.append(str(item[1]))
        print ("      " + str(list_show_dealer))
        list_show_player = []
        for item in self.list_player_cards:
            if (item[1] == 1):
                list_show_player.append("A")
            else:
                if (item[1] == 11):
                    list_show_player.append("J")
                else:
                    if (item[1] == 12):
                        list_show_player.append("Q")
                    else:
                        if (item[1] == 13):
                            list_show_player .append("K")
                        else:
                            list_show_player.append(str(item[1]))
        print ("player:" + str(self.player_sum))
        print ("      " + str(list_show_player))
        print ("******************************************************************************************************")
        '''
        for i in range(0,4):
            for j in range(0,52):
                if(self.cards.matrix[i][j][0]==0):
                    print(i,j)
        '''
    def end_one_round(self):
        self.list_player_cards = []
        self.list_dealer_cards = []
        self.player_sum=0
        self.dealer_sum_show=0
        self.dealer_sum_all = 0

    def dealer_auto(self):
        while(self.dealer_sum_all<16):
            while(1):
                ran_01 = random.randint(0, self.size-1)
                ran_02 = random.randint(0, 51)

                if self.cards.hand_to_dealer(ran_01, ran_02, 0):
                    self.list_dealer_cards.append(self.cards.show_one_position(ran_02))
                    if (ran_02 % 13 + 1 >= 10):
                        self.dealer_sum_all = self.dealer_sum_all + 10
                    else:
                        self.dealer_sum_all = self.dealer_sum_all + ran_02 % 13 + 1
                    break

    def player_auto(self):
        while (1):
            ran_01 = random.randint(0, self.size-1)
            ran_02 = random.randint(0, 51)

            if self.cards.hand_to_player(ran_01, ran_02):
                self.list_player_cards.append(self.cards.show_one_position(ran_02))
                if (ran_02 % 13 + 1 >= 10):
                    self.player_sum = self.player_sum + 10
                else:
                    self.player_sum  = self.player_sum  + ran_02 % 13 + 1
                break

    def judge_winner(self):   #0 is tie; 1 is player ; 2 is dealer
        if(self.player_sum>21):
            return 2
        self.dealer_auto()
        #print(self.dealer_sum_all)
        if(self.dealer_sum_all==self.player_sum):
            return 0
        if(self.dealer_sum_all>21):
            return 1
        if(self.dealer_sum_all>self.player_sum):
            return 2
        if(self.dealer_sum_all<self.player_sum):
            return 1

    def input_action(self):  #TODO
        can_drop=0
        while(1):
            if(can_drop==0):
                value=input("Please input actions (0 is stay; 1 is hit; 2 is to drop the card):")

            else:
                value = input("Please input actions (0 is stay; 1 is hit; can not drop):")
            if (value=='0'):
                win=self.judge_winner()

                self.show_game_condition_for_play()
                if(win==0):
                    print("tie!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if(win==1):
                    print("player win!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                if(win==2):
                    print("dealer win!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                break
            if (value == '1'):
                self.player_auto()
                self.show_game_condition_for_play()
                if(self.player_sum>21):
                    print("dealer win!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                    break
            if (value == '2' and can_drop==0):
                can_drop = 1
                ind_drop = input("Please input which card you want to drop from 0 to "+ str(len(self.list_player_cards)-1)+":")
                ind_drop = int(ind_drop)
                self.recalculate_player_list_and_sum(ind_drop)
                self.show_game_condition_for_play()

    def recalculate_player_list_and_sum(self,ind_drop):
        self.list_player_cards.pop(ind_drop)
        self.player_sum = 0
        for item in self.list_player_cards:
            if item[1] > 10 :
                self.player_sum+=10
            else:
                self.player_sum+=item[1]

    def show_all_matrix_in_hands(self):
        print("show_all_matrix_in_hands:")
        for i in range(0,self.size):
            for j in range(0,52):

                if self.cards.matrix[i][j][1]==1:
                    print("In player hands :", i, j)
                if self.cards.matrix[i][j][2]==1:
                    print("In dealer hands show:", i, j)
                if self.cards.matrix[i][j][3]==1:
                    print("In dealer hands hide:",i,j)


def __main__():
    game = game_class(1)  #size of the cards

    while(1):
        game.start_one_round()
        game.show_all_matrix_in_hands()
        game.show_game_condition_for_play()
        game.input_action()
        game.end_one_round()
        val=input("Do you want to start one new around(0 is yes; 1 is no)")
        if(val=="1"):break

if __name__ == "__main__":
    __main__()
