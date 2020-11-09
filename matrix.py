import numpy as np
import random



class cards_class:


    def __init__(self):
        self.matrix = np.zeros((4, 52, 5))  # 4 * 52 * 5 pokers
        self.matrix[0:4, 0:52, 0:1] = 1  # 0 1 2 3 4 desk,player,dealer_show,dealer_hide,drop   1 exist 0 not exist
    def hand_to_player(self,x,y):
        if self.matrix[x][y][0] == 1 :
            self.matrix[x][y][0] = 0
            self.matrix[x][y][1] = 1
            return True
        else:
            return False
    def hand_to_dealer(self,x,y,is_hide):  # is_hide = 0; show is_hide=1 hide
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
    def __init__(self):
        self.cards = cards_class()
        self.player = player_class(100)
        self.list_player_cards = []
        self.list_dealer_cards = []
        #TODO

        # J Q K = 10 A =1 / 11
        self.player_sum=0
        self.dealer_sum_show=0
        self.dealer_sum_all = 0
        print("game create ok")
    def start_one_round(self):
        while(1):
            ran_01 = random.randint(0, 3)
            ran_02 = random.randint(0, 51)
            if self.cards.hand_to_player(ran_01,ran_02):
                self.list_player_cards.append(self.cards.show_one_position(ran_02))
                if(ran_02 % 13 +1 >=10):
                    self.player_sum = self.player_sum + 10
                else:
                    self.player_sum = self.player_sum+ran_02 % 13+1
                break
        while(1):
            ran_01 = random.randint(0, 3)
            ran_02 = random.randint(0, 51)
            if self.cards.hand_to_player(ran_01,ran_02):
                self.list_player_cards.append(self.cards.show_one_position(ran_02))
                if(ran_02 % 13 +1 >=10):
                    self.player_sum = self.player_sum + 10
                else:
                    self.player_sum = self.player_sum+ran_02 % 13+1
                break
        while(1):
            ran_01 = random.randint(0, 3)
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
            ran_01 = random.randint(0, 3)
            ran_02 = random.randint(0, 51)
            if self.cards.hand_to_dealer(ran_01,ran_02,1):
                self.list_dealer_cards.append(self.cards.show_one_position(ran_02))
                if(ran_02 % 13 +1 >=10):
                    self.dealer_sum_all = self.dealer_sum_all + 10
                else:
                    self.dealer_sum_all = self.dealer_sum_all + ran_02 % 13 + 1
                break
        print("start one game")

    def show_game_condition(self):
        print ("dealer:" + str(self.dealer_sum_all))
        print ("      " + str(self.list_dealer_cards))
        print ("player:" + str(self.player_sum))
        print ("      " + str(self.list_player_cards))
        '''
        for i in range(0,4):
            for j in range(0,52):
                if(self.cards.matrix[i][j][0]==0):
                    print(i,j)
        '''

    def get_best_action(self):
        actions=["hit","stay","drop"]
        best_actions=[]
        if self.player_sum == 21:
            return best_actions.append("stay")
        max_pro=0
        for hit_times in range(0,2): #TODO
            self.minmax(hit_times,self.dealer_sum_all,self.player_sum,0,0,0)  #TODO


    def minmax(self,hit_times,dealer_sum,player_sum,win_times,lose_times,tie_times):
        if hit_times == 0:
            list_score_times=[0,0,0]
            return self.dealer_turn(dealer_sum,list_score_times,self.cards.matrix,player_sum)
        for i in range(0,4):
            for j in range(0,52):
                if self.cards.matrix[i][j][0]==1:
                    self.cards.matrix[i][j][0] = 0
                    self.minmax(hit_times,dealer_sum,player_sum+j%13+1,win_times,lose_times,tie_times)
                    self.cards.matrix[i][j][0] = 1
    def dealer_turn(self,dealer_sum,list_score_times,temp_cards_matrix,player_sum):
        if dealer_sum>=16:
            if dealer_sum==player_sum:
                list_score_times[2] + 1  #tie
            if dealer_sum >player_sum:
                list_score_times[1] + 1  #lose
            if dealer_sum < player_sum:
                list_score_times[0] +1  #win
            return list_score_times
        temp_temp_cards_matrix=temp_cards_matrix.copy()

        for i in range(0,4):
            for j in range(0,52):
                if temp_cards_matrix[i][j][0]==1:
                    temp_cards_matrix[i][j][0]=0
                    self.dealer_turn(dealer_sum+j%13+1, list_score_times, self.cards.matrix, player_sum)
                    temp_cards_matrix=temp_temp_cards_matrix
        




    def end_one_round(self):
        self.list_player_cards = []
        self.list_dealer_cards = []
        self.player_sum=0
        self.dealer_sum_show=0
        self.dealer_sum_all = 0


def __main__():

    game = game_class()
    game.start_one_round()
    #print(game.cards.matrix)
    game.show_game_condition()


if __name__ == "__main__":
    __main__()

