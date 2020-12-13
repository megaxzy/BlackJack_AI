import numpy as np
import random
import matrix_modified
import time



class node_class:
    temp_player_sum=0
    temp_dealer_sum=0
    temp_player_list = []
    list_scores=[]
    can_drop=0
    def __init__(self,matrix):
        self.temp_matrix=matrix
    def show_all_values(self):
        print("******************************************************************************************")
        print("matrix size:" + str(self.temp_matrix.shape))
        print("player_sum:" + str(self.temp_player_sum))
        print("dealer_sum:" + str(self.temp_dealer_sum))
        print("player_list:" + str(self.temp_player_list))
        print("list_scores:" + str(self.list_scores))
        print("can_drop:" + str(self.can_drop))
        print("******************************************************************************************")


class algo_class:
    actions = ["stay","hit" , "drop"]
    nodes_number=0
    def __init__(self,game):
        self.game=game

    def turn_to_value(self,value):
        card=value%13+1
        if card>10:
            card=10
        return int(card)

    def dealer_turn(self,dealer_sum, player_sum, temp_cards_matrix, list_score_times, s):
        if dealer_sum >= 22:

            list_score_times[0] = list_score_times[0] + 100 / s  # win
            return list_score_times
        if dealer_sum >= 17:

            if dealer_sum == player_sum:
                list_score_times[2] = list_score_times[2] + 100 / s  # tie
            if dealer_sum > player_sum:
                list_score_times[1] = list_score_times[1] + 100 / s  # lose
            if dealer_sum < player_sum:
                list_score_times[0] = list_score_times[0] + 100 / s  # win
            return list_score_times
        for i in range(0, 1):
            sum = 0
            for j in range(0, 13):
                if temp_cards_matrix[i][j][0] == 1 or temp_cards_matrix[i][j][3] == 1:
                    sum = sum + 1
            for j in range(0, 13):
                if temp_cards_matrix[i][j][0]==1 or temp_cards_matrix[i][j][3]== 1:
                    temp_cards_matrix[i][j][0] = 0
                    list_score_times = self.dealer_turn(dealer_sum + self.turn_to_value(j), player_sum, temp_cards_matrix,
                                                   list_score_times, s * sum)
                    temp_cards_matrix[i][j][0] = 1
        return list_score_times

    def AI_chose_node(self,can_drop):
        node = node_class(self.game.cards.matrix.copy())
        node.temp_player_sum = self.game.player_sum
        node.temp_dealer_sum = self.game.dealer_sum_show
        node.temp_player_list = self.game.list_player_cards
        node.list_scores=[0,0,0]
        node.can_drop=can_drop
        node.show_all_values()

        self.nodes_number=0
        time.sleep(1)
        list_scores,action,index=self.max_expected_tree(node,1)
        return action,index,self.nodes_number

    def max_expected_tree(self,node,deep):
        self.nodes_number=self.nodes_number+1
        # stay
        node_temp = node
        #if(node.temp_player_sum<=11):
        #    return [100,0,0],"hit", int(0)   # can not do this
        list_stay_score_times =self.dealer_turn(node_temp.temp_dealer_sum, node_temp.temp_player_sum,
                                                            node_temp.temp_matrix, [0,0,0],1)

        #print(list_stay_score_times)
        if deep==3:       # TODO
            return list_stay_score_times, "stay", 0

        #time.sleep(1)
        # hit
        #print("hit action "+str(deep))
        list_hit_score_times = [0,0,0]
        node_temp = node

        drop_index = 0
        for i in range(0,self.game.size):
            for j in range(0,51):
                add_values_hit = [0,0,0]
                if node_temp.temp_matrix[i][j][0]==1:
                    node_temp.temp_matrix[i][j][0] = 0
                    node_temp.temp_player_sum=node_temp.temp_player_sum + self.turn_to_value(j)
                    if node_temp.temp_player_sum > 21:
                        add_values_hit=[0,100,0]
                    else:
                        add_values_hit,chose_action,drop_index = self.max_expected_tree(node_temp,deep+1)
                    node_temp.temp_player_sum = node_temp.temp_player_sum - self.turn_to_value(j)
                    node_temp.temp_matrix[i][j][0] = 1
                if node_temp.temp_matrix[i][j][3]==1:
                    node_temp.temp_matrix[i][j][3] = 0
                    node_temp.temp_player_sum = node_temp.temp_player_sum + self.turn_to_value(j)
                    if node_temp.temp_player_sum > 21:
                        add_values_hit=[0,100,0]
                    else:
                        add_values_hit,chose_action,drop_index = self.max_expected_tree(node_temp,deep+1)
                    node_temp.temp_player_sum = node_temp.temp_player_sum - self.turn_to_value(j)
                    node_temp.temp_matrix[i][j][3]= 1

                for k in range(0,3):
                    list_hit_score_times[k] = list_hit_score_times[k] + add_values_hit[k]
        sum_score=sum(list_hit_score_times)
        for k in range(0,3):
            list_hit_score_times[k] = 100*list_hit_score_times[k]/sum_score
        #print(list_hit_score_times)

        #time.sleep(1)
        # drop
        list_drop_score_times_max = [0, 0, 0]
        node_temp = node
        drop_index = 0

        if node_temp.can_drop==0:
            node_temp.can_drop=1
            for i in range(0,len(node_temp.temp_player_list)):
                print("i",i)
                node_temp.temp_player_sum = node_temp.temp_player_sum - int(node_temp.temp_player_list[i][1])
                list_drop_score_times,chose_action,drop_index=self.max_expected_tree(node_temp,deep+1)
                print("drop " + str(deep))
                print(list_drop_score_times,chose_action)
                node_temp.temp_player_sum = node_temp.temp_player_sum + int(node_temp.temp_player_list[i][1])
                if list_drop_score_times[0] >list_drop_score_times_max[0]:
                    list_drop_score_times_max = list_hit_score_times
                    drop_index=i
            node_temp.can_drop = 0
            print("drop " + str(deep))
            print(list_drop_score_times_max)
            print()
        return self.max_return(list_stay_score_times,list_hit_score_times,list_drop_score_times_max,drop_index)

    def max_return(self,list_stay_score_times,list_hit_score_times,list_drop_score_times,drop_index):
        #print("list_stay_score_times:", list_stay_score_times)
        #print("list_hit_score_times:", list_hit_score_times)
        #print("list_drop_score_times:", list_drop_score_times)
        win_pro_stay = list_stay_score_times[0]/sum(list_stay_score_times)
        win_pro_hit = list_hit_score_times[0] / sum(list_hit_score_times)
        if(sum(list_drop_score_times) == 0):
            win_pro_drop=0
        else:
            win_pro_drop = list_drop_score_times[0] / sum(list_drop_score_times)
        if win_pro_stay>=win_pro_drop and win_pro_stay>=win_pro_hit:
            return list_stay_score_times,"stay", int(drop_index)
        if win_pro_hit>=win_pro_drop and  win_pro_hit>=win_pro_stay:
            return list_hit_score_times,"hit",int(drop_index)
        if win_pro_drop>=win_pro_hit and win_pro_drop>=win_pro_stay:
            return list_drop_score_times,"drop",int(drop_index)



def __main__():
    game = matrix_modified.game_class(1)  #size of the cards
    game.start_one_round()
    game.show_game_condition_for_test()

    algorithm = algo_class(game)

    '''
    while(1):
        game.start_one_round()
        game.show_game_condition_for_play()
        game.input_action()
        game.end_one_round()
        val=input("Do you want to start one new around(0 is yes; 1 is no)")
        if(val=="1"):break
    '''
if __name__ == "__main__":
    __main__()
