
import numpy as np
import random


def dealer_turn(dealer_sum, player_sum, temp_cards_matrix, list_score_times):
    if dealer_sum>= 22 :
        list_score_times[0] = list_score_times[0] + 1  # win
        return list_score_times
    if dealer_sum >= 16:
        if dealer_sum == player_sum:
            list_score_times[2]= list_score_times[2]+1  # tie
        if dealer_sum > player_sum:
            list_score_times[1] = list_score_times[1] + 1  # lose
        if dealer_sum < player_sum:
            list_score_times[0] = list_score_times[0] + 1  # win
        return list_score_times
    temp_temp_cards_matrix = temp_cards_matrix.copy()
    for i in range(0, 1):
        for j in range(0, 52):
            if temp_cards_matrix[i][j][0] == 1:
                print(j)
                temp_cards_matrix[i][j][0] = 0
                list_score_times=dealer_turn(dealer_sum + j % 13 + 1,player_sum,temp_cards_matrix, list_score_times)
                temp_cards_matrix = temp_temp_cards_matrix
    return list_score_times

def __main__():
    matrix = np.zeros((1, 52, 5))
    matrix[0:1, 0:52, 0:1] = 1  # 0 1 2 3 4 desk,player,dealer_show,dealer_hide,drop   1 exist 0 not exist
    # test
    dealer_01=1
    dealer_02=4
    player_01=8
    player_02=3
    matrix[0][dealer_01][0] = 0
    matrix[0][dealer_02][0] = 0
    matrix[0][player_01][0] = 0
    matrix[0][player_02][0] = 0
    print (dealer_turn(5,11,matrix,[0,0,0]))

if __name__ == "__main__":
    __main__()
