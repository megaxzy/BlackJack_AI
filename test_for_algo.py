
import numpy as np
import random


def turn_to_value(value):
    card = value % 13 + 1
    if card > 10:
        card = 10
    return int(card)
def dealer_turn(dealer_sum, player_sum, temp_cards_matrix, list_score_times,s):
    print("s:",str(s))
    if dealer_sum>= 22 :
        print(dealer_sum, " ", player_sum)
        list_score_times[0] = list_score_times[0] + 100/s  # win
        return list_score_times
    if dealer_sum >= 17:
        print(dealer_sum, " ", player_sum)
        if dealer_sum == player_sum:
            list_score_times[2]= list_score_times[2]+ 100/s   # tie
        if dealer_sum > player_sum:
            list_score_times[1] = list_score_times[1] + 100/s  # lose
        if dealer_sum < player_sum:
            list_score_times[0] = list_score_times[0] +100/s   # win
        return list_score_times
    temp_temp_cards_matrix = temp_cards_matrix.copy()
    for i in range(0, 1):
        sum=0
        for j in range(0, 13):
            if temp_cards_matrix[i][j][0] == 1:
                sum=sum+1
        for j in range(0, 13):
            if temp_cards_matrix[i][j][0] == 1:
                print(j+1)
                temp_cards_matrix[i][j][0] = 0
                list_score_times = dealer_turn(dealer_sum + turn_to_value(j),player_sum,temp_cards_matrix, list_score_times, s*sum)
                temp_cards_matrix[i][j][0] = 1
    return list_score_times

def __main__():
    matrix = np.zeros((1, 52, 5))
    matrix[0:1, 0:52, 0:1] = 1  # 0 1 2 3 4 desk,player,dealer_show,dealer_hide,drop   1 exist 0 not exist
    # test
    dealer_01=7
    dealer_02=8
    player_01=0
    player_02=0
    sum_dealer=dealer_01+dealer_02
    sum_player=player_01+player_02
    #matrix[0][dealer_01-1][0] = 0
    #matrix[0][dealer_02-1][0] = 0
    #matrix[0][player_01-1][0] = 0
    #matrix[0][player_02-1][0] = 0
    list_score=dealer_turn(sum_dealer,sum_player,matrix,[0,0,0],1)
    print (list_score)
    print((list_score[0]+list_score[1]))
    print(list_score[0])
    print (list_score[0]/(list_score[0]+list_score[1]))

if __name__ == "__main__":
    __main__()
