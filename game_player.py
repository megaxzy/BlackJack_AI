import numpy as np
import random
import algo
import matrix_modified


def __main__():

    print("")
    while(1):
        size = int(input("Please input the size you want(int and must more than zero): "))
        if size>0:
            break
    game = matrix_modified.game_class(size)  #size of the cards

    while(1):
        game.start_one_round()
        #game.show_all_matrix_in_hands()
        game.show_game_condition_for_play()
        game.input_action()
        game.end_one_round()
        val = input("Do you want to start one new around(0 is yes; 1 is no)")
        if(val=="1"):break

if __name__ == "__main__":
    __main__()
