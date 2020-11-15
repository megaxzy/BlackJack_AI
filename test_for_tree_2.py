import numpy as np
import random
import matrix_modified
import algo as algori


def __main__():
    game = matrix_modified.game_class(1)  #size of the cards
    game.start_one_round()
    game.show_game_condition_for_test()


    algo = algori.algo_class(game)
    algo.AI_init_node()

    '''
    while (1):
        game.start_one_round()
        game.show_game_condition_for_play()
        game.input_action()
        game.end_one_round()
        val = input("Do you want to start one new around(0 is yes; 1 is no)")
        if (val == "1"): break
    '''

if __name__ == "__main__":
    __main__()
