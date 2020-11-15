import numpy as np
import random
import algo
import time
import matrix_modified
import os


def AI_play():

    print("Hello I'm Black Jack AI ")
    while(1):
        size = int(input("Please input the size you want(must be int and must more than zero): "))
        if size>0:
            break

    game = matrix_modified.game_class(size)  #size of the cards

    model = int(input("Please input the model of AI (0 is expected tree AI; 1 is random AI): "))
    if model==0:
        while(1):

            game.start_one_round()
            time.sleep(1)
            game.show_game_condition_for_play()
            can_drop=0
            time.sleep(1)
            os.system("pause")
            print()
            print()
            print()
            while(1):
                algorithm = algo.algo_class(game)
                action,drop_action,nodes_number=algorithm.AI_chose_node(can_drop)
                print()
                end,win = game.ai_input_action(action,drop_action)
                if action=="drop":
                    can_drop=1
                if end==1:
                    break
                os.system("pause")
                print()
                print()
                print()

            game.end_one_round()
            val = input("Do you want AI start one new around(0 is yes; 1 is no)")
            if(val=="1"):break
    if model ==1 :
        while(1):

            game.start_one_round()
            time.sleep(1)
            game.show_game_condition_for_play()
            can_drop=0
            time.sleep(1)
            os.system("pause")
            print()
            print()
            print()
            while(1):
                ran_01 = random.randint(0, 2)
                end = 0
                #print(ran_01)
                if ran_01==0:
                    print("Chose stay!!!")
                    game.ai_input_action("stay",0)
                    break
                if ran_01==1:
                    print("Chose hit!!!")
                    end,win = game.ai_input_action("hit", 0)
                if ran_01 == 2 and can_drop == 1:
                    continue
                if ran_01==2 and can_drop==0:
                    print("Chose drop!!!")
                    can_drop = 1
                    ran_02 = random.randint(0, len(game.list_player_cards)-1)
                    end,win = game.ai_input_action("drop", ran_02)

                if end==1:
                    break
                os.system("pause")
                print()
                print()
                print()

            game.end_one_round()
            val = input("Do you want AI start one new around(0 is yes; 1 is no)")
            if(val=="1"):break


def __main__():
    AI_play()

if __name__ == "__main__":
    __main__()
