# BlackJack_AI
black jack ai work

Operating environment：python3

files:
  .py:
  
    (1) running files
    
    The game_player.py is to play by the user.
    
    The game_ai.py is to play by the AI use the tree-base algorithm.
    
    The game_ai_random.py is to play by the random format.
    
    (2) class files
    
    The modified_matrix.py has include the game class and cards class, which include the information and function of the game.

    The algo.py is about the algorithm which use the minimaxexpect tree algorithm.

    The test_for_algo.py is test about the dealer function of the algo.
    
    (3) test files
    
    The test_for_algo.py is test about the dealer function of the algo.
    
    The test_for_tree.py is test about the tree algorithm.
    
    The test_for_tree_2.py is test about the tree algorithm.
    

cards:

  In the modified_matrix.py ; cards_class:
  
    (0) __init__ (size)  size is to define the size of the matrix, how many desks of cards in the cards.
       In this calss will create one matrix(size*52*5) 
       # size * 52 * 5 pokers; 4 means how many desks; 52 means cards number; 5 means states
       # if want to change the size you can just chang the first value, the other two is static
       # 0 1 2 3 4 desk,player,dealer_show,dealer_hide,drop   1 exist 0 not exist

Play round:

   In the modified_matrix.py ; game_class:
   
    (0) __init__ (size)  size is to define the size of the matrix, how many desks of cards in the game.
  
    (1)start_one_round() means to start one new round of BlackJack, the cards will be same.
    
    (2)show_game_condition_for_test() is to show the condition of the game.
    
    (3)show_game_condition_for_play() is to show the condition of the game.
    
    (4)input_action() is to input the actions 0 is stay; 1 is hit; 3 is drop the first card; 4 is drop the second card. 
    
    (6)end_one_round() is to end one round and re-calculate all values.
    
    (7)dealer_auto() is to calculate the dealer_sum/list value function.
    
    (8)player_auto() is to calculate the player_sum/list value function.
    
    (9)judge_winner() is to judge which one is winner for this round of game.
    
    (10)recalculate_player_list_and_sum() is to re-calculate after drop action.
    
    (11)show_all_matrix_in_hands() is to show the values in player or dealer's hands, use for test.
    
Algo  
  In the algo_class:
  
    (1)get_best_action()  is to tell you the most winnable actions.
  
    (2)minmax()   is to build the tree to calculate each action final results(win,lost,tie), and help to get best action.
