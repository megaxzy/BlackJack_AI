# BlackJack_AI
black jack ai work

Operating environment：python3

Rules:
    
    (1) stay action: player end this round.
    
    (2) hit action: player get one more card.
    
    (3) drop action: player chose one card and drop this.
    
    (4) player can only drop once.
    
    (5) if player's cards' sum more than 21, then player lose.
    
    (6) if player chose stay action, then turn to the dealer's round.
    
    (7) if dealer's cards' sum more than 21 then player win.
    
    (8) else judge which one's cards' sum is higher, judge winner, or tie.
    
    
files:
  .py:
  
    (1) running files
    
    The game_player.py is to play by the user.
    
    The game_ai.py is to play by the AI use the tree-base algorithm and random algorithm.
    
    (2) class files
    
    The modified_matrix.py has include the game class and cards class, which include the information and function of the game.

    The algo.py is about the algorithm which use the minimaxexpect tree algorithm.

    The test_for_algo.py is test about the dealer function of the algo.
    
    (3) test files
    
    The test_for_algo.py is test about the dealer function of the algo.
    
    The test_for_tree.py is test about the tree algorithm.
    
    The test_for_tree_2.py is test about the tree algorithm.
    
    (4) efficency test
    
    algo_no_print.py is one format of algo.py.
    
    algorithm_efficency.py is to test for the algorithm efficency.
    

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
    
    (12)ai_input_action()  is to auto handle the ai actions.
    
Algo  
  In the algo_class:
  
    (0)__init__(game)  you need to input the game_class object when you init the algo class.
  
    (1)dealer_turn()  is to calculate the dealer turn autoly.
  
    (2)AI_chose_node()   is to calculate the best action in this round (stay,hit,drop).
    
    (3)max_expected_tree()  is the expected tree function.
    
    (4)max_return()  is to calculate which one is the best for this action.
