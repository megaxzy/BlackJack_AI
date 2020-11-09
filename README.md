# BlackJack_AI
black jack ai work

Operating environment：python3

files:
  .py:
    
    The matrix.py has include the calss and main function which is runnable and can play the game.

    The algo.py is about the minimax algo(//TODO) 

    The test_for_algo.py is about the algorithm models tests.

cards:

  In the cards_class:

    (1)In this calss will create one matrix(4*52*5) 
       # 4 * 52 * 5 pokers; 4 means how many desks; 52 means cards number; 5 means states
       # if want to change the size you can just chang the first value, the other two is static
       # 0 1 2 3 4 desk,player,dealer_show,dealer_hide,drop   1 exist 0 not exist

Play round:

  In the game_class:
  
    (1)start_one_round() means to start one new round of BlackJack, the cards will be same.
    
    (2)show_game_condition_for_play() is to show the condition of the game.
    
    (3)input_action() is to input the actions 0 is stay; 1 is hit; 3 is drop the first card; 4 is drop the second card. 
    
Algo  //TODO:  
  In the algo_class:
  
    (1)get_best_action()  is to tell you the most winnable actions.
  
    (2)minmax()   is to build the tree to calculate each action final results(win,lost,tie), and help to get best action.
