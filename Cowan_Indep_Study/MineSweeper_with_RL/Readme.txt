Minesweeper with RL
Readme Updated: 10/5/21

Last Commit before Update: 
	Network Update Results on 10/3 (commit 47d28f0)

General Flow:
Minesweeper_Player_Input.py is just set up to allow human to play game and get average score, ignore for now.

1) Minesweeper_Testing_Ground.py
	The primary driver of the code.
	
	The giant block of numbers at the beginning of the code covers everything you might want
	to change about the network/boards used EXCEPT The physical structure of the network layers.
	
2) Minesweeper_Network_Construction.py
	Creation of the network
	Physically defines the layers of the network, would manually need to edit this file 
	to change layer structure, but not layer hyperparameters.
	
3) Minesweeper_TF_Functions.py
	All the update functions for Q Learning, state choice, and how to create and 
	use the replay buffer to generate training data
	
4) Minesweeper_Base_Game.py
	Make the minesweeper board, print out the board for humans to see where the game 
	currently is, and find out how well a single game was played after it finished
	
	Also, define the reward values for the spots on the board (this function will be 
	changing depending on how we manipulate the use of rewards in the future)
