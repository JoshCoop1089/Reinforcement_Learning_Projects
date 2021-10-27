Minesweeper with RL
Readme Updated: 10/27/21

Last Commit before Update: 
	Q_Network fed with Mine Predictor, and Actor-Critic network Complete on 10/27 (commit c1b3c04)

General Flow:

1) Use_Mine_Predictions_As_Training.py
	Creates a mine location prediction network to use to generate next locations, which are then fed
	into the Q network to help choose next states. Q network trains on a decaying epsilon greedy policy,
	where it decreases the epislon random factor by 5% every time it reaches a loss plateau during training.
	
2) Actor_Critic_Network.py
	Utilizes a pair of the same network to work through training via decaying epsilon greedy policy. 
	
	Actor Network is updated by Critic network after a full run through a decaying epsilon policy 
		(ie the run start pure random, and ends pure greedy, then critic weights get copied to actor and it begins again)
	
3) Minesweeper_Base_Game.py
	Make the minesweeper board, print out the board for humans to see where the game 
	currently is, and find out how well a single game was played after it finished
	
	Also, define the reward values for the spots on the board (this function will be 
	changing depending on how we manipulate the use of rewards in the future)
	
4) Multi_Network files
	Used to set up network parameter comparisons to test effectiveness of l2 regularization, dropout layers, 
	dense layer sizes numbers of dense layers in a contained file.
	4a) Multi_Network_Tests -- allows changing all three parameters without much granularity
	4b) Multi_Network_Dropout -- allows for changing only dropout coefficients, no l2 regularizations
	4c) Multi_Network_L2_Vals -- allows for changing only l2 regularizations, no dropout
