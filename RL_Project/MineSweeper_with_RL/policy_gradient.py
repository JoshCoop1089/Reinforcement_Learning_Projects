# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 16:10:40 2021

@author: joshc
"""
def generate_single_game(network, board_specifics):
    
    # How are we generating state/action choices to build the list of changes?
    # If we're using the network to predict probs in the main loop, what's generating the action choice in here?
    
    # Use np.random.choice and pass in a flattened version of the network output to choose next location 
    # stochastically based on probs of squares
    
    return state_list, action_list, reward_list, expected_returns


def apply_policy_grad_to_network(network, num_games, board_specifics):
    
    with tf.GradientTape as tape:
        
        # How many games are needed to get a good result? Can too many cause overfit problems?
        for x in range(num_games):
            
            #Generate training data for single game (each list is N items long, where N = board_dimension**2)
            state_list, action_list, expected_returns_per_state = generate_single_game(network, board_specifics)
        
            # Use the current network to predict the probs across the whole board for a single state
            # output_prob would be a list of all N full board predictions for all N states
            output_prob = network(state_list)
            
            # Turn output_prob into matrix of 0's everywhere except for chosen action in that state
            single_state_prob = output_prob * action_list
            
            # The inner log summation term of eq15, negated due to tape.gradient method being designed for grad descent?
            loss += -1 * tf.log(reduce_sum(single_state_prob)) * expected_returns_per_state
        
        # Average the loss over all games to estimate expected value
        loss /= num_games
        
        # PDF Eq3, update network weights
        # Check documentation re: tape.gradient targets and sources
        tape.gradient(network.weights, loss)
        
        # Applying the grad to the network
        optimizer.apply_gradients(tape, network.weights)
    
    # Network has now been updated from one batch of games
    return network
            