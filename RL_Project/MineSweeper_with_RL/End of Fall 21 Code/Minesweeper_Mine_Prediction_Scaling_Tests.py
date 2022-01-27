# -*- coding: utf-8 -*-
"""
Created on Sat Dec 18 15:54:16 2021

@author: joshc
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import MineSweeper_Base_Game as ms
from Use_Mine_Predictions_As_Training_Limited_Epsilon import play_one_game_mine_predict_network
from Use_Mine_Predictions_As_Training_Limited_Epsilon import play_one_game_random_choice_baseline
from Use_Mine_Predictions_As_Training_Limited_Epsilon import create_and_train_mine_predictor


################################################################
# Board Specifics
dimension = 10
mine_percent = 0.4
dropout_coef = 0.44

# # -----> Testing Outputs Quickly <----- #
# # Training Specifics 
# num_games_mine_train = 1
# num_training_rounds = 2
    
# # Evaluation Specifics
# num_games = 1


# -----> Actual Data Collection Values <-----  #
# Training Specifics 
num_games_mine_train = 2000
num_training_rounds = 10
    
# Evaluation Specifics
num_games = 1000

##############################################################
dense_size = 80*(dimension**2)
scale = [0.5, 1, 1.5]
mine_percent_list = [f"{int(scale[0]*100*mine_percent)}% Mines", f"{int(scale[1]*100*mine_percent)}% Mines", f"{int(scale[2]*100*mine_percent)}% Mines"]
network_style = [f"{int(scale[0]*num_training_rounds)} Training Runs", f"{int(scale[1]*num_training_rounds)} Training Runs", f"{int(scale[2]*num_training_rounds)} Training Runs"]
input_shape = (dimension, dimension, 11)

def freq_dict(inputs):
    freq_dic = {}
    for val in inputs:
        freq_dic[val] = freq_dic.get(val,0)+1
    return freq_dic

def distinct_vals_b_minus_a(a, b):
    a_freq = freq_dict(a)
    b_freq = freq_dict(b)
    
    abkeys = set([*a_freq, *b_freq])
    # print(abkeys)
    bmina = {}
    for key in abkeys:
        if key in b_freq.keys() and key in a_freq.keys():
            bmina[key] = b_freq[key]-a_freq[key]
        elif key in b_freq.keys():
            bmina[key] = b_freq[key]
        elif key in a_freq.keys():
            bmina[key] = -1*a_freq[key]
            
    a_out = {}
    b_out = {}
    for key in bmina:
        if bmina[key] > 0:
            b_out[key] = bmina[key]
        elif bmina[key] < 0:
            a_out[key] = -1*bmina[key]
            
    return a_out, b_out

def get_network_play_results (num_games, dimension, num_mines, mine_prediction_model, version_name, network_type):
    avg_clicks = []    
    vals = [x*num_games//10 for x in range (11)]
    for x in range(1,num_games+1):
        if x in vals:
            print(f"Starting Game: {x} out of {num_games} for network: {version_name} w/ {network_type}")
        avg_clicks.append(play_one_game_mine_predict_network(dimension, num_mines, mine_prediction_model))
    
    avg_score = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks]
    return avg_score   

# So many graphs to set up
fig, ax = plt.subplots(3,3, sharex='col', sharey = 'row', figsize=(10,10)) 
fig.subplots_adjust(hspace=0.4, wspace=0.4)
fig1, ax1 = plt.subplots(3,3, sharex='col', sharey = 'row', figsize=(10,10)) 
fig1.subplots_adjust(hspace=0.4, wspace=0.5)
fig3, ax3 = plt.subplots(3,3, figsize=(10,10))   
fig3.subplots_adjust(hspace=0.5, wspace=0.5) 

for axe, col in zip(ax[0], network_style):
    axe.set_title(col)

for axe, row in zip(ax[:,0], mine_percent_list):
    axe.set_ylabel(row, rotation=90, size='large')

for axe, col in zip(ax1[0], network_style):
    axe.set_title(col)

for axe, row in zip(ax1[:,0], mine_percent_list):
    axe.set_ylabel(row, rotation=90, size='large')
    
for axe, col in zip(ax3[0], network_style):
    axe.set_title(col)

for axe, row in zip(ax3[:,0], mine_percent_list):
    axe.set_ylabel(row, rotation=90, size='large')
    
fig.suptitle(f"Unique Scores from Random and Trained for {num_games} games" + 
              "\n"+ "- "*20 + "\nBlue: Random, Orange: Trained")
fig1.suptitle(f"Score Distributions for {num_games} games\nBlue: Random, Orange: Trained")
fig3.suptitle("Score Breakdowns (Mean, Median)\nRa -> Random, Tr -> Mine Predictor")
fig.tight_layout()  
fig1.tight_layout()  
fig3.tight_layout()  

# Iterate over the models
for i in range(3):
    mine_percent_t = mine_percent*scale[i]
    num_mines = int(mine_percent_t * (dimension**2))
    
    # Set up the random baseline
    avg_clicks_random = []    
    vals = [x*num_games//10 for x in range (11)]
    for x in range(1,num_games+1):
        if x in vals:
            print(f"Starting Game: {x} out of {num_games} for network: Random Play")
        avg_clicks_random.append(play_one_game_random_choice_baseline(dimension, num_mines))
    avg_score_random = [100*ms.optimal_play_percent(dimension, num_mines, score) for score in avg_clicks_random]
    
    for j in range(3):
        version_name = mine_percent_list[i]
        network_type = network_style[j]
        print("\n\n ---> Starting New Network Type <---\n\t", version_name, network_type)
        
        num_training_rounds_t = int(num_training_rounds*scale[j])       
        mine_network_training_params = num_games_mine_train, num_training_rounds_t, dropout_coef, dense_size
        
        model = create_and_train_mine_predictor(dimension, num_mines, mine_network_training_params)
                
        # Use trained model to play game
        trained_results = get_network_play_results(num_games, dimension, num_mines, model, version_name, network_type)
        

        # MAKE THE UBERGRAPH UNLEASH THE KRAKEN
        display_data = []
        display_data.append(avg_score_random)
        display_data.append(trained_results)
        
        # Unique Score Results!
        a_out, b_out = distinct_vals_b_minus_a(avg_score_random, trained_results)      
        ax[i,j].bar(a_out.keys(), a_out.values())
        ax[i,j].bar(b_out.keys(), b_out.values())
        
        # KDE's! But Double! And Vertical! Called Violins! With Quartiles!
        sns.violinplot(data = display_data, legend = False, ax = ax1[i,j], inner = 'quartile')
        
        # Numbers!
        results = str(
            f"Ra:{round(np.mean(avg_score_random), 1)} || Med:{round(np.median(avg_score_random), 1)}" + 
            f"\nTr: {round(np.mean(trained_results), 1)} || Med:{round(np.median(trained_results), 1)}")
        ax3[i, j].text(0.5, 0.45, results,
                      fontsize=14, ha='center')
        
# fig.savefig('scorediff.png')
# fig1.savefig('violins.png')
# fig2.savefig('loss.png')
# fig3.savefig('meanvals.png')
