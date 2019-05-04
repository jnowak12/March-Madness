# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model

#read in spreadsheets containing various basketball statistics
#data contains data for models to train on
#data2 contains data for this year's NCAA tournament teams
data = pd.read_csv('/Users/joenowak/Documents/NCAA Data.csv')
data2 = pd.read_csv('/Users/joenowak/Documents/March Madness.csv')

#X is the training data - only most relevant variables - based on my (poor) judgement
#X2 is training data that contains all variables
#all data is regular season data for teams
#training data is composed of all the teams that were in 2014, 2015, 2016, and 2017 NCAA tournaments
X = data[['Seed', 'SRS', 'SOS']].values
X2 = data[['Seed', 'SRS', 'SOS', 'W-L%', 'Conf W-L%', 'Pace', 'ORtg', 'FTr', '3PAr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA']].values

#Y1 is an indicator of success in tournament in which teams are first sorted by number of wins in tournament, ties broken by average +/- in tournament games 
#For example, tournament champion 'sort' value is 64; first round loser with worst +/- is given 'sort' value of 1.
#Y2 is a composite indicator with formula = average +/- in tournament games + (3 * number of tournamnent wins)
Y1 = data[['Sort']].values
Y2 = data[['Composite']].values

#using regular season data from this year's tournamnent teams to try and predict their outcomes
Xpred = data2[['Seed', 'SRS', 'SOS']].values
Xpred2 = data2[['Seed', 'SRS', 'SOS', 'W-L%', 'Conf W-L%', 'Pace', 'ORtg', 'FTr', '3PAr', '3PAr', 'TS%', 'TRB%', 'AST%', 'STL%', 'BLK%', 'eFG%', 'TOV%', 'ORB%', 'FT/FGA']].values

#model.fit trains the model on prior data
#model.predict uses regression equation to predict indicator values for this year's teams
#modeltype_array contains indicator values for this year's 64 teams
#set normalize = True to normalize all my data

#Ridge regression using 'sort' indicator and condensed data
#alpha represents regularization strength - "improves the conditioning of the problem and reduces the variance of the estimates" according to Scikit Learn
#chose 1.0 arbitrarily
ridge_model1 = linear_model.Ridge(alpha=1.0, normalize=True)
ridge_model1.fit(X,Y1)
ridge_array1 = ridge_model1.predict(Xpred)

#Ridge regression using 'composite' indicator and condensed data
ridge_model2 = linear_model.Ridge(alpha=1.0, normalize=True)
ridge_model2.fit(X,Y2)
ridge_array2 = ridge_model2.predict(Xpred)

#Ridge regression using 'sort' indicator and full data
ridge_model3 = linear_model.Ridge(alpha=1.0, normalize=True)
ridge_model3.fit(X2,Y1)
ridge_array3 = ridge_model3.predict(Xpred2)

#Ridge regression using 'composite' indicator and full data
ridge_model4 = linear_model.Ridge(alpha=1.0, normalize=True)
ridge_model4.fit(X2,Y2)
ridge_array4 = ridge_model4.predict(Xpred2)


#Bayesian regression using 'sort' indicator and condensed data
bayesian_model1 = linear_model.BayesianRidge(normalize=True)
bayesian_model1.fit(X,np.ravel(Y1))
bayesian_array1 = bayesian_model1.predict(Xpred)

#Bayesian regression using 'composite' indicator and condensed data
bayesian_model2 = linear_model.BayesianRidge(normalize=True)
bayesian_model2.fit(X,np.ravel(Y2))
bayesian_array2 = bayesian_model2.predict(Xpred)

#Bayesian regression using 'sort' indicator and full data
bayesian_model3 = linear_model.BayesianRidge(normalize=True)
bayesian_model3.fit(X2,np.ravel(Y1))
bayesian_array3 = bayesian_model3.predict(Xpred2)

#Bayesian regression using 'composite' indicator and full data
bayesian_model4 = linear_model.BayesianRidge(normalize=True)
bayesian_model4.fit(X2,np.ravel(Y1))
bayesian_array4 = bayesian_model4.predict(Xpred2)


#LARS Lasso regression using 'sort' indicator and condensed data
#max_iter is the maximum number of iterations for the regression
#initially did not set this parameter but receiverd convergence warning and subsequently decreased max_iter until no warning
LassoLars1 = linear_model.LassoLars(alpha=.1, max_iter=3, normalize=True)
LassoLars1.fit(X, Y1)
LassoLars1 = LassoLars1.predict(Xpred)

#LARS Lasso regression using 'composite' indicator and condensed data
LassoLars2 = linear_model.LassoLars(alpha=.1, max_iter=3, normalize=True)
LassoLars2.fit(X, Y2)
LassoLars2 = LassoLars2.predict(Xpred)

#LARS Lasso regression using 'sort' indicator and full data
LassoLars3 = linear_model.LassoLars(alpha=.1, max_iter=3, normalize=True)
LassoLars3.fit(X2, Y1)
LassoLars3 = LassoLars3.predict(Xpred2)

#LARS Lasso regression using 'composite' indicator and full data
LassoLars4 = linear_model.LassoLars(alpha=.1, max_iter=3, normalize=True)
LassoLars4.fit(X2, Y2)
LassoLars4 = LassoLars4.predict(Xpred2)
      
#creating empty dictionaries  
predictor_dict1 = {}
predictor_dict2 = {}
predictor_dict3 = {}
predictor_dict4 = {}
predictor_dict5 = {}
predictor_dict6 = {}
predictor_dict7 = {}
predictor_dict8 = {}
predictor_dict9 = {}
predictor_dict10 = {}
predictor_dict11 = {}
predictor_dict12 = {}

#column of school names
team_names = data2[['School']].values

#iterates through team names (keys) and assigns each the corresponding indicator value, filling the dictionaries for each trial
#.replace was needed because there was a space after each team name in the spreadsheet which python was interpreting as "u'\\xao'"
#ridge regression outputs values in arrays, so [0] takes the first (and only) value in each array, same thing with team_names
for i in range(len(team_names)):
  team_name = str(team_names[i][0]).replace(u'\xa0', '')
  predictor_dict1[team_name] = ridge_array1[i][0]
  predictor_dict2[team_name] = ridge_array2[i][0]
  predictor_dict3[team_name] = ridge_array3[i][0]
  predictor_dict4[team_name] = ridge_array4[i][0]
  predictor_dict5[team_name] = bayesian_array1[i]
  predictor_dict6[team_name] = bayesian_array2[i]
  predictor_dict7[team_name] = bayesian_array3[i]
  predictor_dict8[team_name] = bayesian_array4[i]
  predictor_dict9[team_name] = LassoLars1[i]
  predictor_dict10[team_name] = LassoLars2[i]
  predictor_dict11[team_name] = LassoLars3[i]
  predictor_dict12[team_name] = LassoLars4[i]

#list of dictionaries so I can iterate through each to determine score
Dict_List = [predictor_dict1, predictor_dict2, predictor_dict3, predictor_dict4, predictor_dict5, predictor_dict6, predictor_dict7, predictor_dict8, predictor_dict9, predictor_dict10, predictor_dict11, predictor_dict12]

#list of teams in the tournament in order of how they appear in bracket - first team plays second, third plays fourth, etc.
#no, I didn't type them all by hand, I used a for loop to go through team_names and then copy and pasted the output
#yes, I probably should've kept the for loop in here instead deleting it after I copy and pasted the output
round64_teams = ['Duke','North Dakota State','Virginia Commonwealth','Central Florida','Mississippi State','Liberty','Virginia Tech','Saint Louis','Maryland','Belmont','Louisiana State',
                 'Yale','Louisville','Minnesota','Michigan State','Bradley','Gonzaga','Fairleigh Dickinson','Syracuse','Baylor','Marquette','Murray State','Florida State',
                 'Vermont','Buffalo','Arizona State','Texas Tech','Northern Kentucky','Nevada','Florida','Michigan','Montana','Virginia','Gardner-Webb','Mississippi','Oklahoma',
                 'Wisconsin','Oregon','Kansas State','UC-Irvine','Villanova',"Saint Mary's (CA)",'Purdue','Old Dominion','Cincinnati','Iowa','Tennessee','Colgate','North Carolina',
                 'Iona','Utah State','Washington','Auburn','New Mexico State','Kansas','Northeastern','Iowa State','Ohio State','Houston','Georgia State','Wofford','Seton Hall',
                 'Kentucky','Abilene Christian',]

#iterating through all the dictionaries
for h in range(12):
  
  #creating empty lists that will contain the winners of each round, don't need to do this for winner, which will just be a string
  round32_teams = []
  round16_teams = []
  round8_teams = []
  round4_teams = []
  round2_teams = []
  
  #bunch of indices 
  i = 0
  j = 0
  k = 0
  l = 0
  m = 0
  n = 0
  
  #compares the indicator scores for the teams in each matchup with the winner getting added the next round's list
  #keeps doing this until we get a winner
  while i < 64:
    if  Dict_List[h][round64_teams[i]] > Dict_List[h][round64_teams[i+1]]:
      round32_teams.append(round64_teams[i])
      i+=2
    else:
      round32_teams.append(round64_teams[i+1])
      i+=2
  while j < 32:
    if  Dict_List[h][round32_teams[j]] > Dict_List[h][round32_teams[+1]]:
      round16_teams.append(round32_teams[j])
      j+=2
    else:
      round16_teams.append(round32_teams[j+1])
      j+=2
  while k < 16:
    if  Dict_List[h][round16_teams[k]] > Dict_List[h][round16_teams[k+1]]:
      round8_teams.append(round16_teams[k])
      k+=2
    else:
      round8_teams.append(round16_teams[k+1])
      k+=2
  while l < 8:
    if  Dict_List[h][round8_teams[l]] > Dict_List[h][round8_teams[l+1]]:
      round4_teams.append(round8_teams[l])
      l+=2
    else:
      round4_teams.append(round8_teams[l+1])
      l+=2
  while m < 4:
    if  Dict_List[h][round4_teams[m]] > Dict_List[h][round4_teams[m+1]]:
      round2_teams.append(round4_teams[m])
      m+=2
    else:
      round2_teams.append(round4_teams[m+1])
      m+=2
  while n < 2:
    if  Dict_List[h][round64_teams[n]] > Dict_List[h][round64_teams[n+1]]:
      winner = round2_teams[n]
      n+=2
    else:
      winner = round2_teams[n+1]
      n+=2

  #gotta get the whole alphabet involved
  o = 0
  p = 0
  q = 0
  r = 0
  s = 0
  
  #initializing variable to keep the model's score
  score = 0

  #these are the teams that actually made it to each round so that we can compare them to the models' predictions
  correct_round32_teams = ['Duke', 'Central Florida', 'Liberty', 'Virginia Tech', 'Maryland', 'Louisiana State', 'Minnesota', 
                           'Michigan State', 'Gonzaga', 'Baylor', 'Murray State', 'Florida State', 'Buffalo', 'Texas Tech', 'Florida', 
                           'Michigan', 'Virginia', 'Oklahoma', 'Oregon', 'UC-Irvine', 'Villanova', 'Purdue', 'Iowa', 'Tennessee', 
                           'North Carolina', 'Washington', 'Auburn', 'Kansas', 'Ohio State', 'Houston', 'Wofford', 'Kentucky']
  correct_round16_teams = ['Duke', 'Mississippi State', 'Louisiana State', 'Michigan State', 'Gonzaga', 'Florida State', 'Texas Tech', 
                           'Michigan', 'Virginia', 'Oregon', 'Purdue', 'Tennessee', 'North Carolina', 'Auburn', 'Houston', 'Kentucky']
  correct_round8_teams = ['Duke', 'Michigan State', 'Gonzaga', 'Texas Tech', 'Virginia', 'Purdue', 'Auburn', 'Kentucky'] 
  correct_round4_teams = ['Michigan State', 'Texas Tech', 'Virginia', 'Auburn',] 
  correct_round2_teams = ['Texas Tech', 'Virginia']
  correct_winner = 'Virginia'

  #if the teams in the predicted and actual lists match, we add to the score
  #used ESPN's Tournament Challenge scoring system
  for o in range(32):
    if round32_teams[o] == correct_round32_teams[o]:
      score += 10
  for p in range(16):
    if round16_teams[p] == correct_round16_teams[p]:
      score += 20
  for q in range(8):
    if round8_teams[q] == correct_round8_teams[q]:
      score += 40
  for r in range(4):
    if round4_teams[r] == correct_round4_teams[r]:
      score += 80
  for s in range(2):
    if round2_teams[s] == correct_round2_teams[s]:
      score += 160
  if winner == correct_winner:
    score += 320
  
  #print out the scores for each model
  print('Trial', h+1, 'score:  ', score, 'points')