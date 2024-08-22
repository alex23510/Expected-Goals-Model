﻿# Expected-Goals-Model
This repo contians two notebooks and a report, the first notebook Expect Goals v4 is the latest version in creating an expected goals model with real data. The notebook compares popular machine learning models to see their pitfalls and strengths, using poisson statistics. Then the winning model is fine tuned and calibrated to create an XG model, with the parameters distance, angle, no of intervening opponents, no of intervening team mates and pressure on the shot, aswell as header or foot shot variable. 

The second notebook is this model applied to a game scenario, using a real anoynimised tracking and event data from a real match. The tracking data is too large to upload but can be found here.https://github.com/metrica-sports/sample-data

The model takes in tracking data and event data and uses that to estimate the probability that each shot for each team will result in a goal. It then creates a binomial probability model to estimate given the quality of shots, how likley is a winning scoreline for each team at any point in the game.
Given the development of this binomial model, it then creates a strategey model, the optimises strategey based on the game state and predicted outcome, using binomial statistics. 
There is a match report attached which details both the creation of the model and its application to a game. 
