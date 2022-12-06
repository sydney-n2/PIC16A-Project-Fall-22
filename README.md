# PIC16A Project Fall 22

Project Name: Predicting Legendary Pokemon
Authors: Sydney Ngao (sydney-n2), Yuanting Pan (YuantingPan), Christine King (cpk-1)
Description: This project predicts whether a pokemon is legendary or not. A legendary pokemon is a special type of pokemon which is especially rare and powerful. Our project takes the user input for their Pokemon's statistics such as base egg steps, base happiness level, total base stats, base special attack stat, and capture rate. Using a predictive model (decision tree) fitted on the data from the first seven generations of Pokemon, this project will run the inputted statistics through the model to create an output that predicts the legendary status of the Pokemon.
Python packages used: pandas (1.5.2), numpy (1.23.5), matplotlib (3.6.2), sklearn (1.1.3)
Demo file description: *insert later* 
Scope and limitations: The dataset we trained the predictive model on only contains the first seven generations of pokemon; it does not include the more recent generations and thus excludes the newer pokemon. One potential extension of this project would be to create a generator which could generate hypothetical but realistic pokemon, complete with stats. 
References and acknowledgements: Harlin Lee, Jason Schuchardt
Background/source of dataset: We used "The Complete Pokemon Dataset" from Kaggle (https://www.kaggle.com/datasets/rounakbanik/pokemon), provided by Rounak Banik. It contains the stats of 802 Pokemon across seven generations. 
No tutorials were used 
