# PIC16A Project Fall 22

## Project Name: 
Legendary Pokemon Prediction

## Authors: 
Sydney Ngao (sydney-n2), Yuanting Pan (YuantingPan), Christine King (cpk-1) 

## Description: 
This project predicts whether a pokemon is legendary or not. A legendary pokemon is a special type of pokemon which is especially rare and powerful. Our project analyzes data from 802 Pokemon of the first seven generations to find the specific statistics that correlate most with whether a Pokemon is legendary or not and then uses those statistics to create a predictive model. Using a decision tree fitted on the data, this project will predict the legendary status of a Pokemon. 

## Python packages used: 
pandas (1.5.2), numpy (1.23.5), matplotlib (3.6.2), sklearn (1.1.3)

## Demo file description: 
`demo.py`contains a program that allows the user to use the model to find out if their Pokemon of choice is legendary or not. The program prompts for the user to input certain statistics of the Pokemon of choice. The Pokemon could be real or one that the user made up. 
`Notebook Demo.ipynb` is a jupyter notebook which shows graphs of the decision tree and correlation matrices for the dataset. 

## Scope and limitations: 
The dataset we trained the predictive model on only contains the first seven generations of pokemon; it does not include the more recent generations and thus excludes the newer pokemon. One potential extension of this project would be to create a generator which could generate hypothetical but realistic pokemon, complete with stats. 

## References and acknowledgements: 
Harlin Lee, Jason Schuchardt

## Background/source of dataset: 
We used "The Complete Pokemon Dataset" from Kaggle (https://www.kaggle.com/datasets/rounakbanik/pokemon), provided by Rounak Banik. It contains the stats of 802 Pokemon across seven generations.

No tutorials were used 
