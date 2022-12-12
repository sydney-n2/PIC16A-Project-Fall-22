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
### demo.py
`demo.py`contains a program that allows the user to use the model to find out if their Pokemon of choice is legendary or not. The program prompts for the user to input certain statistics of the Pokemon of choice. The Pokemon could be real or one that the user made up. 
![Semantic description of image](/readme_figure/fig_py.png)

*A sample use of `demo.py`. Input corresponding attributes of a Pokemon to the command, and it will tell you whether the Pokemon is legendary.*

### Notebook Demo.ipynb
`Notebook Demo.ipynb` is a jupyter notebook. It shows two more detailed sample uses of the model. The user can specify which features to use to train the model, visualize the correlation matrices for the features, score the models with different complexities, and visualize the decision tree.
![Semantic description of image](/readme_figure/fig_corr.png)

*Visualize the correlation matrix of selected features.*

![Semantic description of image](/readme_figure/fig_score.png)

*Visualize the scores for different model complexities (max-depth of tree).*

![Semantic description of image](/readme_figure/fig_tree.png)

*Visualize the decision tree after training.*

## Scope and limitations: 
The dataset we trained the predictive model on only contains the first seven generations of pokemon; it does not include the more recent generations and thus excludes the newer pokemon. One potential extension of this project would be to create a generator which could generate hypothetical but realistic pokemon, complete with stats. 

## References and acknowledgements: 
Harlin Lee, Jason Schuchardt

## Background/source of dataset: 
We used "The Complete Pokemon Dataset" from Kaggle (https://www.kaggle.com/datasets/rounakbanik/pokemon), provided by Rounak Banik. It contains the stats of 802 Pokemon across seven generations.

## Tutorials used:
No tutorials were used 

## Demo Video:
https://drive.google.com/file/d/1Gy9BXOoZ19ucok9jBPRXVzExetvt1fOU/view?usp=sharing
