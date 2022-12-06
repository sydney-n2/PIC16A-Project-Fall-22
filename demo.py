from PokemonSet import *
import pandas as pd
# ignore some useless warnings which makes things ugly:
# import warnings
# from pandas.core.common import SettingWithCopyWarning
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
pd.options.mode.chained_assignment = None
#there is one more warning at the end that this doesn't clear up lol so it is still there for now

#this demo file will prompt user for a pokemon, then run that pokemon's information through the model and output the model's prediction of if it was a legendary or not 

def validate(value):
    try:
        float(value)
    except: 
        print("Please input a number.") 
        return 1
    else:
        if float(value) < 0: 
            print("Please input a positive number.")
            return 2
        else:
            return 0 
        
def make_model_object(): 
    df = pd.read_csv("pokemon.csv") 
    ps = PokemonSet(data = df,feature = ["base_egg_steps","base_happiness","base_total","sp_attack","capture_rate"])
    ps.clean_data()
    return ps.make_decision_tree_model(extra_output_enabled=False)

print("\n---- Welcome to a subset of the Pokemon wiki -----")
print("This model has been trained on all the Pokemon from generations 1-7. \nIt can tell you if your Pokemon is legendary or not!") 
print("Follow the prompts below for your Pokemon of choice. Let's find out if it's legendary! \n") 

model = make_model_object() 

egg = input("How many steps does it take to hatch this Pokemon's egg? ")
while validate(egg) > 0: 
    egg = input("How many steps does it take to hatch this Pokemon's egg? ") 

happiness = input("What is this Pokemon's base happiness level? ") 
while validate(happiness) > 0: 
    happiness = input("What is this Pokemon's base happiness level? ") 
    
total = input("What is this Pokemon's total base stats? ") 
while validate(total) > 0:
    total = input("What is this Pokemon's total base stats? ") 

spatk = input("What is this Pokemon's base special attack stat? ") 
while validate(spatk) > 0: 
    spatk = input("What is this Pokemon's base special attack stat? ") 

catch = input("What is this Pokemon's capture rate? ") 
while validate(catch) > 0: 
    catch = input("What is this Pokemon's base special attack stat? ") 

if model.predict([[egg, happiness, total, spatk, catch]]) > 0: 
    print("\nThis Pokemon is legendary! ") 
else: 
    print("\nThis Pokemon is not legendary. ") 
