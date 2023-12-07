import random
import pandas as pd
from datetime import datetime

path_dict = pd.DataFrame(columns=['Input', 'Path'])

def buggy_if_statement_test_harness(input_csv):
    for i, row in input_csv.iterrows():
        buggy_if_statement(row['Input'])
    

def buggy_if_statement(input_value):
    if input_value > 500 and input_value < 1000:
        random_number = random.randint(0, 1000)
        print(100/random_number)
        path_dict.loc[len(path_dict.index)] = [input_value, 1]
    else:
        path_dict.loc[len(path_dict.index)] = [input_value, 0]
        
if __name__ == "__main__":
    print("Would you like to use a csv file as input or generate random inputs? \n")
    print("1. CSV file \n")
    print("2. Random inputs \n")
    input_choice = input()
    if input_choice == "1":
        input_csv = pd.read_csv(input("Please enter the path to the csv file: \n"))
        buggy_if_statement_test_harness(input_csv)
    elif input_choice == "2":
        random.seed(datetime.now().timestamp())
        for i in range(100):
            input = random.randint(0, 1000)
            buggy_if_statement(input)
        path_dict.to_csv('Training CSVs/Buggy_test_case.csv', index=False)
    



