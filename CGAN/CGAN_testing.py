from CGAN_training import Generator
from CGAN_training import latent_space_size
import torch
import numpy as np
import pandas as pd

def Generate_test_cases(path_min, path_max, input_min, input_max):
    test_case_file_path = input("Give the path to a file of test cases for the model: \n")
    output_file_name = input("What would you like the output file for these test cases to be called? \n")
    test_cases_df = pd.read_csv(test_case_file_path, dtype=str)
    test_cases_df.insert(0, "Input", np.nan) 
    test_cases_df = test_cases_df.astype({"Input": str})
    for i, test_case_path in enumerate(test_cases_df['Path']):
        latent_space = torch.randn([1, latent_space_size], requires_grad=True)
        test_case_path = test_case_path.split(';')
        test_case_path = [(int(i) - path_min) / (path_max - path_min) for i in test_case_path]
        test_case_path = torch.tensor(test_case_path)
        test_case_path = test_case_path[None, :]
        x = Test_case_generator(latent_space, test_case_path)
        x= [float(i) * (input_max - input_min) + input_min for i in x[0:]]
        test_cases_df["Input"][i] = str(x).replace('[', '').replace(']','').replace(',', ';')
    test_cases_df.to_csv(output_file_name, index=False)
    return test_cases_df

def find_max_and_min_of_training_data_and_lengths():
    training_path = input("Give the path to the data you used to train this model: \n")
    path_min = np.inf
    path_max = -np.inf
    input_min = np.inf
    input_max = -np.inf
    path_length = -np.inf
    paths_df = pd.read_csv(training_path, dtype=str)
    path_length = 0
    input_length = 0
    for i, item in enumerate(paths_df['Path']):
        paths_df['Path'][i] = item.replace(';', '')
        paths_df['Path'][i] = [int(j) for j in paths_df['Path'][i]]
        if len(paths_df['Path'][i]) > path_length:
            path_length = len(paths_df['Path'][i])
    for i, item in enumerate(paths_df['Path']):
        if len(paths_df['Path'][i]) < path_length:
            for j in range(path_length - len(paths_df['Path'][i])):
                paths_df['Path'][i].append(-1)
        for j in paths_df['Path'][i]:
            if j < path_min:
                path_min = j
            if j > path_max:
                path_max = j
    for i, item in enumerate(paths_df['Input']):
        paths_df['Input'][i] = item.split(';')
        paths_df['Input'][i] = [float(j) for j in paths_df['Input'][i]]
        if len(paths_df['Input'][i]) > input_length:
            input_length = len(paths_df['Input'][i])
        for j in paths_df['Input'][i]:
            if j < input_min:
                input_min = j
            if j > input_max:
                input_max = j
    
    return path_min, path_max, input_min, input_max, path_length, input_length
                
if __name__ == "__main__":
    
    model_path = input("Give the path to the model you'd like to test: \n")
    
    path_min, path_max, input_min, input_max, path_length, input_length = find_max_and_min_of_training_data_and_lengths()
    
    Test_case_generator = Generator(path_length=path_length, input_size=input_length)
    
    Test_case_generator.load_state_dict(torch.load(model_path), strict=True)

    Test_case_generator.eval()
    
    Generate_test_cases(path_min, path_max, input_min, input_max)
