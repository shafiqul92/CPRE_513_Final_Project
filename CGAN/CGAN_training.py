import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

pd.options.mode.chained_assignment = None

#Data architecture
input_length = 0
batch_size = 0
path_length = 0
path_instances_unscaled = []
path_instances_scaled = []
input_min = np.inf
input_max = -np.inf

#Model architecture
latent_space_size = 20
generator_layer_size = [256, 512, 1024, 512, 256]
classifier_layer_size = [256, 512, 1024, 512, 256]
generator_dropout_rate = .2

# Training
epochs = 0
learning_rate = 1e-4

class Path_data(Dataset):
    def __init__(self, data_path):
        global input_max
        global input_min
        global path_instances_unscaled
        global path_instances_scaled
        global path_length
        global input_length
        self.path_min = np.inf
        self.path_max = -np.inf
        paths_df = pd.read_csv(data_path, dtype=str)
        for i, item in enumerate(paths_df['Path']):
            paths_df['Path'][i] = item.replace(';', '')
            paths_df['Path'][i] = [int(i) for i in paths_df['Path'][i]]
        for i, item in enumerate(paths_df['Path']):
            if path_length < len(paths_df['Path'][i]):
                path_length = len(paths_df['Path'][i])
        for i, item in enumerate(paths_df['Path']):
            if len(paths_df['Path'][i]) < path_length:
                for j in range(path_length - len(paths_df['Path'][i])):
                    paths_df['Path'][i].append(-1)
            for j in paths_df['Path'][i]:
                if j < self.path_min:
                    self.path_min = j
                if j > self.path_max:
                    self.path_max = j
        for i, item in enumerate(paths_df['Path']):
            paths_df['Path'][i] = [j - self.path_min for j in paths_df['Path'][i]]
            paths_df['Path'][i] = [j / (self.path_max - self.path_min) for j in paths_df['Path'][i]]
        for i, item in enumerate(paths_df['Input']):
            paths_df['Input'][i] = item.split(';')
            paths_df['Input'][i] = [float(i) for i in paths_df['Input'][i]]
        for i, item in enumerate(paths_df['Input']):
            if input_length < len(paths_df['Input'][i]):
                input_length = len(paths_df['Input'][i])
        for i, item in enumerate(paths_df['Input']):
            for j in paths_df['Input'][i]:
                if j < input_min:
                    input_min = j
                if j > input_max:
                    input_max = j
        for i, item in enumerate(paths_df['Input']):
            paths_df['Input'][i] = [j - input_min for j in paths_df['Input'][i]]
            paths_df['Input'][i] = [j / (input_max - input_min) for j in paths_df['Input'][i]]
        self.inputs = paths_df['Input'].values
        self.paths = paths_df['Path'].values
        print('--- Label Counts---')
        path_count = paths_df['Path'].value_counts()
        path_instances_unscaled = [[float(j) for j in i] for i in paths_df['Path'].value_counts().index.values.tolist()]
        path_instances_scaled = [[int(j * (self.path_max - self.path_min) + self.path_min) for j in i] for i in path_instances_unscaled]
        paths_strings = [str(i) for i in path_instances_scaled]
        path_count = pd.Series(dict(zip(paths_strings, list(path_count.values))))
        print(path_count)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        path = self.paths[idx]
        return input, path

class Generator(nn.Module):
    def __init__(self, generator_layer_size = generator_layer_size, latent_space_size = latent_space_size, path_length = path_length, input_size = input_length):
        super().__init__()
        
        self.latent_space_size = latent_space_size
        self.path_length = path_length
        self.input_size = input_size
        
        self.model = nn.Sequential(
            nn.Linear(self.latent_space_size + self.path_length, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(generator_dropout_rate),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(generator_dropout_rate),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(generator_dropout_rate),
            nn.Linear(generator_layer_size[2], generator_layer_size[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout1d(generator_dropout_rate),
            nn.Linear(generator_layer_size[3], self.input_size),
            nn.Sigmoid()
        )
    
    def forward(self, latent_space, labels):
        
        latent_space = latent_space.view(-1, self.latent_space_size)
        if labels.dim() == 1:
            labels = labels[None, :]
        x = torch.cat([latent_space, labels], 1)
        out = self.model(x)
        return out
    
class Classifier(nn.Module):
    def __init__(self, classifier_layer_size, path_length, input_length):
        super().__init__()
        
        self.input_length = input_length
        
        self.model = nn.Sequential(
            nn.Linear(self.input_length, classifier_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(classifier_layer_size[0], classifier_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(classifier_layer_size[1], classifier_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(classifier_layer_size[2], path_length),
            nn.Sigmoid()
        )
    
    def forward(self, inputs):
        out = self.model(inputs)
        return out

def generator_train_step(batch_size, classifier, generator, g_optimizer, criterion):
    classifier.train()
    g_optimizer.zero_grad()
    latent_space = torch.randn([batch_size, latent_space_size], requires_grad=True)
    fake_paths = torch.tensor(np.array([np.random.randint(0, 2, [batch_size, path_length])]), requires_grad=True, dtype=torch.float32)
    fake_paths = fake_paths.squeeze(0)
    fake_inputs = generator(latent_space, fake_paths)
    validity = classifier(fake_inputs)
    g_loss = criterion(validity, fake_paths)
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data

def classifier_train_step(batch_size, classifier, generator, c_optimizer, criterion, real_inputs, real_paths):
    classifier.eval()
    global path_instances_unscaled
    global path_instances_scaled
    c_optimizer.zero_grad()
    real_classifier_results = classifier(real_inputs)
    real_loss = criterion(real_classifier_results, real_paths)
    latent_space = torch.randn([batch_size, latent_space_size], requires_grad=True)
    random_path_indeces = np.random.randint(0, len(path_instances_unscaled), [batch_size])
    fake_paths = []
    for index in random_path_indeces:
        fake_paths.append(path_instances_unscaled[index])
    fake_paths = torch.tensor(fake_paths, requires_grad = True, dtype = torch.float32)
    fake_paths = fake_paths.squeeze(0)
    fake_inputs = generator(latent_space, fake_paths)
    fake_validity = classifier(fake_inputs)
    if fake_paths.dim() == 1:
        fake_paths = fake_paths[None, :]
    fake_loss = criterion(fake_validity, fake_paths)
    c_loss = real_loss + fake_loss
    c_loss.backward()
    c_optimizer.step()
    return c_loss.data

if __name__ == "__main__":
    
    train_data_path = input('Please enter the path to the data you\'d like to use to train this model:\n')
    
    training_data = pd.read_csv(train_data_path, dtype=str)
    
    print(f'This training data has {training_data.index.size} entries.')
    
    batch_size = input('Please enter the batch size for training this generator: \n')
    batch_size = int(batch_size)
    
    epochs = input('Please enter the number of epochs you\'d like to train this generator for: \n')
    epochs = int(epochs)
    
    generator_name = input('What would you like to name this generator?: \n')
    
    dataset = Path_data(train_data_path)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    
    generator = Generator(generator_layer_size, latent_space_size, path_length, input_length)
    classifier = Classifier(classifier_layer_size, path_length, input_length)
    criterion = nn.BCELoss()

    g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
    c_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

    for epoch in range(epochs):
        
        print('Starting epoch {}...'.format(epoch+1))
        
        for i, (real_inputs, real_paths) in enumerate(data_loader):
            
            real_paths = torch.stack(real_paths, dim = 1)
            real_inputs = real_inputs[0]
            if not torch.is_tensor(real_inputs):
                real_inputs = torch.tensor(real_inputs)
            real_paths = real_paths.type(torch.FloatTensor)
            real_inputs = real_inputs.type(torch.FloatTensor)
            real_inputs = real_inputs[:, None]
            generator.train()
            
            c_loss = classifier_train_step(len(real_inputs), classifier,
                                            generator, c_optimizer, criterion,
                                            real_inputs, real_paths)
            
            g_loss = generator_train_step(batch_size, classifier, generator, g_optimizer, criterion)
        
        generator.eval()
        
        if epoch % 25 == 0:
            latent_space = torch.randn([5, latent_space_size], requires_grad=True)
            random_path_indeces = np.random.randint(0, len(path_instances_unscaled), [5])
            fake_paths = []
            for index in random_path_indeces:
                fake_paths.append(path_instances_unscaled[index])
            fake_paths = torch.tensor(fake_paths, requires_grad = True, dtype = torch.float32)
            fake_paths = fake_paths.squeeze(0)
            sample_inputs = generator(latent_space, fake_paths)
            sample_inputs = sample_inputs.mul(input_max - input_min).add(input_min)
            scaled_paths = []
            for index in random_path_indeces:
                scaled_paths.append(path_instances_scaled[index])
            print('Sample paths: ' + str(scaled_paths))
            print('Sample inputs: ' + str(sample_inputs))
            torch.save(generator.state_dict(), f'{generator_name}.pt')