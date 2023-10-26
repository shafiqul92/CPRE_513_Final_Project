import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

train_data_path = 'Sample_training_data.csv'
print('Train data path:', train_data_path)

#Data architecture
input_length = 2
batch_size = 1
path_length = 2

#Model architecture
latent_space_size = 10
generator_layer_size = [256, 512, 1024, 512, 256]
classifier_layer_size = [256, 512, 1024, 512, 256]

# Training
epochs = 100
learning_rate = 1e-4

class Path_data(Dataset):
    def __init__(self, data_path):
        paths_df = pd.read_csv(data_path)
        for i, item in enumerate(paths_df['Path']):
            paths_df['Path'][i] = item.replace(';', '')
            paths_df['Path'][i] = [int(i) for i in paths_df['Path'][i]]
        for i, item in enumerate(paths_df['Input']):
            paths_df['Input'][i] = item.replace(';', '')
            paths_df['Input'][i] = [int(i) for i in paths_df['Input'][i]]
        self.inputs = paths_df['Input'].values
        self.paths = paths_df['Path'].values
        print('--- Label Counts---')
        print(paths_df['Path'].value_counts())

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        input = self.inputs[idx]
        path = self.paths[idx]
        return input, path

class Generator(nn.Module):
    def __init__(self, generator_layer_size, latent_space_size, path_length, input_size):
        super().__init__()
        
        self.latent_space_size = latent_space_size
        self.path_length = path_length
        self.input_size = input_size
        
        self.model = nn.Sequential(
            nn.Linear(self.latent_space_size + self.path_length, generator_layer_size[0]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[0], generator_layer_size[1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[1], generator_layer_size[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[2], generator_layer_size[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(generator_layer_size[3], self.input_size),
            nn.Sigmoid()
        )
    
    def forward(self, latent_space, labels):
        
        latent_space = latent_space.view(-1, self.latent_space_size)
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
    
dataset = Path_data(train_data_path)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)
  
generator = Generator(generator_layer_size, latent_space_size, path_length, input_length)
classifier = Classifier(classifier_layer_size, path_length, input_length)
criterion = nn.BCELoss()

g_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate)
c_optimizer = torch.optim.Adam(classifier.parameters(), lr=learning_rate)

def generator_train_step(batch_size, classifier, generator, g_optimizer, criterion):
    g_optimizer.zero_grad()
    latent_space = torch.randn([batch_size, latent_space_size], requires_grad=True)
    fake_paths = torch.tensor([np.random.randint(0, 2, [batch_size, path_length])], requires_grad=True, dtype=torch.float32)
    fake_paths = fake_paths.squeeze(0)
    fake_inputs = generator(latent_space, fake_paths)
    validity = classifier(fake_inputs)
    g_loss = criterion(validity, fake_paths)
    g_loss.backward()
    g_optimizer.step()
    return g_loss.data

def classifier_train_step(batch_size, classifier, generator, c_optimizer, criterion, real_inputs, real_paths):
    c_optimizer.zero_grad()
    real_classifier_results = classifier(real_inputs)
    real_loss = criterion(real_classifier_results, real_paths)
    latent_space = torch.randn([batch_size, latent_space_size], requires_grad=True)
    fake_paths = torch.tensor([np.random.randint(0, 2, [batch_size, path_length])], requires_grad=True, dtype=torch.float32)
    fake_paths = fake_paths.squeeze(0)
    fake_inputs = generator(latent_space, fake_paths)
    fake_validity = classifier(fake_inputs)
    fake_loss = criterion(fake_validity, fake_paths)
    c_loss = real_loss + fake_loss
    c_loss.backward()
    c_optimizer.step()
    return c_loss.data

for epoch in range(epochs):
    
    print('Starting epoch {}...'.format(epoch+1))
    
    for i, (real_inputs, real_paths) in enumerate(data_loader):
        
        real_paths = torch.tensor(real_paths)
        real_inputs = torch.tensor(real_inputs)
        real_paths = real_paths.type(torch.FloatTensor)
        real_inputs = real_inputs.type(torch.FloatTensor)
        generator.train()
        
        c_loss = classifier_train_step(len(real_inputs), classifier,
                                          generator, c_optimizer, criterion,
                                          real_inputs, real_paths)
        
        g_loss = generator_train_step(batch_size, classifier, generator, g_optimizer, criterion)
    
    generator.eval()
    print('g_loss: {}, c_loss: {}'.format(g_loss, c_loss))
    latent_space = torch.randn([batch_size, latent_space_size], requires_grad=True)
    fake_paths = torch.tensor([np.random.randint(0, 2, [batch_size, path_length])], requires_grad=True, dtype=torch.float32)
    fake_paths = fake_paths.squeeze(0)
    sample_inputs = generator(latent_space, fake_paths)
    print('Sample paths: ' + str(fake_paths))
    print('Sample inputs: ' + str(sample_inputs))