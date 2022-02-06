import os
import math
import random as rn
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import transforms as T, datasets
import pytorch_lightning as pl


class MNISTClassifier(pl.LightningModule):
    def __init__(self, train_data_dir=None, batch_size=128, test_data_dir=None, num_workers=4):
        '''Constructor method 

        Parameters:
        train_data_dir (string): path of training dataset to be used either for training and validation
        batch_size (int): number of images per batch. Defaults to 128.
        test_data_dir (string): path of testing dataset to be used after training. Optional.
        num_workers (int): number of processes used by data loader. Defaults to 4.

        '''

        # Invoke constructor
        super(MNISTClassifier, self).__init__()

        # Set up class attributes
        self.batch_size = batch_size
        self.train_data_dir = train_data_dir
        self.test_data_dir = test_data_dir
        self.num_workers = num_workers
        self.pin_memory = True

        # mnist images are (1, 28, 28) (channels, height, width)
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 256)
        self.layer_3 = nn.Linear(256, 10)

        # Define network layers as class attributes to be used
        """ self.conv_layer_1 = torch.nn.Sequential(
            # The first block is made of a convolutional layer (3 channels, 28x28 images and a kernel mask of 5),
            torch.nn.Conv2d(3, 28, kernel_size=5),
            # a non linear activation function
            torch.nn.ReLU(),
            # a maximization layer, with mask of size 2
            torch.nn.MaxPool2d(kernel_size=2))

        # A second block is equal to the first, except for input size which is different
        self.conv_layer_2 = torch.nn.Sequential(
            torch.nn.Conv2d(28, 10, kernel_size=2),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2))

        # A dropout layer, useful to reduce network overfitting
        self.dropout1 = torch.nn.Dropout(0.25)

        # A fully connected layer to reduce dimensionality
        self.fully_connected_1 = torch.nn.Linear(250, 18)

        # Another fine tuning dropout layer to make network fine tune
        self.dropout2 = torch.nn.Dropout(0.08)

        # The final fully connected layer wich output maps to the number of desired classes
        self.fully_connected_2 = torch.nn.Linear(18, 10) """

    def load_split_train_test(self, valid_size=.2):
        '''Loads data and builds training/validation dataset with provided split size

        Parameters:
        valid_size (float): the percentage of data reserved to validation

        Returns:
        (torch.utils.data.DataLoader): Training data loader
        (torch.utils.data.DataLoader): Validation data loader
        (torch.utils.data.DataLoader): Test data loader

        '''

        num_workers = self.num_workers
        shuffle = True
        random_seed = 1234

        # Create transforms for data augmentation. Since we don't care wheter numbers are upside-down, we add a horizontal flip,
        # then normalized data to PyTorch defaults
        normalize = T.Normalize((0.1307,), (0.3081,))  # MNIST
        train_transform = T.Compose([T.RandomHorizontalFlip(),
                                      T.ToTensor(),
                                      normalize])
        valid_transform = T.Compose([
            T.ToTensor(),
            normalize
        ])                              
        # load the dataset
        train_dataset = datasets.MNIST(os.getcwd(), train=True, 
                    download=True, transform=train_transform)

        valid_dataset = datasets.MNIST(os.getcwd(), train=True, 
                    download=True, transform=valid_transform)

        num_train = len(train_dataset)
        indices = list(range(num_train))
        split = int(np.floor(valid_size * num_train))

        if shuffle == True:
            np.random.seed(random_seed)
            np.random.shuffle(indices)

        train_idx, valid_idx = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_idx)
        #valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = torch.utils.data.DataLoader(train_dataset, 
                        batch_size=self.batch_size, sampler=train_sampler, 
                        num_workers=num_workers, pin_memory=self.pin_memory)

        val_loader = torch.utils.data.DataLoader(valid_dataset, 
                        batch_size=self.batch_size,  
                        num_workers=num_workers, pin_memory=self.pin_memory)                              

        # if testing dataset is defined, we build its data loader as well
        test_loader = None
        test_dataset = datasets.MNIST(os.getcwd(), 
                               train=False, 
                               download=True,
                               transform=valid_transform)

        test_loader = torch.utils.data.DataLoader(test_dataset, 
                                              batch_size=self.batch_size, 
                                              shuffle=False, 
                                              num_workers=num_workers,
                                              pin_memory=self.pin_memory)

        # if self.test_data_dir is not None:
        #     test_transforms = T.Compose([T.ToTensor(), T.Normalize(
        #         [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        #     test_data = datasets.ImageFolder(
        #         self.test_data_dir, transform=test_transforms)
        #     test_loader = torch.utils.data.DataLoader(
        #         train_data, batch_size=self.batch_size, num_workers=num_workers)
        return train_loader, val_loader, test_loader

    def prepare_data(self):
        '''Prepares datasets. Called once per training execution
        '''
        self.train_loader, self.val_loader, self.test_loader = self.load_split_train_test()

    def train_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Training set data loader
        '''
        return self.train_loader

    def val_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Validation set data loader
        '''
        return self.val_loader

    def test_dataloader(self):
        '''
        Returns:
        (torch.utils.data.DataLoader): Testing set data loader
        '''
        return self.test_loader
        #return DataLoader(datasets.MNIST(os.getcwd(), train=False, download=False, transform=T.transform.ToTensor()), batch_size=128)

    def forward(self, x):
        '''Forward pass, it is equal to PyTorch forward method. Here network computational graph is built

        Parameters:
        x (Tensor): A Tensor containing the input batch of the network

        Returns: 
        An one dimensional Tensor with probability array for each input image
        '''

        batch_size, channels, height, width = x.size()

        # (b, 1, 28, 28) -> (b, 1*28*28)
        x = x.view(batch_size, -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)

        x = F.log_softmax(x, dim=1)
        return x

        """ x = self.conv_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.dropout1(x)
        x = torch.relu(self.fully_connected_1(x.view(x.size(0), -1)))
        x = F.leaky_relu(self.dropout2(x))
        return F.softmax(self.fully_connected_2(x), dim=1) """

    def configure_optimizers(self):
        '''
        Returns:
        (Optimizer): Adam optimizer tuned wit model parameters
        '''
        return torch.optim.Adam(self.parameters())

    def training_step(self, batch, batch_idx):
        '''Called for every training step, uses NLL Loss to compute training loss, then logs and sends back 
        logs parameter to Trainer to perform backpropagation

        '''

        # Get input and output from batch
        x, labels = batch

        # Compute prediction through the network
        prediction = self.forward(x)

        loss = F.nll_loss(prediction, labels)

        # Logs training loss
        self.log_dict({'train_loss':loss}, prog_bar=True)
        #print("Loss {:.4f} Loss2 {:.4f}".format(loss, loss))
        return loss
        

    def test_step(self, batch, batch_idx):
        '''Called for every testing step, uses NLL Loss to compute testing loss
        '''
        # Get input and output from batch
        x, labels = batch

        # Compute prediction through the network
        prediction = self.forward(x)

        loss = F.nll_loss(prediction, labels)

        # Logs training loss
        self.log_dict({'test_loss':loss}, prog_bar=True)
  
        return {
            'test_loss':loss
        }    
        """ logs = {'train_loss': loss}

        output = {
            # This is required in training to be used by backpropagation
            'loss': loss,
            # This is optional for logging pourposes
            'log': logs
        }

        return output """

    def validation_step(self, batch, batch_idx):
        ''' Performs model validation computing cross entropy for predictions and labels
        '''
        x, labels = batch
        prediction = self.forward(x)
        loss = F.cross_entropy(prediction, labels)
        # logs
        self.log_dict({'val_loss':loss}, prog_bar=True)
        
        return {
            'val_loss':loss
        }    

    """ def test_epoch_end(self, outputs):
        '''Called after every epoch, stacks testing loss
        '''
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'test_loss': test_loss_mean} """
        
