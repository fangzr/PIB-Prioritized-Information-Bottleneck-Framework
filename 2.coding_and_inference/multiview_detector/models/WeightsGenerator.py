import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import copy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def calculate_loss(weights_generator,camera_weights, delays, original_loss, weights_reg=0.001):
    # Calculate the original loss
    # original_loss = ...

    # Calculate the regularization term
    reg_term = 0
    for weight, delay in zip(camera_weights, delays.squeeze(0)):
        if delay.item() == 0:
            reg_term += weight ** 2
        elif delay.item() > 0:
            reg_term += (weight - weights_generator.target_weight) ** 2

    # Add the regularization term to the original loss
    total_loss = original_loss + weights_reg * reg_term
    total_loss = total_loss.sum()

    return total_loss


#A two-layer MLP network learns to generate a corresponding camera_weights based on the delays entered. 
# This MLP network can be used as a sub-module of PerspectiveTrainer

class WeightsGenerator(nn.Module):
    def __init__(self, input_size=7, hidden_size=32,target_weight=10.0):
        super(WeightsGenerator, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.target_weight = target_weight  
        
        self.to(device)

    def forward(self, x):
        # print('x:',x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x) * self.target_weight 
        return x.squeeze(0)  
    
