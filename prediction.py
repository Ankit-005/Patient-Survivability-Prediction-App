import torch.nn as nn
import torch
import numpy as np
from sklearn.preprocessing import LabelEncoder

class LinearNetwork(nn.Module):
    def __init__(self):
        super(LinearNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(8, 1024),
            nn.BatchNorm1d(num_features=1024),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.BatchNorm1d(num_features=512),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(512,64),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(64,64),
            nn.BatchNorm1d(num_features=64),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(64,2),
            nn.BatchNorm1d(num_features=2),
            nn.Dropout(p=0.3),
            nn.ReLU(),
            nn.Linear(2,1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.linear_relu_stack(x)


def survival(model,inp):
    inp=torch.FloatTensor(inp)
    pred=model(inp)
    pred=(pred>0.5)*1.0
    pred=pred.tolist()
    return pred[0][0]

def label(input_val,feat):
    le=LabelEncoder()
    le.fit(feat)
    value=le.transform(np.array([input_val]))
    return value[0]