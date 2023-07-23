import torch
import torch.nn as nn
from Models import Models

import random

class Model:
    def __init__(self, modelType=None, numClass=None, numCell=None, maxLen=None, test=False):
        if not test:
            self.model = Models.getModel(modelType, numClass, numCell, maxLen).cuda()
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
            self.optimizer22 = torch.optim.Adam(self.model.parameters(), lr=0.0002)
        else:
            self.model = None
    
    def fit(self, x, y, last, x_img=None, x_code=None):
        if x_img != None and x_code != None:
            pred = self.model(x_t = x, x_c = x_code, x_i = x_img)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            # self.optimizer22.step()
            if last:
                loss = None
        elif x_img != None:
            pred = self.model(x_t = x, x_i = x_img)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if last:
                loss = None
        elif x_code != None:
            pred = self.model(x_t = x, x_c = x_code)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()
            if last:
                loss = None
        else:
            pred = self.model(x)
            loss = self.criterion(pred, y)
            self.optimizer.zero_grad()
            loss.backward(retain_graph=True)
            self.optimizer.step()       
            # self.optimizer22.step()
            if last:
                loss = None

    def predict(self, x, x_img=None, x_code=None):
        if x_img != None and x_code != None:
            return self.model(x_t = x, x_c = x_code, x_i = x_img)
        elif x_img != None:
            return self.model(x_t = x, x_i = x_img)
        elif x_code != None:
            return self.model(x_t = x, x_c = x_code)
        else:
            return self.model(x)

    def save(self, path):
        torch.save(self.model, path)

    def load(self, path):
        self.model = torch.load(path)