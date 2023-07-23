import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from Model import Model

class Trainer:
    def __init__(self, project, classifierPath, modelType, epoch, numCell, batchSize, maxLen, numClass, embeddingType):
        self.project = project
        self.modelPath = classifierPath
        self.modelType = modelType
        self.epoch = epoch
        self.numCell = numCell
        self.batchSize = batchSize
        self.maxLen = maxLen
        self.numClass = numClass
        self.embeddingType = embeddingType
        self.model = Model(modelType, numClass, numCell, maxLen)

    def fit(self, X, Y, X_img=None, X_code=None):
        if X_img != None and X_code != None:
            dataset_text = TensorDataset(X, Y)
            dataset_code = TensorDataset(X_code, Y)
            loader_text = DataLoader(dataset_text, batch_size=self.batchSize, shuffle=False)
            loader_code = DataLoader(dataset_code, batch_size=self.batchSize, shuffle=False)
            img_train_loader = DataLoader(X_img, batch_size=self.batchSize, shuffle=False, num_workers=4)
            for e in range(1, self.epoch+1):
                print('{}/{} '.format(e, self.epoch), end='\r')
                _last = False

                for idx, (samples_text, samples_code, [image,label]) in enumerate(zip(loader_text, loader_code, img_train_loader)):
                    if idx+1 == len(loader_text):
                        _last = True
                    x_text, y = samples_text
                    x_code, y_code = samples_code
                    y = y.view(-1)
                    x_text = x_text.cuda()
                    x_code = x_code.cuda()
                    image = image.cuda()
                    y = y.cuda()
                    self.model.fit(x=x_text, y=y, last=_last, x_code=x_code, x_img=image)
                if e%50 == 0:
                    self.model.save('{}{}/{}-{}-{}.pt'.format(self.modelPath, self.project, self.modelType, self.embeddingType, e))
            print()
            # self.model.save('{}{}/{}-{}-{}.pt'.format(self.modelPath, self.project, self.modelType, self.embeddingType, e))
        elif X_img != None:
            dataset = TensorDataset(X, Y)
            loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=False)
            img_train_loader = DataLoader(X_img, batch_size=self.batchSize, shuffle=False, num_workers=4)
            for e in range(1, self.epoch+1):
                print('{}/{} '.format(e, self.epoch), end='\r')
                _last = False

                for idx, (samples, [image,label]) in enumerate(zip(loader, img_train_loader)):
                    x_img = image.cuda()
                    if idx+1 == len(loader):
                        _last = True
                    x, y = samples
                    y = y.view(-1)
                    x = x.cuda()
                    y = y.cuda()
                    self.model.fit(x, y, _last, x_img=x_img)
                if e%50 == 0:
                    self.model.save('{}{}/{}-{}-{}.pt'.format(self.modelPath, self.project, self.modelType, self.embeddingType, e))
            print()
            # self.model.save('{}{}/{}-{}-{}.pt'.format(self.modelPath, self.project, self.modelType, self.embeddingType, e))
        elif X_code != None:
            dataset_text = TensorDataset(X, Y)
            dataset_code = TensorDataset(X_code, Y)
            loader_text = DataLoader(dataset_text, batch_size=self.batchSize, shuffle=False)
            loader_code = DataLoader(dataset_code, batch_size=self.batchSize, shuffle=False)
            
            for e in range(1, self.epoch+1):
                print('{}/{} '.format(e, self.epoch), end='\r')
                _last = False

                for idx, (samples_text, samples_code) in enumerate(zip(loader_text, loader_code)):
                    if idx+1 == len(loader_text):
                        _last = True
                    x_text, y = samples_text
                    x_code, y_code = samples_code
                    y = y.view(-1)
                    x_text = x_text.cuda()
                    x_code = x_code.cuda()
                    y = y.cuda()
                    self.model.fit(x_text, y, _last, x_code=x_code)
                if e%50 == 0:
                    self.model.save('{}{}/{}-{}-{}.pt'.format(self.modelPath, self.project, self.modelType, self.embeddingType, e))
            print()
            # self.model.save('{}{}/{}-{}-{}.pt'.format(self.modelPath, self.project, self.modelType, self.embeddingType, e))
        else:
            dataset = TensorDataset(X, Y)
            loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=False)
            for e in range(1, self.epoch+1):
                print('{}/{} '.format(e, self.epoch), end='\r')
                _last = False
                for idx, samples in enumerate(loader):
                    if idx+1 == len(loader):
                        _last = True
                    x, y = samples
                    if self.modelType == 'rnn':
                        x = torch.flip(x, dims=[1])
                    y = y.view(-1)
                    x = x.cuda()
                    y = y.cuda()
                    self.model.fit(x, y, _last)
                if e%50 == 0:
                    self.model.save('{}{}/{}-{}-{}.pt'.format(self.modelPath, self.project, self.modelType, self.embeddingType, e))
            # self.model.save('{}{}/{}-{}-{}.pt'.format(self.modelPath, self.project, self.modelType, self.embeddingType, e))
            print()