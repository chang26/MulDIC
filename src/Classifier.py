import torch
import os
from Model import Model
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

class Classifier:
    def __init__(self, project, classifierPath, modelName):
        self.modelName = modelName
        self.model = Model(test=True)
        self.model.load('{}{}/{}'.format(classifierPath, project, modelName))
        
        self.batchSize = 256    ##
        # self.batchSize = 128    ##

    def classify(self, data, data_img=None, data_code=None, tp='title'):
        if data_img != None and data_code != None:
            dataset_text = TensorDataset(data)
            dataset_code = TensorDataset(data_code)
            loader_text = DataLoader(dataset_text, batch_size=self.batchSize, shuffle=False)
            loader_code = DataLoader(dataset_code, batch_size=self.batchSize, shuffle=False)
            img_test_loader = DataLoader(data_img, batch_size=self.batchSize, shuffle=False, num_workers=4)
            if self.modelName.startswith('rnn'):
                data = torch.flip(data, dims=[1]).cuda()
            else:
                data = data.cuda()
            if tp=='title':
                prediction = torch.tensor([]).cuda()
                for idx, (samples_text, samples_code, [image,label]) in enumerate(zip(loader_text, loader_code, img_test_loader)):
                    x_img = image.cuda()
                    if idx+1 == len(loader_text):
                        _last = True
                    x_text = samples_text[0]
                    x_text = x_text.cuda()
                    x_code = samples_code[0]
                    x_code = x_code.cuda()
                    p = torch.argmax(self.model.predict(x_text, x_code=x_code, x_img=x_img), dim=1)
                    prediction = torch.cat([prediction, p])

            else:
                prediction = self.model.predict(data, x_img=data_img, x_code=data_code)
            return prediction
        elif data_img != None:
            dataset = TensorDataset(data)
            loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=False)
            img_test_loader = DataLoader(data_img, batch_size=self.batchSize, shuffle=False, num_workers=4)
            if self.modelName.startswith('rnn'):
                data = torch.flip(data, dims=[1]).cuda()
            else:
                data = data.cuda()
            if tp=='title':
                prediction = torch.tensor([]).cuda()
                for idx, (samples, [image,label]) in enumerate(zip(loader, img_test_loader)):
                    x_img = image.cuda()
                    if idx+1 == len(loader):
                        _last = True
                    x = samples[0]
                    x = x.cuda()
                    p = torch.argmax(self.model.predict(x, x_img=x_img), dim=1)
                    prediction = torch.cat([prediction, p])
            else:
                prediction = self.model.predict(data, data_img)
            return prediction
        elif data_code != None:
            dataset_text = TensorDataset(data)
            dataset_code = TensorDataset(data_code)
            loader_text = DataLoader(dataset_text, batch_size=self.batchSize, shuffle=False)
            loader_code = DataLoader(dataset_code, batch_size=self.batchSize, shuffle=False)
            
            dataset = TensorDataset(data)
            loader = DataLoader(dataset, batch_size=self.batchSize, shuffle=False)
            img_test_loader = DataLoader(data_img, batch_size=self.batchSize, shuffle=False, num_workers=4)
            if self.modelName.startswith('rnn'):
                data = torch.flip(data, dims=[1]).cuda()
            else:
                data = data.cuda()
            if tp=='title':
                prediction = torch.tensor([]).cuda()
                for idx, (samples_text, samples_code) in enumerate(zip(loader_text, loader_code)):
                    if idx+1 == len(loader_text):
                        _last = True
                    x_text = samples_text[0]
                    x_text = x_text.cuda()
                    x_code = samples_code[0]
                    x_code = x_code.cuda()
                    p = torch.argmax(self.model.predict(x_text, x_code=x_code), dim=1)
                    prediction = torch.cat([prediction, p])
            else:
                prediction = self.model.predict(data, data_code)
            return prediction
        else:
            if self.modelName.startswith('rnn'):
                data = torch.flip(data, dims=[1]).cuda()
            else:
                data = data.cuda()
            if tp=='title':
                prediction = torch.argmax(self.model.predict(data), dim=1)
            else:
                prediction = self.model.predict(data)
            return prediction
        
    def classify_resultExtract(self, data, data_img=None, data_code=None, tp='title'):
        if data_img != None and data_code != None:
            dataset_text = TensorDataset(data)
            dataset_code = TensorDataset(data_code)
            loader_text = DataLoader(dataset_text, batch_size=self.batchSize, shuffle=False)
            loader_code = DataLoader(dataset_code, batch_size=self.batchSize, shuffle=False)
            img_test_loader = DataLoader(data_img, batch_size=self.batchSize, shuffle=False, num_workers=4)
            if self.modelName.startswith('rnn'):
                data = torch.flip(data, dims=[1]).cuda()
            else:
                data = data.cuda()
            if tp=='title':
                prediction = torch.tensor([]).cuda()
                predict_val = torch.tensor([]).cuda()     # 삭제
                for idx, (samples_text, samples_code, [image,label]) in enumerate(zip(loader_text, loader_code, img_test_loader)):
                    x_img = image.cuda()
                    if idx+1 == len(loader_text):
                        _last = True
                    x_text = samples_text[0]
                    x_text = x_text.cuda()
                    x_code = samples_code[0]
                    x_code = x_code.cuda()
                    p1 = self.model.predict(x_text, x_code=x_code, x_img=x_img)     # 삭제
                    print(p1.shape)     # 삭제
                    predict_val = torch.cat([predict_val, p1])     # 삭제
                    p = torch.argmax(self.model.predict(x_text, x_code=x_code, x_img=x_img), dim=1)
                    prediction = torch.cat([prediction, p])
            else:
                prediction = self.model.predict(data, x_img=data_img, x_code=data_code)
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")
            print(predict_val)
            print(predict_val.shape)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(prediction)
            print(prediction.shape)
            return prediction, predict_val
        else:
            if self.modelName.startswith('rnn'):
                data = torch.flip(data, dims=[1]).cuda()
            else:
                data = data.cuda()
            if tp=='title':
                p1 = self.model.predict(data)   # 삭제
                print(p1.shape)     # 삭제
                prediction = torch.argmax(self.model.predict(data), dim=1)
                print(prediction.shape)     # 삭제
            else:
                prediction = self.model.predict(data)
            return prediction, p1       # 삭제