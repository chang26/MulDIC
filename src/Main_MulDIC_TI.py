import os
import numpy as np
from collections import Counter
import time
import torch
import torch.nn as nn
import pandas as pd
import random

from TextPreprocessor import TextPreprocessor
from Trainer import Trainer
from Classifier import Classifier
from Logger import Logger

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

import torchvision.transforms as transforms # 이미지 변환 툴
import torchvision
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

seed_val = 0
random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EmbeddingLayer:
    def __init__(self, project, wordSet, embeddingSize):
        self.wordSet = wordSet
        self.embeddingSize = embeddingSize
        try:
            self.embedder = torch.load('./EmbeddingModel/{}-EmbeddingLayer_down_Mul_T-I_text.pt'.format(project))
        except FileNotFoundError:
            self.embedder = nn.Embedding(num_embeddings=len(wordSet), embedding_dim=embeddingSize)
            torch.save(self.embedder, './EmbeddingModel/{}-EmbeddingLayer_down_Mul_T-I_text.pt'.format(project))

    def embedding(self, lines, maxLen):
        emWords = None
        for line in lines:
            temp = []
            for i in range(maxLen):
                try:
                    word = line[i]
                    with torch.no_grad():
                        try:
                            idx = torch.tensor(self.wordSet.index(word))
                            if idx > len(self.wordSet):
                                continue
                            temp.append(idx)
                        except ValueError:
                            continue
                except IndexError:
                    continue

            if len(temp)>=1:
                with torch.no_grad():
                    temp = self.embedder(torch.tensor(temp))
                for i in range(maxLen-len(temp)):
                    temp = torch.cat((temp, torch.zeros(1, self.embeddingSize)), axis=0)
                if emWords is None:
                    emWords = temp.view(1, -1, self.embeddingSize)
                else:
                    emWords = torch.cat((emWords, temp.view(1, -1, self.embeddingSize)), axis=0)
        return emWords

class Evaluator:
    def __init__(self, project, resultPath, modelName, task=None, k=None):
        self.resultPath = resultPath
        self.modelName = modelName
        self.project = project

    def evaluate(self, predicted, real, tp='title'):      
        logger = Logger('{}{}/{}.txt'.format(self.resultPath, self.project, self.modelName))
        logger.log('real=> {}\n'.format(str(Counter(real))))
        logger.log('predicted=> {}\n'.format(str(Counter(predicted))))

        precision = precision_score(real, predicted, average='weighted', zero_division=0)
        recall = recall_score(real, predicted, average='weighted', zero_division=0)
        f1 = f1_score(real, predicted, average='weighted', zero_division=0)
        acc = accuracy_score(real, predicted)
        logger.log('-------weighted-------\nprecision: {}\nrecall: {}\nf1-score: {}\naccuracy: {}\n'.format(precision, recall, f1, acc))
        print('================= {} ================='.format(self.modelName))
        print('-------weighted-------\nprecision: {}\nrecall: {}\nf1-score: {}\naccuracy: {}\n'.format(precision, recall, f1, acc))
        print('=====================================')

class MAIN:
    def __init__(self):
        pass

    def run(self):
        #### Set the path to save the model. ####
        modelPath = './model_vscode_MulDIC_TI/'
        # modelPath = './model_kubernetes_MulDIC_TI/'
        # modelPath = './model_flutter_MulDIC_TI/'
        # modelPath = './model_roslyn_MulDIC_TI/'

        #### Set the project name. ####
        project = 'vscode'
        # project = 'kubernetes'
        # project = 'flutter'
        # project = 'roslyn'

        modelType = 'multimodal_TI'
        wordSet = None

        if project == 'vscode':
            labels = ['bug', 'feature']
        elif project == 'kubernetes':
            labels = ['bug', 'feature']
        elif project == 'flutter':
            labels = ['bug', 'feature']
        elif project == 'roslyn':
            labels = ['bug', 'feature']

        print('----{}----{}----'.format(project, modelType))
        train, test, train_img, test_img = self._readFile(project)
        
        print('train len: {}'.format(len(train)))
        print("=========================")
        print('test len: {}'.format(len(test)))
        print("=========================")
        print('train_img len: {}'.format(len(train_img)))
        print("=========================")
        print('test_img len: {}'.format(len(test_img)))
        print("=========================")
        
        trainX, trainY, trainWordSet = self._preprocess(train)

        testX, testY, testWordSet = self._preprocess(test)

        if wordSet is None:
            wordSet = trainWordSet
            wordSet.extend(testWordSet)
        wordSet = sorted(list(wordSet))

        trainLen = np.max([len(x[0]) for x in trainX])
        testLen = np.max([len(x[0]) for x in testX])
        maxLen = np.max([trainLen, testLen])
        maxLen = int(maxLen)
        print('maxLen: ', maxLen)

        trainX, trainY = self._embedding(project, trainX, trainY, wordSet, labels, maxLen)
        testX, testY = self._embedding(project, testX, testY, wordSet, labels, maxLen)

        self._train(trainX, trainY, train_img, project, modelPath, modelType, 1000, 256, maxLen, len(labels))
        self._test(testX, testY, test_img, project, modelPath, modelType)

        wordSet = None

    def _readFile(self, project):
        #### Read the data for that project ####
        if project == 'vscode':
            project_name = 'VS Code'
        elif project == 'kubernetes':
            project_name = 'Kubernetes'
        elif project == 'flutter':
            project_name = 'Flutter'
        elif project == 'roslyn':
            project_name = 'Roslyn'

        df1 = pd.read_csv('C://Users//MyPC//Desktop//Datasets//'+project_name+'//train_data//'+project_name+'_train_bug.csv')
        df2 = pd.read_csv('C://Users//MyPC//Desktop//Datasets//'+project_name+'//train_data//'+project_name+'_train_feature.csv')
        df3 = pd.read_csv('C://Users//MyPC//Desktop//Datasets//'+project_name+'//test_data//'+project_name+'_test_bug.csv')
        df4 = pd.read_csv('C://Users//MyPC//Desktop//Datasets//'+project_name+'//test_data//'+project_name+'_test_feature.csv')


        X_train = pd.concat([df1['title'], df2['title']], axis=0)
        X_test = pd.concat([df3['title'], df4['title']], axis=0)
        y_train = pd.concat([df1['label'], df2['label']], axis=0)
        y_test = pd.concat([df3['label'], df4['label']], axis=0)

        train = pd.concat([X_train, y_train], axis=1).values.tolist()
        test = pd.concat([X_test, y_test], axis=1).values.tolist()

        #### Read the image data for that project ####
        trans = transforms.Compose([transforms.Resize((258,258)), transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
        trainset = torchvision.datasets.ImageFolder(root='C://Users//MyPC//Desktop//Datasets//'+project_name+'//train_data',
                                                    transform=trans)
        testset = torchvision.datasets.ImageFolder(root='C://Users//MyPC//Desktop//Datasets//'+project_name+'//test_data',
                                                    transform=trans)
        train_img = trainset
        test_img = testset
        classes = trainset.classes
        print(classes)

        return train, test, train_img, test_img
    
    def _transform(self, train, test):
        train = [(' '.join(line.split(' ')[1:]).replace('"', ''), line.split(' ')[0], None) for line in train.split('\n')[:-1]]
        test = [(' '.join(line.split(' ')[1:]).replace('"', ''), line.split(' ')[0], None) for line in test.split('\n')[:-1]]
        return train, test

    def _preprocess(self, data):
        preprocessor = TextPreprocessor('tt')
        x = []
        y = []
        for d in data:
            t, l = preprocessor.pp(d)
            if t:
                x.append(t)
                y.append(l)
            else:
                continue
        return x, y, list(preprocessor.wordSet)

    def _embedding(self, project, x, y, wordSet, labels, maxLen):
        embedder = EmbeddingLayer(project, wordSet, 300)
        X = None
        Y = None
        for t, l in zip(x, y):
            emWords = embedder.embedding(t, maxLen)
            l = torch.tensor(labels.index(l)).view(-1)
            if emWords is not None:
                if X is None:
                    X = emWords
                    Y = l
                else:
                    X = torch.cat((X, emWords), dim=0)
                    Y = torch.cat((Y, l), dim=0)
            else:
                continue
        return X, Y

    def _train(self, X, Y, X_img, project, modelPath, modelType, epoch, batchSize, maxLen, numClass):
        trainer = Trainer(project, modelPath, modelType, epoch, 300, batchSize, maxLen, numClass, 'EmbeddingLayer')
        trainer.fit(X, Y, X_img)

    def _test(self, X, Y, X_img, project, modelPath, modelType, task=None, k=None):
        Y = Y.detach().cpu().numpy()
        for modelName in os.listdir(modelPath+project):
            classifier = Classifier(project, modelPath, modelName)
            prediction = classifier.classify(X, data_img=X_img).detach().cpu().numpy()
            self._evaluate(prediction, Y, project, modelName)

    def _evaluate(self, prediction, real, project, modelName, task=None, k=None):
        evaluator = Evaluator(project, './result/', modelName)
        evaluator.evaluate(prediction, real)

if __name__ == '__main__':
    start = time.time()

    main = MAIN()
    main.run()

    print("time :", time.time() - start)
    