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

seed_val = 0
random.seed(seed_val)
torch.manual_seed(seed_val)
torch.cuda.manual_seed_all(seed_val)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

class EmbeddingLayer:
    def __init__(self, project, wordSet, embeddingSize, data_type):
        self.wordSet = wordSet
        self.embeddingSize = embeddingSize
        if data_type == 'text':
            try:
                self.embedder = torch.load('./EmbeddingModel/{}-EmbeddingLayer_down_Mul_T-C_text.pt'.format(project))
            except FileNotFoundError:
                self.embedder = nn.Embedding(num_embeddings=len(wordSet), embedding_dim=embeddingSize)
                torch.save(self.embedder, './EmbeddingModel/{}-EmbeddingLayer_down_Mul_T-C_text.pt'.format(project))
        elif data_type == 'code':
            try:                
                self.embedder = torch.load('./EmbeddingModel/{}-EmbeddingLayer_down_Mul_T-C_code.pt'.format(project))
            except FileNotFoundError:
                self.embedder = nn.Embedding(num_embeddings=len(wordSet), embedding_dim=embeddingSize)
                torch.save(self.embedder, './EmbeddingModel/{}-EmbeddingLayer_down_Mul_T-C_code.pt'.format(project))


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
        modelPath = './model_vscode_MulDIC_TC/'
        # modelPath = './model_kubernetes_MulDIC_TC/'
        # modelPath = './model_flutter_MulDIC_TC/'
        # modelPath = './model_roslyn_MulDIC_TC/'

        #### Set the project name. ####
        project = 'vscode'
        # project = 'kubernetes'
        # project = 'flutter'
        # project = 'roslyn'

        modelType = 'multimodal_TC'
        wordSet_text = None
        wordSet_code = None

        if project == 'vscode':
            labels = ['bug', 'feature']
        elif project == 'kubernetes':
            labels = ['bug', 'feature']
        elif project == 'flutter':
            labels = ['bug', 'feature']
        elif project == 'roslyn':
            labels = ['bug', 'feature']

        print('----{}----{}----'.format(project, modelType))
        train_text, test_text, train_code, test_code = self._readFile(project)
        
        print('train_text len: {}'.format(len(train_text)))
        print("=========================")
        print('test len: {}'.format(len(test_text)))
        print("=========================")
        print('train_code len: {}'.format(len(train_code)))
        print("=========================")
        print('test_code len: {}'.format(len(test_code)))
        print("=========================")

        trainX_text, trainY_text, trainWordSet_text = self._preprocess(train_text)
        trainX_code, trainY_code, trainWordSet_code = self._preprocess_code(train_code)              

        testX_text, testY_text, testWordSet_text = self._preprocess(test_text)
        testX_code, testY_code, testWordSet_code = self._preprocess_code(test_code)
        
        if wordSet_text is None:
            wordSet_text = trainWordSet_text
            wordSet_text.extend(testWordSet_text)
        wordSet_text = sorted(list(wordSet_text))
        
        if wordSet_code is None:
            wordSet_code = trainWordSet_code
            wordSet_code.extend(testWordSet_code)
        wordSet_code = sorted(list(wordSet_code))

        trainLen = np.quantile([len(x[0]) for x in trainX_code], 0.75)   # Third quartile
        testLen = np.quantile([len(x[0]) for x in testX_code], 0.75)
        maxLen = np.max([trainLen, testLen])
        maxLen = int(maxLen)
        print("maxLen:", maxLen)

        trainX_text, trainY_text = self._embedding(project, trainX_text, trainY_text, wordSet_text, labels, maxLen)
        trainX_code, trainY_code = self._embedding_code(project, trainX_code, trainY_code, wordSet_code, labels, maxLen)
        
        testX_text, testY_text = self._embedding(project, testX_text, testY_text, wordSet_text, labels, maxLen)
        testX_code, testY_code = self._embedding_code(project, testX_code, testY_code, wordSet_code, labels, maxLen)

        self._train(trainX_text, trainY_text, trainX_code, project, modelPath, modelType, 1000, 256, maxLen, len(labels))
        self._test(testX_text, testY_text, testX_code, project, modelPath, modelType)

        wordSet = None

    def _readFile(self, project):
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

        X_text_train = pd.concat([df1['title'], df2['title']], axis=0)
        X_text_test = pd.concat([df3['title'], df4['title']], axis=0)
        X_code_train = pd.concat([df1['code'], df2['code']], axis=0)
        X_code_test = pd.concat([df3['code'], df4['code']], axis=0)
        y_train = pd.concat([df1['label'], df2['label']], axis=0)
        y_test = pd.concat([df3['label'], df4['label']], axis=0)

        train_text = pd.concat([X_text_train, y_train], axis=1).values.tolist()
        test_text = pd.concat([X_text_test, y_test], axis=1).values.tolist()
        train_code = pd.concat([X_code_train, y_train], axis=1).values.tolist()
        test_code = pd.concat([X_code_test, y_test], axis=1).values.tolist()

        return train_text, test_text, train_code, test_code
    
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
    
    def _preprocess_code(self, data):
        preprocessor = TextPreprocessor('code')
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
        embedder = EmbeddingLayer(project, wordSet, 300, 'text')
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
    
    def _embedding_code(self, project, x, y, wordSet, labels, maxLen):
        embedder = EmbeddingLayer(project, wordSet, 300, 'code')
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

    def _train(self, X, Y, X_code, project, modelPath, modelType, epoch, batchSize, maxLen, numClass):
        trainer = Trainer(project, modelPath, modelType, epoch, 300, batchSize, maxLen, numClass, 'EmbeddingLayer')
        trainer.fit(X, Y, X_code=X_code)

    def _test(self, X, Y, X_code, project, modelPath, modelType, task=None, k=None):
        Y = Y.detach().cpu().numpy()
        for modelName in os.listdir(modelPath+project):
            classifier = Classifier(project, modelPath, modelName)
            prediction = classifier.classify(X, data_code=X_code).detach().cpu().numpy()
            self._evaluate(prediction, Y, project, modelName)

    def _evaluate(self, prediction, real, project, modelName, task=None, k=None):
        evaluator = Evaluator(project, './result/', modelName)
        evaluator.evaluate(prediction, real)

if __name__ == '__main__':
    start = time.time()

    main = MAIN()
    main.run()

    print("time :", time.time() - start)
    