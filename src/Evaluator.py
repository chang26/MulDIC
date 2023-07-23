from collections import Counter
import pandas as pd
from Logger import Logger

from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score

class Evaluator:
    def __init__(self, project, modelName, resultPath, manuals, issues, numClass):
        self.resultPath = resultPath
        self.modelName = modelName
        self.project = project
        self.manuals = manuals
        self.issues = issues
        self._prepare()
        self.numClass = numClass

    def _prepare(self):
        if self.project == 'komodo':
            self.docs = [manual.url.replace(' ', '-').lower() for manual in self.manuals]
        else:
            if self.project == 'vscode':
                self.docs = [manual.name.replace(' ', '-').lower().split('/')[-1] for manual in self.manuals]
            else:
                self.docs = [manual.name.replace(' ', '-').lower() for manual in self.manuals]

        ansPath = './AnswerSet/'
        df = pd.read_csv(ansPath+self.project+'.csv', encoding='cp949')
        df = df[['Document Files', 'Issue Number']]
        df = df.dropna()
        for issue in self.issues:
            idx = df[df['Issue Number'] == int(issue.number)].index
            try:
                doc = df.iloc[idx]['Document Files'].values[0].split('.')[0].strip().replace(' ', '-').lower()
                issue.realClass = self.docs.index(doc)
            except IndexError:
                del issue

    def evaluate(self, tp='title'):
        cm = self.modelName.split('-')[0]
        em = self.modelName.split('-')[1]
        if em == 'EmbeddingLayer':
            em = 'E'
        elif em == 'FastText':
            em = 'F'
        elif em == 'W2V':
            em = 'W'
        else:
            em = 'G'
        e = self.modelName.split('-')[-1].split('.')[0]

        logger = Logger('{}{}/{}/{}/{}/{}.txt'.format(self.resultPath, self.project, tp, cm, em, e))
        logger.seek(0)
        real = []
        predicted = []
        for issue in self.issues:
            if tp == 'title':
                if issue.titleVectors is not None and issue.realClass is not None:
                    real.append(issue.realClass)    
                    predicted.append(issue.titlePredictedClass)
            else:
                if issue.bodyVectors is not None and issue.realClass is not None:
                    real.append(issue.realClass)    
                    predicted.append(issue.bodyPredictedClass)

        _real = dict(Counter(real))
        for doc in self.docs:
            try:
                _real[doc] = _real.pop(self.docs.index(doc))
            except KeyError:
                continue

        _predicted = dict(Counter(predicted))
        for doc in self.docs:
            try:
                _predicted[doc] = _predicted.pop(self.docs.index(doc))
            except KeyError:
                continue

        logger.log('real=> {}\n'.format(str(Counter(_real))))
        logger.log('predicted=> {}\n'.format(str(Counter(_predicted))))

        precision = precision_score(real, predicted, average='weighted', zero_division=0)
        recall = recall_score(real, predicted, average='weighted', zero_division=0)
        f1 = f1_score(real, predicted, average='weighted', zero_division=0)
        acc = accuracy_score(real, predicted)
        logger.log('-------weighted-------\nprecision: {}\nrecall: {}\nf1-score: {}\naccuracy: {}\n'.format(precision, recall, f1, acc))