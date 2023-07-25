import numpy as np
import torch
import torch.nn as nn


class Embedders:
    def __init__(self):
        self._wordSet = None

    @property
    def wordSet(self):
        return self._wordSet

    @wordSet.setter
    def wordSet(self, wordSet):
        self._wordSet = wordSet

    def embedding(self):
        raise ValueError

    def getEmbedder(self, project, embeddingType, embeddingSize):
        if embeddingType == "EmbeddingLayer":
            return EmbeddingLayer(project, embeddingSize)
        

class EmbeddingLayer(Embedders):
    def __init__(self, project, embeddingSize):
        self.embeddingSize = embeddingSize
        try:
            self.embedder = torch.load('./EmbeddingModel/{}-EmbeddingLayer.pt'.format(project))
        except FileNotFoundError:
            self.embedder = nn.Embedding(num_embeddings=len(super().wordSet), embedding_dim=embeddingSize)
            torch.save(self.embedder, './EmbeddingModel/{}-EmbeddingLayer.pt'.format(project))

    def embedding(self, lines, maxLen):
        emWords = None
        for line in lines:
            temp = []
            for i in range(maxLen):
                try:
                    word = line[i]
                    with torch.no_grad():
                        try:
                            idx = torch.tensor(super().wordSet.index(word))
                            temp.append(idx)
                        except ValueError:
                            continue
                except IndexError:
                    continue
            if len(temp)>3:
                with torch.no_grad():
                    temp = self.embedder(torch.tensor(temp))
                for i in range(maxLen-len(temp)):
                    temp = torch.cat((temp, torch.zeros(1, self.embeddingSize)), axis=0)
                if emWords is None:
                    emWords = temp.view(1, -1, self.embeddingSize)
                else:
                    emWords = torch.cat((emWords, temp.view(1, -1, self.embeddingSize)), axis=0)
        return emWords
