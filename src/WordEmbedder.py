from Embedders import Embedders

class WordEmbedder:
    embedder = None
    def __init__(self, project, embeddingType, embeddingSize, wordSet, maxLen):
        self.project = project
        self.embeddingType = embeddingType
        self.embeddingSize = embeddingSize
        self.wordSet = wordSet
        self.maxLen = maxLen
    
    def embedding(self, data):
        import time
        embedder = Embedders()
        Embedders.wordSet = self.wordSet

        if WordEmbedder.embedder is None:
            print('start to load embedding model')
            start = time.time()
            WordEmbedder.embedder = embedder.getEmbedder(self.project, self.embeddingType, self.embeddingSize)
            end = time.time()
            print('complete to load embedding model, {}s'.format(end-start))
            embedded = WordEmbedder.embedder.embedding(data, self.maxLen)
        else:
            embedded = WordEmbedder.embedder.embedding(data, self.maxLen)
        return embedded