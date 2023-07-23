import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
# nltk.download()

class TextPreprocessor:
    def __init__(self, dataType):
        # nltk.download()
        self.wordSet = set()
        self.maxLen = 0
        self.dataType = dataType
        try:
            self.stopwords = list(set(stopwords.words('english')))
        except LookupError:
            nltk.download('stopwords')
        finally:
            self.stopwords = list(set(stopwords.words('english')))
        self.lemmatizer = WordNetLemmatizer()

    def pp(self, data):
        return self._preprocess(data)

    def _preprocess(self, data):
        processed = []
        if self.dataType == 'UserManual':
            data = data.lower()
            lines = data.split('\n')
            for line in lines:
                tokens = self._run(line)
                if tokens and len(tokens) >= 3:
                    if len(tokens) > self.maxLen:
                        self.maxLen = len(tokens)
                    processed.append(self._run(line))
                else:
                    continue
            return processed
        elif self.dataType == 'code':
            code, _= data
            code = [self._run_code(code.lower())]

            if len(code[0]) == 0:
                code[0].append('code')
            return code, _
        else:
            title, _= data
            title = [self._run(title.lower())]
            
            if len(title[0]) == 0:
                title[0].append('title')
            return title, _

    def _run_code(self, line):
        tokens = self._tokenize_code(line)
        tokens = self._lemmatize(tokens)
        return tokens

    def _run(self, line):
        tokens = self._tokenize(line)
        tokens = self._stopwordsRemoval(tokens)
        tokens = self._posTag(tokens)
        tokens = self._tagFilter(tokens)
        tokens = self._lemmatize(tokens)
        return tokens

    def _clean_text(self, text):
        text = text.replace('.', ' ').strip()
        text = text.replace("·", " ").strip()
        pattern = '[^ 0-9|a-zA-Z]+'
        text = re.sub(pattern=pattern, repl=' ', string=text)
        return text

    def _tokenize_code(self, sentence):
        tokenizer = nltk.SpaceTokenizer()
        return tokenizer.tokenize(sentence)

    def _tokenize(self, sentence):
        tokenizer = nltk.word_tokenize
        return tokenizer(sentence)

    def _stopwordsRemoval(self, words):
        return [word for word in words if word not in self.stopwords]

    def _posTag(self, tokens):
        tagger = nltk.pos_tag    
        return tagger(tokens, tagset='universal')
    
    def _tagFilter_code(self, tagged):
        filtered = []
        for word, pos in tagged:
            filtered.append(word)
        return filtered
    
    def _tagFilter(self, tagged):
        filtered = []
        for word, pos in tagged:
            if pos == '.' or pos == 'Num':
                continue 
            elif not word.isalpha():
                continue
            else:
                filtered.append(word)
        return filtered

    def _lemmatize(self, tokens):
        lemedTokens = []
        for token in tokens:
            lemedTokens.append(token)
            if token not in self.wordSet:
                self.wordSet.add(token)
        return lemedTokens