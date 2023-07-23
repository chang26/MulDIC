class Logger:
    def __init__(self, logPath):
        self.logger = open(logPath, 'a', encoding='utf-8')

    def log(self, data):
        self.logger.write(data)
        self.logger.flush()
    
    def seek(self, idx):
        self.logger.seek(0)