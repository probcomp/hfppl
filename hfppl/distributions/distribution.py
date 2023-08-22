class Distribution:

    def log_prob(self, x):
        raise NotImplementedError()
    
    def sample(self):
        raise NotImplementedError()
    
    def argmax(self, idx):
        raise NotImplementedError()