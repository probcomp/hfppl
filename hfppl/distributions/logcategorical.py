from .distribution import Distribution

class LogCategorical(Distribution):
    def __init__(self, logits):
        self.log_probs = log_softmax(logits)

    def sample(self):
        n = np.random.choice(len(self.log_probs), p=np.exp(self.log_probs))
        return n, self.log_prob(n)

    def log_prob(self, value):
        return self.log_probs[value]
    
    def argmax(self, idx):
        return np.argsort(self.log_probs)[-idx]