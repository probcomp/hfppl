from .distribution import Distribution

class Geometric(Distribution):
    def __init__(self, p):
        self.p = p

    def sample(self):
        n = np.random.geometric(self.p)
        return n, self.log_prob(n)

    def log_prob(self, value):
        return np.log(self.p) + np.log(1 - self.p)*(value - 1)
    
    def argmax(self, idx):
        return idx - 1 # Most likely outcome is 0, then 1, etc.