class Model:
    def __init__(self):
        self.weight = 0.0
        self.finished = False
        self.mode = "sample"
        self.beam_idx = 0
        self.force_eos = False
        self.s = "<Particle>"

    def reset(self):
        self.weight = 0.0
        self.finished = False
        self.mode = "sample"
        self.beam_idx = 0
        self.force_eos = False
        self.s = "<Particle>"

    def finish(self):
        self.finished = True
    
    def done_stepping(self):
        return self.finished

    def step(self):
        if not self.done_stepping():
            raise NotImplementedError("Model.step() must be implemented by subclasses")
    
    def __str__(self):
        return str(self.s)
    
    def start(self):
        pass
    
    def score(self, score):
        self.weight += score

    def condition(self, b):
        if not b:
            self.score(float('-inf'))
            self.finish()
    
    def observe(self, dist, x):
        self.score(dist.log_prob(x))
        return x

    def sample(self, dist, proposal=None):
        # Special logic for beam search
        if self.mode == "beam":
            d = dist if proposal is None else proposal
            x, w = d.argmax(self.beam_idx)
            if proposal is not None:
                self.score(dist.log_prob(x))
            else:
                self.score(w)
            return x

        # If no proposal, sample from the distribution
        if proposal is None:
            x, _ = dist.sample()
            return x
        # Otherwise, sample from the proposal
        else:
            x, q = proposal.sample()
            self.score(dist.log_prob(x) - q)
            return x