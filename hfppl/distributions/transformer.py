from .distribution import Distribution
from ..llms import TokenSequence, Token
import numpy as np

# Transformer(lm, prompt) -- where prompt can either be a string or a list of Tokens.
class Transformer(Distribution):

    # TODO: support custom temperatures
    def __init__(self, lm, prompt):
        self.lm = lm
        
        # prompt will be a list of ints
        if isinstance(prompt, str):
            prompt = self.lm.tokenizer.encode(prompt)
        elif isinstance(prompt, TokenSequence):
            prompt = prompt.seq
            
        self.log_probs = lm.next_token_logprobs(prompt)

    def log_prob(self, x):
        # Check if x is a token or an int
        if isinstance(x, Token):
            x = x.token_id
        
        return self.log_probs[x]
        
    def sample(self):
        probs = np.exp(self.log_probs)
        token_id = np.random.choice(len(probs), p=(probs))
        logprob = self.log_probs[token_id]
        return Token(self.lm, token_id, self.lm.tokenizer.convert_ids_to_tokens(token_id)), logprob
    
    def argmax(self, idx):
        token_id = np.argsort(self.log_probs)[-idx]
        return Token(self.lm, token_id, self.lm.tokenizer.convert_ids_to_tokens(token_id)), self.log_probs[token_id]