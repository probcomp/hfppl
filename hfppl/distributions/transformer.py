from .distribution import Distribution
from ..llms import TokenSequence, Token
import numpy as np

# Transformer(lm, prompt) -- where prompt can either be a string or a list of Tokens.
class Transformer(Distribution):

    def __init__(self, lm, prompt, temp=1.0):
        """Create a Categorical distribution whose values are Tokens, with probabilities given
        by a language model. Supports auto-batching when used with `sample_async` and `observe_async`.
        
        Args:
            lm (hfppl.llms.CachedCausalLM): the language model.
            prompt (str | hfppl.llms.TokenSequence): the sequence of tokens to use as the prompt. If a string, `lm.tokenizer` is used to encode it.
            temp (float): temperature at which to generate (0 < `temp` < `float('inf')`).
        """
        self.lm = lm
        self.temp = temp
                
        # prompt will be a list of ints
        if isinstance(prompt, str):
            prompt = self.lm.tokenizer.encode(prompt)
        elif isinstance(prompt, TokenSequence):
            prompt = prompt.seq
            
            
        self.prompt = prompt
        
        
    async def log_prob_async(self, x):
        log_probs = await self.lm.next_token_logprobs_async(self.prompt)
        log_probs = log_probs / self.temp
        
        if isinstance(x, Token):
            x = x.token_id
        
        return log_probs[x]
    
    async def sample_async(self):
        log_probs = await self.lm.next_token_logprobs_async(self.prompt)
        log_probs = log_probs / self.temp
        probs = np.exp(log_probs)
        token_id = np.random.choice(len(probs), p=(probs))
        logprob = log_probs[token_id]
        return Token(self.lm, token_id, self.lm.tokenizer.convert_ids_to_tokens(token_id)), logprob
        
    def log_prob(self, x):
        log_probs = lm.next_token_logprobs(prompt)
        log_probs = log_probs / self.temp
        
        # Check if x is a token or an int
        if isinstance(x, Token):
            x = x.token_id
        
        return log_probs[x]
        
    def sample(self):
        log_probs = lm.next_token_logprobs(prompt)
        log_probs = log_probs / self.temp
        probs = np.exp(log_probs)
        token_id = np.random.choice(len(probs), p=(probs))
        logprob = log_probs[token_id]
        return Token(self.lm, token_id, self.lm.tokenizer.convert_ids_to_tokens(token_id)), logprob
    
#     def argmax(self, idx):
#         token_id = np.argsort(self.log_probs)[-idx]
#         return Token(self.lm, token_id, self.lm.tokenizer.convert_ids_to_tokens(token_id)), log_probs[token_id]