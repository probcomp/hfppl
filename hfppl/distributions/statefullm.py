from ..util import log_softmax, logsumexp
from .distribution import Distribution
from ..llms import Token, TokenSequence
import numpy as np

class StatefulLMNextToken(Distribution):
    
    def __init__(self, slm):
        self.slm = slm
    
    async def log_prob(self, x):
        if isinstance(x, Token):
            x = x.token_id
        
        lp = self.slm.next_token_logprobs[x]
        self.slm.s += x
        updated_logprobs = await self.slm.lm.next_token_logprobs(self.slm.s.seq)
        self.slm.next_token_logprobs = log_softmax(updated_logprobs / self.slm.temp)
        self.slm.model_mask = self.slm.NO_MASK
        
        return lp
    
    async def sample(self):
        probs = np.exp(self.slm.next_token_logprobs)
        token_id = np.random.choice(len(probs), p=(probs))
        logprob = self.slm.next_token_logprobs[token_id]
        t = Token(self.slm.lm, token_id, self.slm.lm.tokenizer.convert_ids_to_tokens(token_id))
        self.slm.s += t
        self.slm.model_mask = self.slm.NO_MASK
        updated_logprobs = await self.slm.lm.next_token_logprobs(self.slm.s.seq)
        self.slm.next_token_logprobs = log_softmax(updated_logprobs / self.slm.temp)
        return t, logprob
    
    
class StatefulLMMask(Distribution):
    def __init__(self, slm, mask):
        self.slm  = slm
        self.mask = mask
        
    async def sample(self):
        newly_bad_tokens  = [i for i in self.slm.model_mask if i not in self.mask]
        good_tokens       = [i for i in self.slm.model_mask if i in self.mask]
        logprob_no_mask   = logsumexp(self.slm.next_token_logprobs[newly_bad_tokens])
        logprob_yes_mask  = np.log1p(-np.exp(logprob_no_mask))
        decide_no_mask    = np.random.rand() < np.exp(logprob_no_mask)
        if decide_no_mask:
            self.slm.model_mask = self.slm.model_mask - self.mask
            self.slm.next_token_logprobs[good_tokens] = float('-inf')
            self.slm.next_token_logprobs -= logprob_no_mask
            return False, logprob_no_mask
        else:
            self.slm.model_mask = self.slm.model_mask.intersection(self.mask)
            self.slm.next_token_logprobs[newly_bad_tokens] = float('-inf')
            self.slm.next_token_logprobs -= logprob_yes_mask
            return True, logprob_yes_mask
        
    async def log_prob(self, v):
        good_tokens  = self.slm.model_mask.intersection(self.mask) if v else self.slm.model_mask - self.mask
        bad_tokens   = [i for i in self.slm.model_mask if i not in good_tokens]
        logprob_good = logsumexp(self.slm.next_token_logprobs[list(good_tokens)])
        self.slm.next_token_logprobs[bad_tokens] = float('-inf')
        self.slm.next_token_logprobs -= logprob_good
        self.slm.model_mask = good_tokens
        return logprob_good
        
    
class StatefulLM:
    """Represents a generation-in-progress from a language model.
    
    The state tracks two pieces of information:
    
    * A sequence of tokens — the ever-growing context for the language model.
    * A *current mask* — a set of tokens that have not yet been ruled out as the next token.
    
    Storing a mask enables _sub-token_ generation: models can use `StatefulLM` to sample
    the next token in _stages_, first deciding, e.g., whether to use an upper-case or lower-case
    first letter, and only later deciding which upper-case or lower-case token to generate.
    
    The state of a `StatefulLM` can be advanced in three ways:
    
    1. Sampling, observing, or intervening the `next_token()` distribution. This causes a token
    to be added to the growing sequence of tokens.
    2. Sampling, observing, or intervening the `mask_dist(mask)` distribution for a given mask (set of
    token ids). This changes the current mask.
    """
    
    def __init__(self, lm, prompt, temp=1.0):
        self.lm                  = lm
        self.s                   = TokenSequence(lm, prompt)
        self.next_token_logprobs = log_softmax(lm.next_token_logprobs_unbatched(self.s.seq) / temp)
        self.temp                = temp
        self.NO_MASK    = set(range(len(self.lm.vocab)))
        self.model_mask = self.NO_MASK
    
    def next_token(self):
        """Distribution over the next token.
        
        Sampling or observing from this distribution advances the state of this `StatefulLM` instance."""
        return StatefulLMNextToken(self)
    
    def mask_dist(self, mask):
        """Bernoulli distribution, with probability of True equal to the probability that the next token of this `StatefulLM` belongs
        to the given mask.
        
        Sampling or observing from this distribution modifies the state of this `StatefulLM` instance, so that
        the `next_token()` distribution either *will* (if True) or *will not* (if False) generate a token from
        the given mask.
        
        Args:
            mask: a `set(int)` specifying which token ids are included within the mask."""
        return StatefulLMMask(self, mask)