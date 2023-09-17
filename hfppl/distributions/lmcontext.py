from ..util import log_softmax, logsumexp
from .distribution import Distribution
from ..llms import Token, TokenSequence
import numpy as np
import copy

class LMNextToken(Distribution):
    
    def __init__(self, ctx):
        self.ctx = ctx
    
    async def log_prob(self, x):
        if isinstance(x, Token):
            x = x.token_id
        
        lp = self.ctx.next_token_logprobs[x]
        self.ctx.s += x
        updated_logprobs = await self.ctx.lm.next_token_logprobs(self.ctx.s.seq)
        self.ctx.next_token_logprobs = log_softmax(updated_logprobs / self.ctx.temp)
        self.ctx.model_mask = self.ctx.NO_MASK
        
        return lp
    
    async def sample(self):
        probs = np.exp(self.ctx.next_token_logprobs)
        token_id = np.random.choice(len(probs), p=(probs))
        logprob = self.ctx.next_token_logprobs[token_id]
        t = Token(self.ctx.lm, token_id, self.ctx.lm.tokenizer.convert_ids_to_tokens(token_id))
        self.ctx.s += t
        self.ctx.model_mask = self.ctx.NO_MASK
        updated_logprobs = await self.ctx.lm.next_token_logprobs(self.ctx.s.seq)
        self.ctx.next_token_logprobs = log_softmax(updated_logprobs / self.ctx.temp)
        return t, logprob
    
    
class LMTokenMask(Distribution):
    def __init__(self, ctx, mask):
        self.ctx  = ctx
        self.mask = mask
        
    async def sample(self):
        newly_bad_tokens  = [i for i in self.ctx.model_mask if i not in self.mask]
        good_tokens       = [i for i in self.ctx.model_mask if i in self.mask]
        logprob_no_mask   = logsumexp(self.ctx.next_token_logprobs[newly_bad_tokens])
        logprob_yes_mask  = np.log1p(-np.exp(logprob_no_mask))
        decide_no_mask    = np.random.rand() < np.exp(logprob_no_mask)
        if decide_no_mask:
            self.ctx.model_mask = self.ctx.model_mask - self.mask
            self.ctx.next_token_logprobs[good_tokens] = float('-inf')
            self.ctx.next_token_logprobs -= logprob_no_mask
            return False, logprob_no_mask
        else:
            self.ctx.model_mask = self.ctx.model_mask.intersection(self.mask)
            self.ctx.next_token_logprobs[newly_bad_tokens] = float('-inf')
            self.ctx.next_token_logprobs -= logprob_yes_mask
            return True, logprob_yes_mask
        
    async def log_prob(self, v):
        good_tokens  = self.ctx.model_mask.intersection(self.mask) if v else self.ctx.model_mask - self.mask
        bad_tokens   = [i for i in self.ctx.model_mask if i not in good_tokens]
        logprob_good = logsumexp(self.ctx.next_token_logprobs[list(good_tokens)])
        self.ctx.next_token_logprobs[bad_tokens] = float('-inf')
        self.ctx.next_token_logprobs -= logprob_good
        self.ctx.model_mask = good_tokens
        return logprob_good
        
    
class LMContext:
    """Represents a generation-in-progress from a language model.
    
    The state tracks two pieces of information:
    
    * A sequence of tokens — the ever-growing context for the language model.
    * A *current mask* — a set of tokens that have not yet been ruled out as the next token.
    
    Storing a mask enables _sub-token_ generation: models can use `LMContext` to sample
    the next token in _stages_, first deciding, e.g., whether to use an upper-case or lower-case
    first letter, and only later deciding which upper-case or lower-case token to generate.
    
    The state of a `LMContext` can be advanced in two ways:
    
    1. Sampling, observing, or intervening the `next_token()` distribution. This causes a token
    to be added to the growing sequence of tokens. Supports auto-batching.
    2. Sampling, observing, or intervening the `mask_dist(mask)` distribution for a given mask (set of
    token ids). This changes the current mask.
    
    Attributes:
        lm (hfppl.llms.CachedCausalLM): the language model for which this is a context
        s (hfppl.llms.TokenSequence): the underlying sequence of tokens, including prompt, in this context
        next_token_logprobs (numpy.array): numpy array holding the log probabilities for the next token. Unlike the log probabilities reported by `CachedCausalLM.next_token_logprobs`, these probabilities are rescaled for this `LMContext`'s temperature parameter, and for any active masks. This vector is managed by the `LMContext` object internally; do not mutate.
        temp (float): temeprature for next-token distribution (0 < temp < float('inf'))
        model_mask (set[int]): set of tokens that have not been ruled out as the next token. This mask is managed by the `LMContext` object internally; do not mutate.
        show_prompt (bool): controls whether the string representation of this `LMContext` includes the initial prompt or not. Defaults to `False`.
    """
    
    def __init__(self, lm, prompt, temp=1.0):
        """Create a new `LMContext` with a given prompt and temperature.
        
        Args:
            lm (hfppl.llms.CachedCausalLM): the language model for which this is a context.
            prompt (str): a string with which to initialize the context. Will be tokenized using `lm.tokenizer`.
            temp (float): temeprature for next-token distribution (0 < temp < float('inf'))"""
        self.lm                  = lm
        self.s                   = TokenSequence(lm, prompt)
        self.next_token_logprobs = log_softmax(lm.next_token_logprobs_unbatched(self.s.seq) / temp)
        self.temp                = temp
        self.NO_MASK    = set(range(len(self.lm.vocab)))
        self.model_mask = self.NO_MASK
        self.prompt_string_length = len(str(self.s))
        self.show_prompt = False
        
    def next_token(self):
        """Distribution over the next token.
        
        Sampling or observing from this distribution advances the state of this `LMContext` instance."""
        return LMNextToken(self)
    
    def mask_dist(self, mask):
        """Bernoulli distribution, with probability of True equal to the probability that the next token of this `LMContext` belongs
        to the given mask.
        
        Sampling or observing from this distribution modifies the state of this `LMContext` instance, so that
        the `next_token()` distribution either *will* (if True) or *will not* (if False) generate a token from
        the given mask.
        
        Args:
            mask: a `set(int)` specifying which token ids are included within the mask."""
        return LMTokenMask(self, mask)        
    
    def __str__(self):
        base = 0 if self.show_prompt else self.prompt_string_length
        return str(self.s)[base:]
            
    def __deepcopy__(self, memo):        
        cpy = type(self).__new__(type(self))
        
        for k, v in self.__dict__.items():
            if k in set(['lm', 'NO_MASK']):
                setattr(cpy, k, v)
            else:
                setattr(cpy, k, copy.deepcopy(v, memo))
                
        return cpy