import hfppl as hp

from transformers import AutoTokenizer, AutoModelForCausalLM

import string

import asyncio

# Define a file hf_api_token.py with the constant HF_AUTH_TOKEN set to your HuggingFace API token (a string)
from hf_api_token import HF_AUTH_TOKEN

# Load the LLaMA model
LLAMA = hp.CachedCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", auth_token=HF_AUTH_TOKEN)
LLAMA.batch_size = 40


def can_follow(str_so_far, s):
    if isinstance(s, hp.Token):
        s = str(s)
    if len(s.strip()) > 5:
        return False
    if len(s.strip()) == 0:
        return True
    if not s[0].isalpha():
        return True
    if len(str_so_far) == 0:
        return True # First token, can be alphanumeric
    words = str_so_far.split()
    if len(words) >= 1 and len(words[-1]) + len(s) <= 5:
        return True
    else:
        return False

MASKS = {i : set(j for (j,v) in enumerate(LLAMA.vocab)
                 if j != LLAMA.tokenizer.eos_token_id and '\n' not in v and
                 any(c.isalpha() or c in string.punctuation for c in v) and
                 can_follow("a" * i, v)) for i in range(6)}

def constraint_mask(str_so_far):
    words = str_so_far.split()
    if len(words) >= 1:
        return MASKS[min(5, len(words[-1]))]
    else:
        return MASKS[0]


class ConstraintModel(hp.Model):
    def __init__(self, prompt, max_tokens):
        super().__init__()
        self.lm         = hp.StatefulLM(LLAMA, prompt)
        self.q          = hp.StatefulLM(LLAMA, prompt)
        self.prompt_len = len(str(self.lm.s))
        self.max_tokens = max_tokens
        

    async def step(self):
        # Generate proposed token.
        token = await self.sample(self.lm.next_token(), 
                                  proposal = await self.locally_optimal_proposal())

        # Condition on constraint — a no-op since proposal already guarantees the constraint
        # self.condition(can_follow(str(self.lm.s), token))
        
        # Reduce number of max tokens remaining
        self.max_tokens -= 1
        
        #if self.max_tokens % 5 == 0:
        print(str(self.lm.s)[self.prompt_len:])

        # Check if done
        if token == LLAMA.tokenizer.eos_token_id or self.max_tokens == 0:
            self.finish()
            return
    
    
    async def locally_optimal_proposal(self):
        string_so_far = str(self.lm.s)
        
        # Force the proposal StatefulLM to adhere to this mask
        await self.intervene(self.q.mask_dist(constraint_mask(string_so_far)), True)
        
        # Return the proposal's modified next-token distribution
        return self.q.next_token()

        
# From Politico.com
prompt = """3 things to watch …

1. The return of the House means new energy for the GOP’s Biden impeachment push, and Democrats are starting their pushback early. Rep. Jamie Raskin (D-Md.) is out this morning with a 14-page rebuttal memo that seeks to paint the GOP campaign as a “complete and total bust” and an attempt at distracting from the “overwhelming evidence of [Trump’s] criminal and corrupt conduct during his term of office.”

2. The Senate is back this evening for a bed-check vote. With Minority Leader Mitch McConnell having successfully quieted (public) chatter about his health, expect senators to be quizzed anew about Sen. Tommy Tuberville’s (R-Ala.) Pentagon nominee blockade, especially with the Joint Chiefs chair, Gen. Mark Milley, just weeks away from retirement and the confirmation of his successor, Gen. C.Q. Brown, in limbo.

3."""

LLAMA.cache_kv(LLAMA.tokenizer.encode(prompt))

async def main():
    constraint_model = ConstraintModel(prompt, 50)
    particles = await hp.smc_standard(constraint_model, 40)
    for p in particles:
        print(str(p.lm.s)[p.prompt_len:])

asyncio.run(main())