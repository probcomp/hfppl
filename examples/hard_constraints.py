import string
import asyncio
from hfppl import Model, CachedCausalLM, Token, StatefulLM, smc_standard

import os
HF_AUTH_TOKEN = os.environ['HF_AUTH_TOKEN']
# Load the language model. 
# Vicuna is an open model; to use a model with restricted access, like LLaMA 2,
# pass your HuggingFace API key as the optional `auth_token` argument:
LLM = CachedCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", auth_token=HF_AUTH_TOKEN)
# LLM = CachedCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")
LLM.batch_size = 40

MASKS = {i : set(j for (j,v) in enumerate(LLM.vocab)
                 if j != LLM.tokenizer.eos_token_id and '\n' not in v and
                 any(c.isalpha() or c in string.punctuation for c in v) and
                 len(v.strip()) <= 5 and (not v[0].isalpha() or i+len(v) <= 5))
             for i in range(6)}

class ConstraintModel(Model):
    def __init__(self, prompt, max_tokens):
        super().__init__()
        self.lm         = StatefulLM(LLM, prompt)
        self.q          = StatefulLM(LLM, prompt)
        self.prompt_len = len(str(self.lm.s))
        self.max_tokens = max_tokens
        

    async def step(self):
        # Which tokens are allowed?
        mask = self.active_constraint_mask()
        
        # Generate proposed token.
        token = await self.sample(self.lm.next_token(), 
                                  proposal = await self.proposal(mask))

        # Condition on constraint — a no-op since proposal already guarantees the constraint
        self.condition(token.token_id in mask)
        
        # Reduce number of max tokens remaining
        self.max_tokens -= 1
        
        #if self.max_tokens % 5 == 0:
        print(str(self.lm.s)[self.prompt_len:])

        # Check if done
        if token == LLM.tokenizer.eos_token_id or self.max_tokens == 0:
            self.finish()
            return
    
    def active_constraint_mask(self):
        string_so_far = str(self.lm.s)
        words = string_so_far.split()
        last_word = words[-1] if len(words) > 0 else ""
        return MASKS[min(5, len(last_word))]
    
    async def proposal(self, mask):
        string_so_far = str(self.lm.s)
        
        # Force the proposal StatefulLM to adhere to this mask
        await self.intervene(self.q.mask_dist(mask), True)
        
        # Return the proposal's modified next-token distribution
        return self.q.next_token()

        
# From Politico.com
prompt = """3 things to watch …

1. The return of the House means new energy for the GOP’s Biden impeachment push, and Democrats are starting their pushback early. Rep. Jamie Raskin (D-Md.) is out this morning with a 14-page rebuttal memo that seeks to paint the GOP campaign as a “complete and total bust” and an attempt at distracting from the “overwhelming evidence of [Trump’s] criminal and corrupt conduct during his term of office.”

2. The Senate is back this evening for a bed-check vote. With Minority Leader Mitch McConnell having successfully quieted (public) chatter about his health, expect senators to be quizzed anew about Sen. Tommy Tuberville’s (R-Ala.) Pentagon nominee blockade, especially with the Joint Chiefs chair, Gen. Mark Milley, just weeks away from retirement and the confirmation of his successor, Gen. C.Q. Brown, in limbo.

3."""

LLM.cache_kv(LLM.tokenizer.encode(prompt))

async def main():
    constraint_model = ConstraintModel(prompt, 50)
    particles = await smc_standard(constraint_model, 40)
    for p in particles:
        print(str(p.lm.s)[p.prompt_len:])

asyncio.run(main())