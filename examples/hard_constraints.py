import hfppl as hp
from transformers import AutoTokenizer, AutoModelForCausalLM
from hf_api_token import HF_AUTH_TOKEN

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


class ConstraintModel(hp.Model):
    def __init__(self, llm, prompt, can_follow):
        super().__init__()
        self.llm = llm
        self.s   = hp.TokenSequence(llm, prompt)
        self.can_follow = can_follow

    def step(self):
        # Generate proposed token.
        token = self.sample(hp.Transformer(self.llm, self.s), proposal=self.locally_optimal_proposal())

        # Condition on constraint
        self.condition(self.can_follow(str(self.s), token))

        # Check if done
        if token == self.llm.tokenizer.eos_token_id:
            self.finish()
            return
    
        self.s += token
    
    def locally_optimal_proposal(self):
        # Get next token logits
        logprobs   = self.llm.next_token_logprobs(self.s.seq)
        s_str = str(self.s)
        bad_tokens = [i for (word, i) in self.llm.tokenizer.vocab.items() if not self.can_follow(s_str, word)]
        logprobs[bad_tokens] = float('-inf')
        # Compute locally optimal proposal
        return hp.TokenCategorical(self.llm, logprobs)
    

lm = hp.CachedCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf", auth_token=HF_AUTH_TOKEN)
constraint_model = ConstraintModel(lm, "In economic news, the Fed says", can_follow)
particles = hp.smc_steer(constraint_model, 5, 5)
print(particles)
