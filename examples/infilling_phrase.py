import asyncio
import copy
import os
import string

from hfppl import CachedCausalLM
from hfppl import LMContext
from hfppl import Model
from hfppl import observe_word
from hfppl import sample_token_constrained
from hfppl import sample_word
from hfppl import smc_standard

if "HF_AUTH_TOKEN" in os.environ:
    HF_AUTH_TOKEN = os.environ["HF_AUTH_TOKEN"]

# Load the language model.
# Mistral and Vicuna are open models; to use a model with restricted access, like LLaMA 3,
# pass your HuggingFace API key as the optional `auth_token` argument:
LLM = CachedCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-8B-Instruct", auth_token=HF_AUTH_TOKEN
)

LLM.batch_size = 40


def instruction_prompt(prefix, suffix):
    return f"""<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 Jul 2024

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

I need to fill in the blank in the following partial text:
{prefix}...[missing context]...{suffix}.

Please provide a fragment that could go in the blank. It should be a continuation of "{prefix}" and a lead-in to "{suffix}".
It should be roughly 8-15 words long.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Sure, I can help with that. The passage might go:

{prefix}"""


def model_prompt(prefix):
    return f"{prefix}"


class InfillingModel(Model):
    # Format of constraints: a dictionary mapping word index to observed word as a string
    def __init__(self, prefix, suffix):
        super().__init__()
        self.context = LMContext(LLM, model_prompt(prefix))
        self.proposal_context = LMContext(
            LLM, instruction_prompt(prefix, suffix), temp=1.2
        )
        self.suffix = LLM.tokenizer.encode(suffix, add_special_tokens=False)
        self.tokens_left = 15

    async def step(self):
        self.tokens_left -= 1
        await self.intervene(
            self.proposal_context.mask_dist(set([self.suffix[0]])), False
        )
        await self.sample(
            self.context.next_token(), proposal=self.proposal_context.next_token()
        )
        if self.tokens_left > 0:
            # Twist the target distribution.
            twist_context = copy.deepcopy(self.context)
            tokens_infix = LLM.tokenizer.encode(
                f" ...[{self.tokens_left} words missing]...", add_special_tokens=False
            )
            await self.intervene(
                twist_context.next_tokens(len(tokens_infix)), tokens_infix
            )
            lp = await twist_context.next_tokens(len(self.suffix)).log_prob(self.suffix)
            self.twist(lp)
        if self.tokens_left == 0:
            await self.observe(self.context.next_tokens(len(self.suffix)), self.suffix)
            self.finish()
        print(f"{self.context}")

    def string_for_serialization(self):
        return f"{self.context}"


prefix = """Once upon a time, there was a"""
suffix = """ others and make them feel safe. Mia liked to"""

LLM.cache_kv(LLM.tokenizer.encode(instruction_prompt(prefix, suffix)))

# Run inference
particles = asyncio.run(
    smc_standard(
        InfillingModel(prefix, suffix),
        80,
        0.5,
        "html",
        "results/infilling_sis.json",
    )
)
