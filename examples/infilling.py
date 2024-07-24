import asyncio
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
    "meta-llama/Meta-Llama-3.1-8B", auth_token=HF_AUTH_TOKEN
)

LLM.batch_size = 40


class InfillingModel(Model):
    # Format of constraints: a dictionary mapping word index to observed word as a string
    def __init__(self, prompt, total_word_length, constrained_words):
        super().__init__()
        self.context = LMContext(LLM, prompt)
        self.constraints = constrained_words
        self.total_words = total_word_length
        self.next_word_index = 0

    async def step(self):

        # Generate words until next constraint or end of sentence
        while (
            self.next_word_index < self.total_words
            and self.next_word_index not in self.constraints
        ):
            word, _ = await self.call(
                sample_word(self.context, max_tokens=5, allow_mid_punctuation=True)
            )
            self.next_word_index += 1

        # If constraint, enforce
        if self.next_word_index in self.constraints:
            await self.call(
                observe_word(
                    self.context,
                    self.constraints[self.next_word_index],
                    allow_mid_punctuation=True,
                )
            )
            self.next_word_index += 1

        # If end of sentence, finish
        if self.next_word_index == self.total_words:
            await self.call(
                sample_token_constrained(
                    self.context, LLM.masks.END_PUNCTUATION, force=True
                )
            )
            await self.observe(self.context.next_token(), LLM.tokenizer.eos_token_id)
            self.finish()
            return

    def string_for_serialization(self):
        return f"{self.context}"


prompt = """<|begin_of_text|>Consider the following five unrelated sentences.
1. But while the mustache looked suitably grand, it was heavy and ungainly, and presented many practical difficulties for Branagh as a performer.
2. It was the longest seven hours of my life, I remember thinking.
3. Tesco removed the aids, which are sleep positioners known as nests, following a warning from the US.
4. Not one second later, John had completely reversed the situation; I was now on the floor.
5."""

LLM.cache_kv(LLM.tokenizer.encode(prompt))

# Run inference
particles = asyncio.run(
    smc_standard(
        InfillingModel(prompt, 13, {3: " her", 7: " down", 10: " the"}),
        20,
        0.5,
        "html",
        "results/infilling.json",
    )
)
