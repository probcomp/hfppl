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
    "meta-llama/Meta-Llama-3.1-8B-Instruct", auth_token=HF_AUTH_TOKEN
)

LLM.batch_size = 40


class VocabModel(Model):
    # Format of constraints: a set of words that must be used before the story can end.
    def __init__(self, prompt, words, max_tokens=150):
        super().__init__()
        self.context = LMContext(LLM, prompt)
        self.words = words
        self.max_len_tokens = max_tokens

    async def step(self):
        print(str(self.context))
        if self.context.token_count >= self.max_len_tokens:
            if len(self.words) > 0:
                self.condition(False)
                self.finish()
                return
            await self.observe(self.context.next_token(), LLM.vocab.index("<|eot_id|>"))
            self.finish()
            return

        num_words_left = len(self.words)

        if num_words_left == 0:
            # Generate until <|eot_id|> and finish
            while self.context.token_count <= self.max_len_tokens:
                token = await self.sample(self.context.next_token())
                if token == LLM.vocab.index("<|eot_id|>"):
                    self.finish()
                    return
            await self.observe(self.context.next_token(), LLM.vocab.index("<|eot_id|>"))
            self.finish()
            return

        # Generate words until next constraint or end of sentence
        while len(self.words) == num_words_left:
            word, _ = await self.call(
                sample_word(
                    self.context,
                    max_tokens=5,
                    allow_mid_punctuation=True,
                    allow_end_punctuation=True,
                )
            )
            print(self.words)
            if word.strip().lower() in self.words:
                print("GOT ONE!")
                self.words.remove(word.strip().lower())
            print(f"\t{(self.context)}")
        print("DONE WITH WHILE LOOP")

    def string_for_serialization(self):
        return f"{self.context}"


def instruction_prompt(words):
    words_string = "{" + ", ".join(words) + "}"
    return f"""<|start_header_id|>system<|end_header_id|>

Cutting Knowledge Date: December 2023
Today Date: 23 Jul 2024

You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>

Please write a story using ALL of the following words at least once (though they need not be in order): {words_string}.
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Sure, I can help with that. Here is a story using all those words:
"""


words = {"dog", "throw", "catch", "frisbee", "park", "carrot", "finger", "pillow"}
LLM.cache_kv(LLM.tokenizer.encode(instruction_prompt(words)))

# Run inference

particles = asyncio.run(
    smc_standard(
        VocabModel(instruction_prompt(words), words),
        20,
        0.5,
        "html",
        "results/force_vocab.json",
    )
)
