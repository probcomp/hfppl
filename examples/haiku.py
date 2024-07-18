from hfppl import Model, CachedCausalLM, LMContext, smc_standard, sample_word
import asyncio
import nltk
import os

# download the CMU pronunciation dictionary (if we haven't already)
nltk.download("cmudict")

# Load the CMU pronunciation dictionary and use it for syllable counting
from nltk.corpus import cmudict

CMUDICT = cmudict.dict()


def count_syllables(word, unknown_word_syllables=100):
    # Use the dictionary to get the list of possible phonetic representations for the word
    phonetic_transcriptions = CMUDICT.get(word.strip().lower(), [])

    # Count the number of syllables based on the number of phonetic transcriptions
    syllable_count = min(
        [
            len([ph for ph in transcription if ph[-1].isdigit()])
            for transcription in phonetic_transcriptions
        ],
        default=unknown_word_syllables,
    )

    return syllable_count


# Load the language model (llama2 if authorized, else mistral-7b).
if "HF_AUTH_TOKEN" in os.environ:
    HF_AUTH_TOKEN = os.environ["HF_AUTH_TOKEN"]
    LLM = CachedCausalLM.from_pretrained(
        "meta-llama/Llama-2-7b-hf", auth_token=HF_AUTH_TOKEN
    )
else:
    LLM = CachedCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1")

# Set batch size
LLM.batch_size = 40

# Example poems for the prompt.
# Authors:
#   - Amy Lowell
#   - Sonia Sanchez
#   - Katsushika Hokusai
#   - Matsuo Basho
# Note that not all of these follow the syllabic constraints of a Haiku; the goal is
# to encode a certain 'poetic style' but to leave the syllabic constraints to be enforced
# by the probabilistic program (enabling generalization to other syllabic constraints).
example_poems = """Example poems. Note how they tend to end on a somewhat surprising or otherwise satisfying note, and are not repetitive at the end.

1. "Portrait"
Sweet smell of wet flowers
Over an evening garden.
Your portrait, perhaps?

2. "River of Love"
love between us is
speech and breath. loving you is
a long river running.

3. "Practice"
I write, erase, rewrite
Erase again, and then
A poppy blooms.

4. "Caterpillar"
A caterpillar,
this deep in fall,
still not a butterfly."""

# Ask user for poem title (without newline)
poem_title = input("Enter a title for your Haiku: ")
poem_prompt = f"""{example_poems}

5. "{poem_title}"
"""

# Cache prompt for faster generation
LLM.cache_kv(LLM.tokenizer.encode(poem_prompt))

# Useful constants
NEWLINE_TOKEN, EOS_TOKEN = 13, LLM.tokenizer.eos_token_id


# LLaMPPL model
class Haiku(Model):
    def __init__(self, prompt, syllable_pattern=[5, 7, 5]):
        super().__init__()
        self.context = LMContext(LLM, prompt, 0.7)
        self.syllable_pattern = syllable_pattern

    async def step(self):
        # Get the number of syllables required in the next line
        syllables_remaining = self.syllable_pattern.pop(0)

        # Loop to sample words until this line is over
        while syllables_remaining > 0:
            # Sample a word
            word, punctuation = await self.call(sample_word(self.context))

            # Subtract syllables from the remaining count
            syllables_remaining -= count_syllables(word)

        # Reject if we overshot
        self.condition(syllables_remaining == 0)

        # If there are no more lines, finish
        if not self.syllable_pattern:
            await self.observe(self.context.next_token(), EOS_TOKEN)
            self.finish()
            return

        # Otherwise, observe a line break
        await self.observe(self.context.next_token(), NEWLINE_TOKEN)

        # Print current result
        print(str(self.context))


# Run inference
SYLLABLES_PER_LINE = [5, 7, 5]  # [5, 3, 5] for a Lune
particles = asyncio.run(smc_standard(Haiku(poem_prompt, SYLLABLES_PER_LINE), 120))

print("--------")
for i, particle in enumerate(particles):
    print(f"Poem {i} (weight {particle.weight}):")
    print(f"{particle.context}")
