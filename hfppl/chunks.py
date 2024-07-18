import string
from .modeling import submodel

@submodel
async def sample_word(self, context, max_tokens=5, allow_punctuation=True):
    """Sample a word from the `LMContext` object `context`."""
    last_token = context.lm.vocab[context.tokens[-1]] if len(context.tokens) > 0 else ""
    last_character = last_token[-1] if len(last_token) > 0 else ""
    needs_space = last_character not in string.whitespace and last_character not in ['-', "'", '"']
    if needs_space:
        starts_word_mask = context.lm.masks.STARTS_NEW_WORD
    else:
        starts_word_mask = context.lm.masks.CONTINUES_CURRENT_WORD

    # Force model to start a new word
    await self.observe(context.mask_dist(starts_word_mask), True)

    word = ""
    num_tokens = 0
    while True:
        token       = await self.sample(context.next_token())
        word       += context.lm.vocab[token.token_id]
        num_tokens += 1

        if num_tokens == max_tokens:
            await self.observe(context.mask_dist(context.lm.masks.CONTINUES_CURRENT_WORD), False)
            break

        if not (await self.sample(context.mask_dist(context.lm.masks.CONTINUES_CURRENT_WORD))):
            break

    # Sample punctuation, if desired
    punctuation = ""
    if allow_punctuation and await self.sample(context.mask_dist(context.lm.masks.PUNCTUATION)):
        punctuation_token = await self.sample(context.next_token())
        punctuation = context.lm.vocab[punctuation_token.token_id]

    return word, punctuation


@submodel
async def sample_word_2(self, context, max_chars: int = None, allow_punctuation: bool = True):
    """Sample a word from the `LMContext` object `context`."""

    # This approach sometimes breaks with max_chars = 1
    if max_chars is not None:
        assert max_chars > 1

    last_token = context.lm.vocab[context.tokens[-1]] if len(context.tokens) > 0 else ""
    last_character = last_token[-1] if len(last_token) > 0 else ""
    needs_space = last_character not in string.whitespace and last_character not in ['-', "'", '"']
    if needs_space:
        starts_word_mask = context.lm.masks.STARTS_NEW_WORD
    else:
        starts_word_mask = context.lm.masks.CONTINUES_CURRENT_WORD

    # Force model to start a new word
    await self.observe(context.mask_dist(starts_word_mask), True)

    word = ""
    while True:
        # Force model to sample a token with an appropriate number of characters
        if max_chars is not None:
            await self.observe(context.mask_dist(context.lm.masks.MAX_TOKEN_LENGTH[max_chars - len(word.strip())]), True)

        token       = await self.sample(context.next_token())
        word       += context.lm.vocab[token.token_id]

        # If we ran out of chars, break
        if max_chars is not None and len(word.strip()) >= max_chars:
            await self.observe(context.mask_dist(context.lm.masks.CONTINUES_CURRENT_WORD), False)
            break

        # If the model wants to end the word, break
        if not (await self.sample(context.mask_dist(context.lm.masks.CONTINUES_CURRENT_WORD))):
            break

    # Sample punctuation, if desired
    punctuation = ""
    if allow_punctuation and await self.sample(context.mask_dist(context.lm.masks.END_PUNCTUATION)):
        punctuation_token = await self.sample(context.next_token())
        punctuation = context.lm.vocab[punctuation_token.token_id]

    return word, punctuation
