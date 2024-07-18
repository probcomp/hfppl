"""Utilities for working with HuggingFace language models, including caching and auto-batching."""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import asyncio
import string


class Masks:
    def __init__(self, lm):
        self.ALL_TOKENS = set(range(len(lm.vocab)))
        self.STARTS_NEW_WORD = set(
            i
            for (i, v) in enumerate(lm.vocab)
            if v[0] == " "
            and len(v) > 1
            and v[1] not in string.whitespace
            and v[1] not in string.punctuation
        )
        self.CONTINUES_CURRENT_WORD = set(
            i
            for (i, v) in enumerate(lm.vocab)
            if all(c in "'" or c.isalpha() for c in v)
        )
        self.MID_PUNCTUATION = set(
            i for (i, v) in enumerate(lm.vocab) if v in (",", ":", ";", "-", '"')
        )
        self.END_PUNCTUATION = set(
            i for (i, v) in enumerate(lm.vocab) if v in (".", "!", "?")
        )
        self.PUNCTUATION = self.MID_PUNCTUATION | self.END_PUNCTUATION
        self.CONTAINS_WHITESPACE = set(
            i
            for (i, v) in enumerate(lm.vocab)
            if any(c in string.whitespace for c in v)
        )

        self.MAX_TOKEN_LENGTH = self.precompute_token_length_masks(lm)

    def precompute_token_length_masks(self, lm) -> Dict[int, Set[int]]:
        """Precompute masks for tokens of different lengths.

        Each mask is a set of token ids that are of the given length or shorter."""
        max_token_length = max([len(t) for t in lm.vocab])

        masks = defaultdict(lambda: self.ALL_TOKENS)
        masks[0] = set([lm.tokenizer.eos_token_id])
        for token_length in range(1, max_token_length + 1):
            masks[token_length] = set(
                i
                for (i, v) in enumerate(lm.vocab)
                if len(v) <= token_length and i != lm.tokenizer.eos_token_id
            )

        return masks


class TokenSequence:
    """A sequence of tokens.

    Supports addition (via `+` or mutating `+=`) with:

    * other `TokenSequence` instances (concatenation)
    * individual tokens, represented as integers or `Token` instances
    * strings, which are tokenized by `lm.tokenizer`

    Attributes:
        lm (hfppl.llms.CachedCausalLM): the language model whose vocabulary the tokens come from.
        seq (list[hfppl.llms.Token]): the sequence of tokens."""

    def __init__(self, lm, seq=None):
        """Create a `TokenSequence` from a language model and a sequence.

        Args:
            lm (hfppl.llms.CachedCausalLM): the language model whose vocabulary the tokens come from.
            seq (str | list[int]): the sequence of token ids, or a string which will be automatically tokenized. Defaults to the singleton sequence containing a bos token.
        """
        self.lm = lm
        if seq is None:
            self.seq = [lm.tokenizer.bos_token_id]
        elif isinstance(seq, str):
            self.seq = self.lm.tokenizer.encode(seq)
        else:
            self.seq = seq

    def __str__(self):
        return self.lm.tokenizer.decode(self.seq)

    def __iadd__(self, other):
        if isinstance(other, Token):
            assert other.lm is self.lm
            self.seq.append(other.token_id)
        elif isinstance(other, TokenSequence):
            assert other.lm is self.lm
            self.seq.extend(other.seq)
        elif isinstance(other, str):
            self.seq.extend(self.lm.tokenizer.encode(other, add_special_tokens=False))
        elif isinstance(other, int):
            self.seq.append(other)
        else:
            raise RuntimeError(f"Addition not supported on {type(other)}")
        return self

    def __radd__(self, other):
        if isinstance(other, Token):
            assert other.lm is self.lm
            return TokenSequence(self.lm, [other.token_id, *self.seq])
        elif isinstance(other, TokenSequence):
            assert other.lm is self.lm
            return TokenSequence(self.lm, other.seq + self.seq)
        elif isinstance(other, str):
            return TokenSequence(
                self.lm,
                self.lm.tokenizer.encode(other, add_special_tokens=False) + self.seq,
            )
        elif isinstance(other, int):
            return TokenSequence(self.lm, [other, *self.seq])
        else:
            raise RuntimeError(f"Addition not supported on {type(other)}")

    def __add__(self, other):
        s = TokenSequence(self.lm, self.seq)
        s += other
        return s


class Token:
    """Class representing a token.

    Attributes:
        lm (hfppl.llms.CachedCausalLM): the language model for which this is a Token.
        token_id (int): the integer token id (an index into the vocabulary).
        token_str (str): a string, which the token represents—equal to `lm.vocab[token_id]`.
    """

    def __init__(self, lm, token_id, token_str):
        self.lm = lm
        self.token_id = token_id
        self.token_str = token_str

    # Adding tokens
    def __add__(self, other):
        s = TokenSequence(self.lm, [self.token_id])
        s += other
        return s

    def __radd__(self, other):
        s = TokenSequence(self.lm, [self.token_id])
        return other + s

    # Support checking for EOS
    def __eq__(self, other):
        if isinstance(other, Token):
            return self.lm is other.lm and self.token_id == other.token_id
        elif isinstance(other, int):
            return self.token_id == other
        else:
            return self.token_str == other

    def __str__(self):
        return self.token_str

    def __repr__(self):
        return f"<{self.token_str}|{self.token_id}>"


class TokenTrie:
    """Class used internally to cache language model results."""

    # Trie of tokens.

    def __init__(self, parent=None, logprobs=None):
        self.children = {}  # maps token ID to child
        self.logprobs = logprobs  # for next token
        self.past_key_values = None

    def __repr__(self):
        return (
            f"{'*' if self.past_key_values is not None else ''}["
            + ", ".join(
                [
                    f"{node_id}: {node.__repr__()}"
                    for (node_id, node) in self.children.items()
                ]
            )
            + "]"
        )

    def clear_kv_cache(self):
        self.past_key_values = None
        for child, node in self.children.items():
            node.clear_kv_cache()

    def has_token(self, token_id):
        return token_id in self.children

    def get_token(self, token_id):
        return self.children[token_id]

    def add_token(self, token_id, logprobs=None):
        self.children[token_id] = TokenTrie(self, logprobs)
        return self.children[token_id]

    def extend_cache(self, next_token_index, token_ids, logits, base):
        node = self

        for j in range(next_token_index, len(token_ids)):
            token_id = token_ids[j]
            token_logits = logits[j - base]
            token_logprobs = torch.log_softmax(token_logits, 0)

            node = node.add_token(token_id, token_logprobs.cpu().numpy())

        return node


class Query:
    """A query to a language model, waiting to be batched."""

    def __init__(self, prompt, future, past=None):
        self.prompt = prompt
        self.future = future
        self.past = past

        if self.past is not None:
            self.past_len = past[0][0].shape[
                2
            ]  # layers, key or value, batch size, num heads, num tokens, head repr length
        else:
            self.past_len = 0

    @torch.no_grad()
    def past_padded(self, layer, j, to_length, dtype, device, past_shape):

        if self.past is not None:
            return torch.cat(
                (
                    self.past[layer][j],
                    torch.zeros(
                        1,
                        past_shape[1],
                        to_length - self.past_len,
                        past_shape[3],
                        dtype=dtype,
                        device=device,
                    ),
                ),
                dim=2,
            )
        else:
            return torch.zeros(
                1, past_shape[1], to_length, past_shape[3], dtype=dtype, device=device
            )

    def prompt_padded(self, pad_token, to_length):
        return [*self.prompt, *[pad_token for _ in range(to_length - len(self.prompt))]]

    def attention_mask(self, total_past_length, total_seq_length):
        return [
            *[1 for _ in range(self.past_len)],
            *[0 for _ in range(total_past_length - self.past_len)],
            *[1 for _ in range(len(self.prompt))],
            *[0 for _ in range(total_seq_length - len(self.prompt))],
        ]

    def position_ids(self, total_past_length, total_seq_length):
        return [
            *range(self.past_len, self.past_len + len(self.prompt)),
            *[0 for _ in range(total_seq_length - len(self.prompt))],
        ]


class CachedCausalLM:
    """Wrapper around a HuggingFace causal language model, with support for caching.

    Attributes:
        model: the underlying HuggingFace model.
        tokenizer: the underlying HuggingFace tokenizer.
        device (str): the PyTorch device identifier (e.g. "cpu" or "cuda:0") on which the model is loaded.
        cache (hfppl.llms.TokenTrie): the cache of previously evaluated log probabilities and key/value vectors.
        vocab (list[str]): a list mapping token ids to strings.
        batch_size (int): when auto-batching, maximum number of queries to process in one batch.
        timeout (float): number of seconds to wait since last query before processing the current batch of queries, even if not full.
    """

    @classmethod
    def from_pretrained(cls, model_id, auth_token=False, load_in_8bit=True):
        """Create a [`CachedCausalLM`][hfppl.llms.CachedCausalLM] from a pretrained HuggingFace model.

        Args:
            model_id (str): the string identifier of the model in HuggingFace's model library.
            auth_token (str): a HuggingFace API key. Only necessary if using private models, e.g. Meta's Llama models, which require authorization.
            load_in_8bit (bool): whether to use the `bitsandbytes` library to load the model in 8-bit quantized form.

        Returns:
            model (hfppl.llms.CachedCausalLM): the LLaMPPL-compatible interface to the HuggingFace model.
        """
        bnb_config = BitsAndBytesConfig(load_in_8bit=load_in_8bit)

        if not auth_token:
            tok = AutoTokenizer.from_pretrained(model_id)
            mod = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                quantization_config=bnb_config,
            )
        else:
            tok = AutoTokenizer.from_pretrained(model_id, token=auth_token)
            mod = AutoModelForCausalLM.from_pretrained(
                model_id,
                token=auth_token,
                device_map="auto",
                quantization_config=bnb_config,
            )

        return CachedCausalLM(mod, tok)

    @torch.no_grad()
    def __init__(self, hf_model, hf_tokenizer, batch_size=20):
        """
        Create a `CachedCausalLM` from a loaded HuggingFace model and tokenizer.

        Args:
            hf_model: a HuggingFace `CausalLM`.
            hf_tokenizer: a HuggingFace `Tokenizer`.
            batch_size (int): when auto-batching, maximum number of queries to process in one batch.
        """
        self.model = hf_model
        self.tokenizer = hf_tokenizer
        self.device = hf_model.device

        # TODO: remove required BOS token
        if self.tokenizer.bos_token_id is None:
            raise RuntimeError(
                "Causal LM has no BOS token, distribution of first word unclear"
            )

        # Evaluate BOS token
        logits = self.model(
            torch.tensor([[self.tokenizer.bos_token_id]]).to(self.model.device)
        ).logits[0][0]
        logprobs = torch.log_softmax(logits, 0)

        self.cache = TokenTrie(None, logprobs.cpu().numpy())

        # Cache vocabulary
        bos_len = len(self.tokenizer.decode([self.tokenizer.bos_token_id]))
        self.vocab = [
            self.tokenizer.decode([self.tokenizer.bos_token_id, i])[bos_len:]
            for i in range(len(hf_tokenizer.vocab))
        ]

        # Precompute useful masks
        self.masks = Masks(self)

        # Queries to be batched. Each query is a sequence of tokens,
        # and a Future to be called when the query is resolved.
        self.queries = []
        self.batch_size = batch_size
        self.timeout = 0.02
        self.timer = None

    def __deepcopy__(self, memo):
        return self

    def clear_cache(self):
        """Clear the cache of log probabilities and key/value pairs."""
        self.cache = TokenTrie(None, self.cache.logprobs)

    def clear_kv_cache(self):
        """Clear any key and value vectors from the cache."""
        self.cache.clear_kv_cache()

    def reset_async_queries(self):
        """Clear any pending language model queries from the queue. Use this method when an exception prevented an inference algorithm from executing
        to completion."""
        self.queries = []

    @torch.no_grad()
    def cache_kv(self, prompt_tokens):
        """Cache the key and value vectors for a prompt. Future queries that have this prompt as a prefix will only run the LLM on new tokens.

        Args:
            prompt_tokens (list[int]): token ids for the prompt to cache.
        """
        result = self.model(torch.tensor([prompt_tokens]).to(self.device))

        node = self.cache.extend_cache(1, prompt_tokens, result.logits[0], 0)
        node.past_key_values = result.past_key_values

    @torch.no_grad()
    def batch_evaluate_queries(self):

        queries, self.queries = self.queries, []
        if len(queries) == 0:
            return

        past_example = next((q.past for q in queries if q.past), False)
        max_past_length = max(q.past_len for q in queries)
        max_query_length = max(len(q.prompt) for q in queries)

        padding_token_id = (
            self.tokenizer.pad_token_id
            if self.tokenizer.pad_token_id is not None
            else 0
        )

        input_ids = torch.tensor(
            [q.prompt_padded(padding_token_id, max_query_length) for q in queries]
        ).to(self.device)
        attn_masks = torch.tensor(
            [q.attention_mask(max_past_length, max_query_length) for q in queries]
        ).to(self.device)
        posn_ids = torch.tensor(
            [q.position_ids(max_past_length, max_query_length) for q in queries]
        ).to(self.device)
        if past_example:
            pasts = [
                [
                    torch.cat(
                        (
                            *(
                                q.past_padded(
                                    layer,
                                    j,
                                    max_past_length,
                                    past_example[0][0].dtype,
                                    self.device,
                                    past_example[0][0].shape,
                                )
                                for q in queries
                            ),
                        ),
                        dim=0,
                    )
                    for j in range(2)
                ]
                for layer in range(len(past_example))
            ]
        else:
            pasts = None

        results = self.model(
            input_ids,
            attention_mask=attn_masks,
            position_ids=posn_ids,
            past_key_values=pasts,
            use_cache=pasts is not None,
        )

        for i, q in enumerate(queries):
            q.future.set_result(results.logits[i])

    @torch.no_grad()
    def add_query(self, query, future, past):
        self.queries.append(Query(query, future, past))

        if self.timer:
            self.timer.cancel()
            self.timer = None
        if len(self.queries) >= self.batch_size:
            self.batch_evaluate_queries()
        else:
            self.timer = asyncio.get_running_loop().call_later(
                self.timeout, lambda: self.batch_evaluate_queries()
            )

    def walk_cache(self, token_ids):
        # Walk while tokens can be found
        node = self.cache
        next_token_index = 1

        past = None
        base = 0
        while next_token_index < len(token_ids):
            if node.past_key_values is not None:
                past = node.past_key_values
                base = next_token_index
            if node.has_token(token_ids[next_token_index]):
                node = node.get_token(token_ids[next_token_index])
                next_token_index += 1
            else:
                break

        return node, next_token_index, past, base

    @torch.no_grad()
    async def next_token_logprobs(self, token_ids):
        """Request log probabilities of next token. This version is asynchronous because it automatically batches concurrent requests; use with `await`.

        Args:
            token_ids (list[int]): a list of token ids starting with `tokenizer.bos_token_id`, representing a prompt to the language model.

        Returns:
            logprobs (numpy.array): a numpy array of `len(vocab)`, with the language model's log (normalized) probabilities for the next token following the prompt.
        """

        # Ensure that token list begins with BOS
        assert token_ids[0] == self.tokenizer.bos_token_id

        node, next_token_index, past, base = self.walk_cache(token_ids)

        # If we processed all tokens, then we're done.
        if next_token_index == len(token_ids):
            return node.logprobs

        # Create a future with the prompt
        future = asyncio.get_running_loop().create_future()
        self.add_query(token_ids[base:], future, past)
        logits = await future

        # Create new nodes
        node = node.extend_cache(next_token_index, token_ids, logits, base)

        return node.logprobs

    @torch.no_grad()
    def next_token_logprobs_unbatched(self, token_ids):
        """Request log probabilities of next token. Not asynchronous, and does not support auto-batching.

        Args:
            token_ids (list[int]): a list of token ids starting with `tokenizer.bos_token_id`, representing a prompt to the language model.

        Returns:
            logprobs (numpy.array): a numpy array of `len(vocab)`, with the language model's log (normalized) probabilities for the next token following the prompt.
        """

        # Ensure that token list begins with BOS
        assert token_ids[0] == self.tokenizer.bos_token_id

        # Walk while tokens can be found
        node, next_token_index, past, base = self.walk_cache(token_ids)

        if next_token_index == len(token_ids):
            return node.logprobs

        logits = self.model(
            torch.tensor([token_ids[base:]]).to(self.device),
            past_key_values=node.past_key_values,
            use_cache=node.past_key_values is not None,
        ).logits[0]

        node = node.extend_cache(next_token_index, token_ids, logits, base)

        return node.logprobs
