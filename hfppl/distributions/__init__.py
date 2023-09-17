"""Exposes distributions for use with `sample`, `observe`, and `intervene` methods in LLaMPPL models.

Currently supported distributions:

* `Bernoulli(p: float) -> bool`
* `Geometric(p: float) -> int`
* `LogCategorical(logits: array) -> int`
* `TokenCategorical(lm: hfppl.llms.CachedCausalLM, logits: array) -> hfppl.llms.Token`
* `Transformer(lm: hfppl.llms.CachedCausalLM) -> hfppl.llms.Token`
* `LMContext(lm: hfppl.llms.CachedCausalLM, prompt: list[int]).next_token() -> hfppl.llms.Token`
* `LMContext(lm: hfppl.llms.CachedCausalLM, prompt: list[int]).mask_dist(mask: set[int]) -> bool`
"""

from .distribution import Distribution
from .geometric import Geometric
from .logcategorical import LogCategorical
from .tokencategorical import TokenCategorical
from .transformer import Transformer
from .lmcontext import LMContext
from .bernoulli import Bernoulli