"""Provides inference methods for use with LLaMPPL models.

This module currently provides the following inference methods:

* `smc_standard(model, num_particles, ess_threshold=0.5)`: Standard SMC with multinomial resampling.

* `smc_standard_async(model, num_particles, ess_threshold=0.5)`: Standard SMC with multinomial resampling, with support for autobatching.

* `smc_steer(model, num_beams, num_expansions)`: a without-replacement SMC algorithm that resembles beam search.

* `smc_steer_async(model, num_beams, num_expansions)`: a without-replacement SMC algorithm that resembles beam search, with support for autobatching."""

from .smc_standard import smc_standard 
from .smc_standard import smc_standard_async
from .smc_steer import smc_steer
from .smc_steer import smc_steer_async