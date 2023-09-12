# Auto-Batching

If running in a GPU-accelerated environment, you will probably want to exploit **auto-batching**.

## What is auto-batching?

The `step` method of a LLaMPPL model describes how to advance a *single* particle one step of generation. 
But inference methods must maintain many particles at once. 
By default, this is achieved by running each particle's `step` method *sequentially*,
which results in executing separate calls to the large language model for each particle.
This fails to fully exploit the GPU's parallelism.

With auto-batching, LLaMPPL will execute particles' `step` methods concurrently, and automatically batch calls
to large language models. This batching is handled by the `CachedCausalLM` object, and its behavior is controlled by two parameters:

* `lm.batch_size`: the maximum number of requests to batch. The default value is 20.
* `lm.timeout`: if `lm.timeout` seconds pass with no new request, the current batch is processed even if not full. The default value is 0.02.

## Enabling auto-batching for your LLaMPPL model

To exploit auto-batching, your model must:

1. **Define the [`Model.step`][hfppl.Model.step] method as `#!python async`.**  
   <br>
   When overriding [`Model.step`][hfppl.Model.step], write `#!python async def step(self):` instead of `#!python def step(self):`.  
   <br>

2. **Use [`Model.sample_async`][hfppl.Model.sample_async] and [`Model.observe_async`][hfppl.Model.observe_async] within the body of [`Model.step`][hfppl.Model.step], with the `#!python await` keyword.**  
   <br>
   Instead of `#!python self.sample(dist[, proposal])`, use `#!python await self.sample_async(dist[, proposal])`, and instead of
   `#!python self.observe(dist, val)`, use `#!python await self.observe_async(dist, val)`. Technically, you 
   only need to do this when `dist` (or `proposal`) might involve calls to the large language model (e.g.,
   the `Transformer` distribution, or the `#!python StatefulLM.next_token()` distribution), but
   it doesn't hurt to use it elsewhere too, if you're unsure.  
   <br>
   If your model manually queries the next-token probabilities of a language model, using `#!python lm.next_token_logprobs`,
   you should change `#!python lm.next_token_logprobs(prompt)` to `#!python await lm.next_token_logprobs_async(prompt)`.
   If your `step` method calls helper methods that use `#!python self.sample_async`, 
   `#!python self.observe_async`, or `#!python lm.next_token_logprobs_async`, those methods will also need to be declared `#!python async`,
   and called with `#!python await`. 
   <br>


In addition, the script that runs inference must call `#!python await smc_standard_async(model, num_particles)` instead of `#!python smc_standard(model, num_particles)`.
In Jupyter notebooks, a cell with a top-level `#!python await` can be executed directly. In stand-alone scripts, create an `#!python async def main():` method that calls `#!python smc_standard_async` with `#!python await`, and run it using `#!python asyncio.run(main())`.
   
You may want to set the batch size (`#!python lm.batch_size`) to the number of particles you are using (if the number of particles is not too large).