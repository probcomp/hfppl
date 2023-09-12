# Immutability

When a particle is promising, the sequential Monte Carlo algorithm may _clone_ it, by calling `copy.deepcopy`. 

Depending on your model, this may be more or less expensive. 

To make it faster, override the `immutable_properties(self)` method of your Model class, to return a `set[str]` of property names that are guaranteed not to change during `step`. For all properties in this set, LLaMPPL will use shared memory across particles, and avoid copying when cloning particles.