# Improving performance of LLaMPPL models

If your LLaMPPL model is running slowly, consider exploiting the following features to improve performance:

- [Auto-Batching](batching.md) — to run multiple particles concurrently, with batched LLM calls
- [Caching](caching.md) - to cache key and value vectors for long prompts
- [Immutability hinting](immutability.md) - to significantly speed up the bookkeeping performed by SMC inference
