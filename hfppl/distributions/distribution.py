class Distribution:
    """Abstract base class for a distribution."""

    def log_prob(self, x):
        """Compute the log probability of a value under this distribution,
        or the log probability density if the distribution is continuous.
        
        Args:
            x: the point at which to evaluate the log probability.
        Returns:
            logprob (float): the log probability of `x`."""
        raise NotImplementedError()
    
    def sample(self):
        """Generate a random sample from the distribution.
        
        Returns:
            x: a value randomly sampled from the distribution."""
        raise NotImplementedError()
        
    async def sample_async(self):
        """Generate a random sample from the distribution.
        
        This method is asynchronous, and so the sampling process can make (auto-batched, asynchronous) queries to `CachedCausalLM` objects.
        If not overridden by subclasses, defaults to calling `self.sample()`.
        
        Returns:
            x: a value randomly sampled from the distribution."""
        return self.sample()
    
    async def log_prob_async(self, x):
        """Compute the log probability of a value under this distribution,
        or the log probability density if the distribution is continuous.
        
        This method is asynchronous, and so the probability computation can make (auto-batched, asynchronous) queries to `CachedCausalLM` objects.
        If not overridden by subclasses, defaults to calling `self.log_prob(x)`.
        
        Args:
            x: the point at which to evaluate the log probability.
        Returns:
            logprob (float): the log probability of `x`."""        
        return self.log_prob(x)
    
    def argmax(self, n):
        """Return the nth most probable outcome under this distribution (assuming this is a discrete distribution).
        
        Args:
            n (int): which value to return to, indexed from most probable (n=0) to least probable (n=|support|).
        Returns:
            x: the nth most probable outcome from this distribution."""
        raise NotImplementedError()