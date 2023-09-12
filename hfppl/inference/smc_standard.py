import copy
from ..util import logsumexp
import numpy as np
import asyncio

async def smc_standard_async(model, n_particles, ess_threshold=0.5):
    """
    Standard sequential Monte Carlo algorithm with multinomial resampling
    and autobatching of LLM calls.
    
    Args:
        model (hfppl.model.Model): The model to perform inference on. Its `step` method must be `async`.
        n_particles (int): Number of particles to execute concurrently.
        ess_threshold (float): Effective sample size below which resampling is triggered, given as a fraction of `n_particles`.
    
    Returns:
        particles (list[hfppl.model.Model]): The completed particles after inference.
    """
    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    weights = [0.0 for _ in range(n_particles)]
    
    while (any(map(lambda p: not p.done_stepping(), particles))):
        # Step each particle
        for p in particles:
            p.untwist()
        await asyncio.gather(*[p.step() for p in particles if not p.done_stepping()])
        
        # Normalize weights
        W = np.array([p.weight for p in particles])
        w_sum = logsumexp(W)
        normalized_weights = W - w_sum
        
        # Resample if necessary
        if -logsumexp(normalized_weights * 2) < np.log(ess_threshold) + np.log(n_particles):
            # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
            probs = np.exp(normalized_weights)
            particles = [copy.deepcopy(particles[np.random.choice(range(len(particles)), p=probs)]) for _ in range(n_particles)]
            avg_weight = w_sum - np.log(n_particles)
            for p in particles:
                p.weight = avg_weight
        
    return particles

def smc_standard(model, n_particles, ess_threshold=0.5):
    """
    Standard sequential Monte Carlo algorithm with multinomial resampling,
    and **no autobatching** of LLM calls.
    
    Args:
        model (hfppl.model.Model): The model to perform inference on. Its `step` method must not be `async`.
        n_particles (int): Number of particles to execute concurrently.
        ess_threshold (float): Effective sample size below which resampling is triggered, given as a fraction of `n_particles`.
    
    Returns:
        particles (list[hfppl.model.Model]): The completed particles after inference.
    """
    
    # Create n_particles copies of the model
    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    weights = [0.0 for _ in range(n_particles)]

    while any(map(lambda p: not p.done_stepping(), particles)):
        # Step each particle
        for i, p in enumerate(particles):
            p.untwist()
            if not p.done_stepping():
                p.step()
            print(f"Particle {i}: {p} (weight {p.weight})")

        # Normalize weights
        W = np.array([p.weight for p in particles])
        w_sum = logsumexp(W)
        normalized_weights = W - w_sum
        
        # Resample if necessary
        if -logsumexp(normalized_weights * 2) < np.log(ess_threshold) + np.log(n_particles):
            # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
            probs = np.exp(normalized_weights)
            particles = [copy.deepcopy(particles[np.random.choice(range(len(particles)), p=probs)]) for _ in range(n_particles)]
            avg_weight = w_sum - np.log(n_particles)
            for p in particles:
                p.weight = avg_weight

    # Return the particles
    return particles