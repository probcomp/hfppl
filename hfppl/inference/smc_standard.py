import copy
from ..util import logsumexp
import numpy as np
import asyncio
from .smc_record import SMCRecord


async def smc_standard(model, n_particles, ess_threshold=0.5, record=False):
    """
    Standard sequential Monte Carlo algorithm with multinomial resampling.

    Args:
        model (hfppl.modeling.Model): The model to perform inference on.
        n_particles (int): Number of particles to execute concurrently.
        ess_threshold (float): Effective sample size below which resampling is triggered, given as a fraction of `n_particles`.

    Returns:
        particles (list[hfppl.modeling.Model]): The completed particles after inference.
    """
    particles = [copy.deepcopy(model) for _ in range(n_particles)]
    history = SMCRecord(n_particles) if record else None

    ancestor_indices = list(range(n_particles))
    did_resample = False
    while any(map(lambda p: not p.done_stepping(), particles)):
        # Step each particle
        for p in particles:
            p.untwist()
        await asyncio.gather(*[p.step() for p in particles if not p.done_stepping()])

        # Record history
        if record:
            if len(history.history) == 0:
                history.add_init(particles)
            elif did_resample:
                history.add_resample(ancestor_indices, particles)
            else:
                history.add_smc_step(particles)

        # Normalize weights
        W = np.array([p.weight for p in particles])
        w_sum = logsumexp(W)
        normalized_weights = W - w_sum

        # Resample if necessary
        if -logsumexp(normalized_weights * 2) < np.log(ess_threshold) + np.log(
            n_particles
        ):
            # Alternative implementation uses a multinomial distribution and only makes n-1 copies, reusing existing one, but fine for now
            probs = np.exp(normalized_weights)
            ancestor_indices = [
                np.random.choice(range(len(particles)), p=probs)
                for _ in range(n_particles)
            ]

            if record:
                # Sort the ancestor indices
                ancestor_indices.sort()

            particles = [copy.deepcopy(particles[i]) for i in ancestor_indices]
            avg_weight = w_sum - np.log(n_particles)
            for p in particles:
                p.weight = avg_weight

            did_resample = True
        else:
            did_resample = False

    if record:
        return particles, history

    return particles
