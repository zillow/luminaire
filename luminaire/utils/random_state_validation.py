import numpy as np
import numbers

def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance

    :param int seed: seed for the random state
    :return: None, int or instance of RandomState
             If seed is None, return the RandomState singleton used by np.random.
             If seed is an int, return a new RandomState instance seeded with seed.
             If seed is already a RandomState instance, return it.
             Otherwise raise ValueError.
    :rtype: np.random.RandomState or None
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )