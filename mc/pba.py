
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

def pba(noisy_oracle,
        start,
        stop,
        p=0.6,
        max_iterations=1000,
        early_termination_width=0):
    """Query the noisy_oracle at the median of the current belief
    distribution, then update the belief accordingly.  Start from a
    uniform belief over the search_interval, and repeat n_iterations
    times.

    Parameters
    ----------

        noisy_oracle : stochastic function that accepts a float and
                       returns a bool we assume that
                       E[noisy_oracle(x)] is non-decreasing in x, and
                       crosses 0.5 within the search_interval start
        start : lower bound of search interval
        stop : upper bound of search_interval
        p : float assumed constant known probability of correct
            responses from noisy_oracle (must be > 0.5)
        max_iterations : int maximum number of times to query the
                         noisy_oracle
        early_termination_width : float if 95% of our belief is in an
                                  interval of this width or smaller,
                                  stop early

    Returns
    -------
        x : numpy.ndarray
            discretization of search_interval
        zs : list of bools
             oracle responses after each iteration
        fs : list of numpy.ndarrays
             belief pdfs after each iteration, including initial belief pdf
    References
    ----------
    [1] Bisection search with noisy responses (Waeber et al., 2013)
        http://epubs.siam.org/doi/abs/10.1137/120861898
    Notes
    -----
        For convenience / clarity / ease of implementation, we
        represent the belief pdf numerically, by uniformly
        discretizing the search_interval. This puts a cap on precision
        of the solution, which could be reached in as few as
        log_2(resolution) iterations (in the noiseless case). It is
        also wasteful of memory.  Later, it would be better to
        represent the belief pdf using the recursive update equations,
        but I haven't yet figured out how to use them to find the
        median efficiently.
    """

    if p <= 0.5:
        raise (ValueError('the probability of correct responses must be > 0.5'))

    # initialize a uniform belief over the search interval
    x = np.linspace(start, stop,100)
    f = np.ones(len(x))
    f /= np.sum(f)

    f = np.log(f)


    # initialize empty list of oracle responses
    zs = []

    def get_median(f):
        exp_f = np.exp(f)
        alpha = exp_f.sum() * 0.5

        # Finding the median of the distribution requires
        # adding together many very small numbers, so it's not
        # very stable. In part, we address this by randomly
        # approaching the median from below or above.
        if random.choice([True, False]):
            return x[exp_f.cumsum() < alpha][-1]
        else:
            return x[::-1][exp_f[::-1].cumsum() < alpha][-1]
            
    def get_belief_interval(f, fraction=0.95):
        exp_f = np.exp(f)

        eps = 0.5 * (1 - fraction)
        eps = exp_f.sum() * eps

        left = x[exp_f.cumsum() < eps][-1]
        right = x[exp_f.cumsum() > (exp_f.sum() - eps)][0]
        return left, right

    def describe_belief_interval(f, fraction=0.95):
        median = get_median(f)
        left, right = get_belief_interval(f, fraction)
        description = "median: {}, {}% belief interval: ({}, {})".format(
            median, fraction * 100, left, right)
        return description

    for _ in range(max_iterations):

        # query the oracle at median of previous belief pdf
        median = get_median(f)
        z = noisy_oracle(median)
        zs.append(z)

        if z > 0:  # to handle noisy_oracles that return
                   # bools or binary
            f[x >= median] += np.log(p)
            f[x < median] += np.log(1 - p)
        else:
            f[x >= median] += np.log(1 - p)
            f[x < median] += np.log(p)

        # shift distribution to avoid underflow
        f -= np.max(f)

        logging.info(describe_belief_interval(f))

        belief_interval = get_belief_interval(f, 0.95)
        if (belief_interval[1] - belief_interval[0]) <= early_termination_width:
            break

    return x, zs, f