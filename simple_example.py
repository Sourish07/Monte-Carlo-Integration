import random
import math
from tqdm import trange

from parallelize import parallelize

NUM_OF_SAMPLES = 1_000_000

def estimate_pi(idx, num_of_samples):
    half_circle = lambda x: math.sqrt(1-x**2)

    domain_a = -1 # Left bound of domain
    domain_b = 1 # Right bound of domain
    size_of_domain = domain_b - domain_a

    mc_estimate = 0
    for _ in trange(num_of_samples, position=idx):
        # Uniformly sampling our domain
        rand_x = random.uniform(domain_a, domain_b)
        # Area of rectangle is size_of_domain * value of function at rand_x
        mc_estimate += half_circle(rand_x) * size_of_domain 

    mc_estimate /= num_of_samples
    return mc_estimate * 2 # Multiply by 2 to get full circle

if __name__ == '__main__':
    estimate = parallelize(estimate_pi, NUM_OF_SAMPLES)
    print(f'Estimate of pi: {estimate}')
