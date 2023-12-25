import os
from multiprocessing import Pool, RLock, freeze_support
from tqdm import tqdm

NUM_OF_PROCESSES = os.cpu_count()

def parallelize(estimator, num_of_samples):
    freeze_support()  # for Windows support
    tqdm.set_lock(RLock())  # for managing output contention

    samples_per_process = [num_of_samples // NUM_OF_PROCESSES] * NUM_OF_PROCESSES
    # Distribute the remaining samples evenly
    remaining_samples = num_of_samples % NUM_OF_PROCESSES

    for i in range(remaining_samples):
        samples_per_process[i] += 1

    assert sum(samples_per_process) == num_of_samples

    with Pool(NUM_OF_PROCESSES, initializer=tqdm.set_lock, initargs=(tqdm.get_lock(),)) as p:
        estimates = p.starmap(estimator, enumerate(samples_per_process))

    # Weighted average of estimates
    weighted_sum = sum(estimate * samples for estimate, samples in zip(estimates, samples_per_process))
    combined_estimate = weighted_sum / num_of_samples
    return combined_estimate
    
