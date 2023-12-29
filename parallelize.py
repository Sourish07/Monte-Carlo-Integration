from joblib import Parallel, delayed
import os
NUM_OF_PROCESSES = os.cpu_count()


def parallelize(estimator, num_of_samples):
    samples_per_process = [num_of_samples // NUM_OF_PROCESSES] * NUM_OF_PROCESSES
    remaining_samples = num_of_samples % NUM_OF_PROCESSES

    for i in range(remaining_samples):
        samples_per_process[i] += 1

    assert sum(samples_per_process) == num_of_samples

    with Parallel(n_jobs=NUM_OF_PROCESSES) as parallel:
        estimates = parallel(delayed(estimator)(s) for s in samples_per_process)

    weighted_sum = sum(estimate * samples for estimate, samples in zip(estimates, samples_per_process))
    combined_estimate = weighted_sum / num_of_samples
    return combined_estimate

    