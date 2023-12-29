import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate

from sample_tent_dist import tent_pdf, sample_tent_dist
from sample_bimodal_dist import bimodal_pdf, sample_binomal_dist
from parallelize import parallelize

def f(x):
    j = -0.9
    k = -1.5
    l = 1.14
    m = 0.5
    n = 1.593
    o = -1.594

    a = n * x + o
    return -(a + j) * (a - j) * (a + k) * (a - k) * (a + l) * (a - l) + m


def estimate_f(pdf, sample_pdf, num_of_samples):
    # pdf returns the value of the pdf at parameter x
    # sample_pdf is used to sample from the pdf

    if num_of_samples == 0:
        return 0

    # 1. Sample from our PDF
    samples_initial = [sample_pdf() for _ in range(num_of_samples)]

    # 2. Evaluate our function that we're trying to integrate at each of the samples
    samples_f = np.array([f(x) for x in samples_initial])

    # 3. Evaluate the PDF at each of the samples to get the probability of each sample
    samples_p = np.array([pdf(x) for x in samples_initial])

    # 4. Divide the function evaluations by the PDF evaluations weight each sample by its probablity accordingly
    samples = samples_f / samples_p

    # 5. Return the average of the weighted samples
    return np.sum(samples) / num_of_samples

if __name__ == '__main__':
    f_left_bound = 0
    f_right_bound = 2
    interval_width = f_right_bound - f_left_bound
    true_answer = integrate.quad(f, f_left_bound, f_right_bound)[0]

    # Tuples of the form (pdf, sample_pdf), i.e. the pdf function and the function that allows us to sample from the pdf
    uniform_distribution = (lambda x: 1 / interval_width, lambda: np.random.random() * interval_width - f_left_bound)
    tent_distribution = (tent_pdf, sample_tent_dist)
    bimodal_distribution = (bimodal_pdf, sample_binomal_dist)

    num_samples_list = []

    uniform_dist_estimates = []
    uniform_dist_error = []

    tent_dist_estimates = []
    tent_dist_error = []

    bimodal_dist_estimates = []
    bimodal_dist_error = []

    for i in range(1, 25):
        num_of_samples = 2 ** i
        num_samples_list.append(num_of_samples)

        print(f"Number of samples: {num_of_samples}")
        uniform_pdf_wrapper = lambda s: estimate_f(*uniform_distribution, num_of_samples=s)
        uniform_pdf_estimate = parallelize(uniform_pdf_wrapper, num_of_samples)
        # uniform_pdf_estimate = estimate_f(*uniform_distribution, num_of_samples=num_of_samples)
        uniform_dist_error.append(abs(true_answer - uniform_pdf_estimate))

        tent_pdf_wrapper = lambda s: estimate_f(*tent_distribution, num_of_samples=s)
        tent_pdf_est = parallelize(tent_pdf_wrapper, num_of_samples)
        # tent_pdf_est = estimate_f(*tent_distribution, num_of_samples=num_of_samples)
        tent_dist_error.append(abs(true_answer - tent_pdf_est))

        bimodal_pdf_wrapper = lambda s: estimate_f(*bimodal_distribution, num_of_samples=s)
        bimodal_pdf_est = parallelize(bimodal_pdf_wrapper, num_of_samples)
        # bimodal_pdf_est = estimate_f(*bimodal_distribution, num_of_samples=num_of_samples)
        bimodal_dist_error.append(abs(true_answer - bimodal_pdf_est))

    # plt.style.use('./sourish.mplstyle')
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(num_samples_list, uniform_dist_error, label="Uniform Distribution")
    plt.plot(num_samples_list, tent_dist_error, label="Tent Distribution")
    plt.plot(num_samples_list, bimodal_dist_error, label="Bimodal Distribution")
    plt.legend()
    plt.savefig("monte_carlo.png", dpi=240, pad_inches=0.5)