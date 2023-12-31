import numpy as np
from matplotlib import pyplot as plt
import scipy.integrate as integrate

from sample_tent_dist import tent_pdf, sample_tent_dist, tent_pdf_vec, sample_tent_dist_vec
from sample_bimodal_dist import bimodal_pdf, sample_binomal_dist, bimodal_pdf_vec, sample_binomal_dist_vec
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


def estimate_f_vec(pdf, sample_pdf, num_of_samples):
    # pdf returns the value of the pdf at parameter x
    # sample_pdf is used to sample from the pdf

    if num_of_samples == 0:
        return 0

    # 1. Sample from our PDF
    samples_initial = sample_pdf(num_of_samples)

    # 2. Evaluate our function that we're trying to integrate at each of the samples
    samples_f = f(samples_initial)

    # 3. Evaluate the PDF at each of the samples to get the probability of each sample
    samples_p = pdf(samples_initial)

    # 4. Divide the function evaluations by the PDF evaluations weight each sample by its probablity accordingly
    samples = samples_f / samples_p

    # 5. Return the average of the weighted samples
    return np.sum(samples) / num_of_samples


if __name__ == '__main__':
    f_left_bound = 0
    f_right_bound = 2
    interval_width = f_right_bound - f_left_bound
    true_answer = integrate.quad(f, f_left_bound, f_right_bound)[0]

    # Tuples of the form (pdf, sample_pdf), i.e. the pdf and the function that allows us to sample from the pdf
    # uniform_distribution = (lambda x: 1 / interval_width, lambda: np.random.random() * interval_width - f_left_bound)
    uniform_distribution = (lambda x: np.ones_like(x) / interval_width, lambda num_samples: np.random.random(num_samples) * interval_width - f_left_bound)
    tent_distribution = (tent_pdf_vec, sample_tent_dist_vec)
    bimodal_distribution = (bimodal_pdf_vec, sample_binomal_dist_vec)

    num_samples_list = []

    uniform_dist_estimates = []

    tent_dist_estimates = []

    bimodal_dist_estimates = []

    for i in range(1, 400):
        # The sqrt is to have less sparse points on the graph, but still keep the exponential growth
        num_of_samples = int(2 ** np.sqrt(i))
        num_samples_list.append(num_of_samples)

        print(f"Iteration {i}, Number of samples: {num_of_samples}")
        uniform_pdf_wrapper = lambda s: estimate_f_vec(*uniform_distribution, num_of_samples=s)
        uniform_pdf_estimate = parallelize(uniform_pdf_wrapper, num_of_samples)
        # uniform_pdf_estimate = estimate_f(*uniform_distribution, num_of_samples=num_of_samples)
        uniform_dist_estimates.append(uniform_pdf_estimate)

        tent_pdf_wrapper = lambda s: estimate_f_vec(*tent_distribution, num_of_samples=s)
        tent_pdf_est = parallelize(tent_pdf_wrapper, num_of_samples)
        # tent_pdf_est = estimate_f(*tent_distribution, num_of_samples=num_of_samples)
        tent_dist_estimates.append(tent_pdf_est)

        bimodal_pdf_wrapper = lambda s: estimate_f_vec(*bimodal_distribution, num_of_samples=s)
        bimodal_pdf_est = parallelize(bimodal_pdf_wrapper, num_of_samples)
        # bimodal_pdf_est = estimate_f(*bimodal_distribution, num_of_samples=num_of_samples)
        bimodal_dist_estimates.append(bimodal_pdf_est)

    error_threshold = 9e-5
    num_samples_list = np.array(num_samples_list)

    uniform_dist_estimates = np.array(uniform_dist_estimates)
    uniform_error_threshold = np.argmax(np.abs(uniform_dist_estimates - true_answer) < error_threshold)
    # a = np.cumprod((np.abs(uniform_dist_estimates - true_answer) < error_threshold)[::-1])[::-1]
    # uniform_error_threshold = num_samples_list[np.argmax(a)]
    
    tent_dist_estimates = np.array(tent_dist_estimates)
    tent_error_threshold = np.argmax(np.abs(tent_dist_estimates - true_answer) < error_threshold)
    # a = np.cumprod((np.abs(tent_dist_estimates - true_answer) < error_threshold)[::-1])[::-1]
    # tent_error_threshold = num_samples_list[np.argmax(a)]

    bimodal_dist_estimates = np.array(bimodal_dist_estimates)
    bimodal_error_threshold = np.argmax(np.abs(bimodal_dist_estimates - true_answer) < error_threshold)
    # a = np.cumprod((np.abs(bimodal_dist_estimates - true_answer) < error_threshold)[::-1])[::-1]
    # bimodal_error_threshold = num_samples_list[np.argmax(a)]

    plt.style.use('./sourish.mplstyle')
    plt.figure(figsize=(16, 9))
    
    plt.xscale('log')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.title('Sampling Method vs Estimator Performance', size=30, pad=25)

    plt.xlabel('Number of Samples', size=20, labelpad=25)
    plt.ylabel('Estimate of Integral', size=20, labelpad=25)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.gcf().subplots_adjust(bottom=0.15)

    plt.plot(num_samples_list, bimodal_dist_estimates, label="Bimodal Distribution", color='#df3b43')
    # plot vertical line where error threshold is reached
    plt.axvline(bimodal_error_threshold, color='#df3b43', linestyle='--')

    plt.plot(num_samples_list, uniform_dist_estimates, label="Uniform Distribution", color='#21cda2')
    plt.axvline(uniform_error_threshold, color='#21cda2', linestyle='--')

    plt.plot(num_samples_list, tent_dist_estimates, label="Tent Distribution", color='#3673d6')
    plt.axvline(tent_error_threshold, color='#3673d6', linestyle='--')
    
    plt.axhline(true_answer, label="True Answer", color='white', linestyle='--')
    plt.legend(labelcolor='#FFFFFF', fontsize=20)
    plt.savefig("monte_carlo.png", dpi=240, pad_inches=0.5)