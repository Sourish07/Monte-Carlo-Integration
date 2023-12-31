import matplotlib.pyplot as plt
import numpy as np
import time

def bimodal_pdf(x):
    p, q, r, s, t, u = 0.7100104371534262, 1.4, -1.4, 0.9, 1, 1.1
    a = q*x+r
    return p * (-(a+s) * (a-s) * (a+t) * (a-t)+u )

bimodal_pdf_vec = bimodal_pdf # Just for naming consistency since the function works with scalers and arrays

f_left_bound = 0
f_right_bound = 2
interval_width = f_right_bound - f_left_bound
# represents the height of our function in the range [0, 1]
FUNCTION_MAX = np.max(bimodal_pdf(np.arange(f_left_bound, f_right_bound, 0.0001)))

def sample_binomal_dist():
    # Keep on genrating samples until one is under the curve
    while True:
        x = np.random.random() * interval_width - f_left_bound
        y = np.random.random() * FUNCTION_MAX
        if y < bimodal_pdf(x):
            return x
        
def sample_binomal_dist_vec(num_samples):
    accepted_samples = None
    while accepted_samples is None or len(accepted_samples) < num_samples:
        x_samples = np.random.random(num_samples * 2) * interval_width - f_left_bound
        y_samples = np.random.random(num_samples * 2) * FUNCTION_MAX
        
        new_accepted = x_samples[y_samples < bimodal_pdf(x_samples)]
        if accepted_samples is None:
            accepted_samples = new_accepted
        else:
            accepted_samples = np.concatenate((accepted_samples, new_accepted))
    return accepted_samples[:num_samples]


if __name__=="__main__":
    s = time.time()
    NUM_OF_SAMPLES = 10_000_000

    samples = sample_binomal_dist_vec(NUM_OF_SAMPLES)

    plt.style.use('./sourish.mplstyle')
    plt.figure(figsize=(16, 9))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.title('Sampling a Bimodal Distribution with the Rejection Method', size=30, pad=25)

    plt.xlabel('X value', size=20, labelpad=25)
    plt.ylabel('Y value', size=20, labelpad=25)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.gcf().subplots_adjust(bottom=0.15)

    plt.hist(samples, bins=250, color="#df3b43", density=True)
    plt.savefig('rejection_method.png', dpi=240, pad_inches=0.5)
