import numpy as np
import matplotlib.pyplot as plt
import time

def tent_pdf(x):
    if x == 0:
        return 1e-14 # Avoid divide by zero error
    elif 0 < x <= 1:
        return x
    elif 1 < x <= 2:
        return -x + 2
    else:
        return 0 # Outside of domain
    
def tent_pdf_parallel(x):
    equal_to_zero = x == 0
    # between_zero_and_one = (0 < x) & (x <= 1) # Don't need because we don't change the value
    between_one_and_two = (1 < x) & (x <= 2)
    greater_than_two = x > 2

    x[equal_to_zero] = 1e-14
    # x[between_zero_and_one] = x[between_zero_and_one]
    x[between_one_and_two] = -x[between_one_and_two] + 2
    x[greater_than_two] = 0
    return x

def inv_tent_cdf(x):
    if x < 0.5:
        return np.sqrt(2*x)
    else:
        return 2 - np.sqrt(2 - 2*x)
    
def inv_tent_cdf_parallel(x):
    less_than_half = x < 0.5
    greater_than_half = ~less_than_half

    x[less_than_half] = np.sqrt(2*x[less_than_half])
    x[greater_than_half] = 2 - np.sqrt(2 - 2*x[greater_than_half])
    return x
    
def sample_tent_dist():
    # np.random.random() returns a random number between 0 and 1
    return inv_tent_cdf(np.random.random())

def sample_tent_dist_parallel(num_samples):
    return inv_tent_cdf_parallel(np.random.random(num_samples))
    
if __name__=="__main__":
    s = time.time()
    NUM_OF_SAMPLES = 100_000

    # samples = [sample_tent_dist() for _ in range(NUM_OF_SAMPLES)]
    print(time.time() - s)
    s = time.time()
    samples = sample_tent_dist_parallel(NUM_OF_SAMPLES)
    print(time.time() - s)

    plt.style.use('./sourish.mplstyle')
    plt.figure(figsize=(16, 9))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.title('Sampling a Tent Distribution with the Inversion Method', size=30, pad=25)

    plt.xlabel('X value', size=20, labelpad=25)
    plt.ylabel('Y value', size=20, labelpad=25)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.gcf().subplots_adjust(bottom=0.15)

    plt.hist(samples, bins=250, color="#df3b43", density=True)
    # plt.show()
    plt.savefig('inversion_method.png', dpi=240, pad_inches=0.5)
