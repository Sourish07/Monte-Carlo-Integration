import matplotlib.pyplot as plt
import numpy as np

def bimodal_pdf(x):
    c = 1 / (2 * 0.704215)
    j = 1.4
    k = -1.4
    n = 0.9
    o = 1
    l = 1.1

    return c * (-((j*x+k)+n) * ((j*x+k)-n) * ((j*x+k)+o) * ((j*x+k)-o)+l )

f_left_bound = 0
f_right_bound = 2
interval_width = f_right_bound - f_left_bound
# represents the height of our function in the range [0, 1]
FUNCTION_MAX = np.max([bimodal_pdf(x) for x in np.arange(f_left_bound, f_right_bound, 0.0001)])

def sample_binomal_dist():
    while True:
        x = np.random.random() * interval_width - f_left_bound
        y = np.random.random() * FUNCTION_MAX
        if y < bimodal_pdf(x):
            return x

if __name__=="__main__":
    NUM_OF_SAMPLES = 100_000

    samples = [sample_binomal_dist() for _ in range(NUM_OF_SAMPLES)]

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
    # plt.show()
    plt.savefig('rejection_method.png', dpi=240, pad_inches=0.5)