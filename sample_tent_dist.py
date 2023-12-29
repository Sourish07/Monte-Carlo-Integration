import numpy as np
import matplotlib.pyplot as plt

def tent_pdf(x):
    if 0 < x <= 1:
        return x
    elif 1 < x <= 2:
        return -x + 2
    else:
        return 0.000000000001

def inv_cdf(x):
    if x < 0.5:
        return np.sqrt(2*x)
    else:
        return 2 - np.sqrt(2 - 2*x)
    
def sample_tent_dist():
    # np.random.random() returns a random number between 0 and 1
    return inv_cdf(np.random.random())
    
if __name__=="__main__":
    NUM_OF_SAMPLES = 100_000

    samples = [sample_tent_dist() for _ in range(NUM_OF_SAMPLES)]

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
