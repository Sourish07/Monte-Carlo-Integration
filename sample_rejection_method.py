import matplotlib.pyplot as plt
import numpy as np

def bimodal_func(x):
    c = 1 / 0.704215
    j = 2.8
    k = -1.4
    n = 0.9
    o = 1
    l = 1.1

    return c * (-((j*x+k)+n) * ((j*x+k)-n) * ((j*x+k)+o) * ((j*x+k)-o)+l )

f_left_bound = 0
f_right_bound = 1
interval_width = f_right_bound - f_left_bound
# represents the height of our function in the range [0, 1]
FUNCTION_MAX = np.max([bimodal_func(x) for x in np.arange(f_left_bound, f_right_bound, 0.0001)])

def sample_binomal_func():
    while True:
        x = np.random.random() * interval_width - f_left_bound
        y = np.random.random() * FUNCTION_MAX
        if y < bimodal_func(x):
            return x


NUM_OF_SAMPLES = 100_000

samples = [sample_binomal_func() for _ in range(NUM_OF_SAMPLES)]

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