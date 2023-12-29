import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm, trange

NUM_OF_SAMPLES = 25

def curse(dim_num):
    points = np.random.rand(NUM_OF_SAMPLES, dim_num)

    dist = 0
    counter = 0
    for i in range(NUM_OF_SAMPLES - 1):
        for j in range(i + 1, NUM_OF_SAMPLES):
            dist += np.linalg.norm(np.array(points[i]) - np.array(points[j]))
            counter += 1

    return dist / counter

if __name__ == '__main__':
    plt.style.use('./sourish.mplstyle')
    
    plt.figure(figsize=(16, 9))
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.title('Curse of Dimensionality | Dimension vs Sparsity of Data', size=30, pad=25)

    plt.xlabel('Dimension', size=20, labelpad=25)
    plt.ylabel('Average Distance Between Points', size=20, labelpad=25)

    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    dists = [curse(i) for i in range(1, 100)]

    plt.plot(dists, color="#df3b43")
    plt.gcf().subplots_adjust(bottom=0.15)

    plt.savefig('curse_of_dimensionality.png', dpi=240, pad_inches=0.5)
    