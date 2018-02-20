import numpy as np
from scipy.special import gamma
import matplotlib.pyplot as plt


def func_s(d):
    return 2*np.power(np.pi, d/2) / gamma(d/2)


def prob_density(d):
    def density_under_sigma(sigma):
        first = (func_s(d) / (np.sqrt(2 * np.pi) * sigma))
        second = np.power((d-1) / 2*np.pi*np.e, (d-1)/2)
        return first * second
    return density_under_sigma

def plot():
    x = np.arange(0.1, 1, 0.01)
    plt.figure(1)
    d_range = [1, 2, 5, 10, 20]
    for index in range(len(d_range)):
        d_value = d_range[index]
        plt.subplot(321 + index)
        plt.plot(x, prob_density(d_value)(x))
        plt.title("d = {0}".format(d_value))

    plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

    plt.show()

if __name__ == '__main__':
    plot()