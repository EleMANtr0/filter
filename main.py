import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import random as rnd

class Cache:
    def __init__(self, df, n):
        self.df = df
        self.n = n
        self.cache = [0 for i in range(n)]
    def update_cache(self, value):
        self.cache.append(value)
        self.cache.pop(0)
    def update_cacheRMS(self, value):
        self.cache[-1] = value
    def get_cache(self,index):
        return self.cache[index]
    def rms(self):
        temp = 0
        for i in range(len(self.cache)):
            temp += self.cache[i] ** 2 * np.sign(self.cache[i])
        temp /= self.n
        rms_val = np.sqrt(abs(temp))*np.sign(temp)
        return rms_val

class File:
    def read_CSV(self, path):
        df = pd.read_csv(path)
        return df

class Filter:
    def __init__(self, df=pd.DataFrame):
        self.df = df
        df_col = df.columns
        self.angles = df[df_col[0]].values

    def RootMS(self, prm = 0.2):
        x = self.angles
        cache_obj = Cache(self.df, 50)
        x_rms = []
        j = 0
        delta = 0.0

        for i in range(len(cache_obj.df)):
            cache_obj.update_cache(x[i])
            temp_rms = cache_obj.rms()
            delta = abs(temp_rms - x[i])
            if abs(delta / temp_rms) <= prm:
                j += 1
                x_rms.append(x_rms[i - j])
                continue
            x_rms.append(temp_rms)
            j = 0
        self.f_plot(x, x_rms)
    def _f_plot(self, y1, y2 = [], x1 = [], labely = "angle", labelx = "t"):
        if not len(x1):
            x1 = self.df.index
        fig, ax = plt.subplots()
        ax.set_xlabel(labelx)
        ax.set_ylabel(labely)
        ax.plot(x1, y1, color='red', linewidth=0.5)
        if len(y2):
            ax.plot(x1, y2, c='blue', linewidth = 0.5)
        fig.show()
        # fig.savefig(r"C:\Users\EleMANtrO\Desktop\python\fig.svg", format='svg')

    def _f_scatter(self, y1, y2 = [], x1 = [], labely = "angle", labelx = "t"):
        if not len(x1):
            x1 = self.df.index
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.scatter(x = x1, y = y1, color='red', linewidth=0.5)
        if len(y2):
            plt.scatter(x = x1, y = y2, c='blue', linewidth = 0.5)
        plt.show()
        plt.grid(True)

    def _f_hist(self, y1,  labely="number", label="Gauss"):
        fig, ax = plt.subplots()
        ax.set_label(label)
        a, b, c = ax.hist(y1, bins = len(y1), edgecolor='black', linewidth=0.5, label=labely)
        ax.hist(y1, bins=len(y1), edgecolor='black', linewidth=0.5, label=labely)
        return a, b, c

    def gauss(self):
        x_an = self.angles
        hist_values, bin_edges, _ = self._f_hist(x_an)
        mids = []
        for i in range(len(bin_edges) - 1):
            mids.append((bin_edges[i + 1] + bin_edges[i]) / 2)
        maximum = max(hist_values)
        index = list(hist_values).index(maximum)
        max_x = mids[index]

        def gaussian(x, sigma, a):
            return 1 / (sigma * a) * np.exp(-np.power((x - max_x) / sigma, 2) / 2)

        params, covariance = curve_fit(gaussian, mids, hist_values, maxfev=10000, ftol=1e-6, xtol=1e-6)
        sigma, a = params[0], params[1]
        y_fit = [gaussian(mid, sigma, a) for mid in mids]

        xn,mu = gen_file(max_x)
        x = gen_noise(xn,sigma)
        t = [i for i in range(len(x))]

        sigma *= sigma
        x_fit = []
        mu0 = max_x
        # m_col = mu.columns
        # m = mu[m_col[0]].values
        m = mu0
        # for i in range(len(x)):
        #     m = mu0 + mu[i]
        #     xt = x[i] ** 2
        #     print("x**2:",xt)
        #     print("mu0-dmu ** 2:",m**2)
        #     print("diff:",abs(xt-m**2))
        #     if sigma > abs(xt-m**2):
        #         if mu0 < x[i]:
        #             mu0 -= np.sqrt(abs(sigma-xt))
        #         elif mu0 > x[i]:
        #             mu0 += np.sqrt(abs(sigma-xt))
        #     x_fit.append(mu0)
        # self._f_plot(x, xn, t)
        # self._f_plot(x, x_fit, t)
        # self._f_plot(xn, x_fit, t)

def gen_file(start_pos):
    t = [i for i in range(2500)]
    y = []
    d_mu = []
    y.append(start_pos)
    d_mu.append(0)

    for i in range(1, len(t)):
        y.append(3 * np.sin(t[i]/1000) + start_pos)
        d_mu.append((y[i] - y[i-1])/(t[i]-t[i-1]))

    return y, d_mu

def gen_noise(y, sigma):
    yn = []
    for i in range(len(y)):
        yn.append(y[i] + rnd.gauss(0, sigma))
    return yn

if __name__ == '__main__':
    CSV = File()
    data = CSV.read_CSV(r"C:\Users\EleMANtrO\PycharmProjects\filter\test.txt")
    gauss_filter = Filter(data)
    # gauss_filter.gauss()