import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

y_to_sec = 365.25*24*60*60


def alpha_function(x, R, h):
    return R*x * (1/np.tan(R*x)) + R*h - 1


def Miyamoto_solution(t, r, T_0=200, R=85_000, h=1.0, K=1.0, A=11.67*(5e-6), lamda=9.63e-7/y_to_sec, kappa=5e-7):
    summed_term = 0
    for n in range(1, 10_000):
        alpha = opt.root_scalar(alpha_function, (R, 1), bracket=[((n-1)*np.pi+1)/big_R, (n*np.pi-0.000001)/big_R]).root

        numer0 = 1
        denom0 = 1 - lamda / (kappa * (alpha**2))
        numer1 = (np.exp(-lamda*t) - np.exp(-kappa * (alpha**2) * t)) * np.sin(r*alpha)
        denom1 = (alpha**2) * ( (R**2)*(alpha**2) + R*h*(R*h-1) ) * np.sin(R*alpha)

        #print((numer0/denom0) * (numer1/denom1))
        summed_term += (numer0/denom0) * (numer1/denom1)
        #print(summed_term)

    prefactor = 2*h*(R**2)*A / (r*K)
    return T_0 + prefactor*summed_term

big_R = 85_000
# There is an asymptote every pi/R, and one root on smaller/left side
# for big_R between 1000 and 1_000_000 and for n between 1 and 1000:
# inputting ((n-1)*np.pi+1)/big_R is positive, and (n*np.pi-0.000001)/big_R is negative

# t is in seconds because we have to convert it all to SI

"""
for n in [1, 1000]:
    print(((n-1)*np.pi+1)/big_R, (n*np.pi-0.000001)/big_R)
    print(alpha_function(((n-1)*np.pi+1)/big_R, big_R, 1))
    print(alpha_function((n*np.pi-0.000001)/big_R, big_R, 1))
"""

for depth in [1000, 67000, 78000, 82500]:

    time_array =  np.logspace(5.0, 8.0, 100) * y_to_sec
    core_temp_array = Miyamoto_solution(time_array, r=depth)

    plt.semilogx(time_array / y_to_sec, core_temp_array, label="{:.1f}".format(depth/1000))
    plt.ylim([200, 1200])
    plt.xlim([1e5, 1e8])

plt.show()
