import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import glob

# Model from Miyamoto et al. 1981, code written by Jonas Hallstrom 2022

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
modeled_depths = [1, 64900, 79900, 82700]
# There is an asymptote every pi/R, and one root on smaller/left side
# for big_R between 1000 and 1_000_000 and for n between 1 and 1000:
# inputting ((n-1)*np.pi+1)/big_R is positive, and (n*np.pi-0.000001)/big_R is negative

# t is in seconds because we have to convert it all to SI

""" Comparison to the H and L chondrite calculated using my finite difference (FD) solver"""
FD_H_file = glob.glob("MiyamotoLikeResults_H_08.10.2022*")[0]
FD_L_file = glob.glob("MiyamotoLikeResults_L_08.10.2022*")[0]

with open(FD_H_file) as f:
    [f.readline() for i in range(10)]
    data_header = f.readline()
    FD_H_shells = np.array([float(data.split("m")[0]) for data in data_header.split("T(")[1:]])
with open(FD_L_file) as f:
    [f.readline() for i in range(10)]
    data_header = f.readline()
    FD_L_shells = np.array([float(data.split("m")[0]) for data in data_header.split("T(")[1:]])

FD_H_data = np.genfromtxt(FD_H_file, skip_header=11)
FD_L_data = np.genfromtxt(FD_L_file, skip_header=11)

FD_H_times = (FD_H_data[:, 0] - FD_H_data[0,0]) / y_to_sec  # in years from formation now
FD_L_times = (FD_L_data[:, 0] - FD_L_data[0,0]) / y_to_sec

print("H Chondrite")
print("Dist. from center, peak temperature")
for depth in modeled_depths:

    time_array =  np.logspace(5.0, 8.0, 100) * y_to_sec
    H_core_temp_array = Miyamoto_solution(time_array, r=depth)
    print(depth, np.max(H_core_temp_array))

    plt.semilogx(time_array / y_to_sec, H_core_temp_array, label="1981: {:.1f}".format(depth/1000))
    FD_H_idx = np.argmin(np.abs(depth - FD_H_shells)) + 1
    plt.semilogx(FD_H_times, FD_H_data[:, FD_H_idx], label="FD: {:.1f}".format(FD_H_shells[FD_H_idx-1]/1000))
    plt.ylim([100, 1300])
    plt.xlim([1e5, 1e8])

plt.title("H Chondrite Body")
plt.legend(title="From Center (km)")
plt.show()

print("\nL Chondrite")
for depth in modeled_depths:

    time_array =  np.logspace(5.0, 8.0, 100) * y_to_sec
    L_core_temp_array = Miyamoto_solution(time_array, r=depth, T_0=180, A=11.67*(5e-6)*1.1)
    print(depth, np.max(L_core_temp_array))

    plt.semilogx(time_array / y_to_sec, L_core_temp_array, label="{:.1f}".format(depth/1000))
    FD_L_idx = np.argmin(np.abs(depth - FD_L_shells)) + 1
    plt.semilogx(FD_L_times, FD_L_data[:, FD_L_idx], label="FD: {:.1f}".format(FD_L_shells[FD_L_idx-1]/1000))
    plt.ylim([100, 1300])
    plt.xlim([1e5, 1e8])

plt.title("L Chondrite Body")
plt.legend(title="From Center (km)")
plt.show()
