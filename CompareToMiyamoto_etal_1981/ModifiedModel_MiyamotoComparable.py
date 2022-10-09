# Simulations for the thermal evolution of the parent body of Itokawa

# Jonas Hallstrom, Arizona State University, 2022

# Citations:
# Henke S. et al. 2012. A&A 537:A45
# Henke S. public FORTRAN thermal model at https://www.ita.uni-heidelberg.de/~gail/AsteroidEvol.html
# Wakita S. et al. 2014. Meteoritics & Planetary Science 49(2):228–236
# Miyamoto M. et al. 1981. Lunar Planet. Sci., 12B p. 1145-1152.

# Imported packages
import numpy as np  # General math
from scipy.linalg import solve_banded  # Solving banded matrix equation
import matplotlib.pyplot as plt  # Plotting
import time as time_module  # Timing of code
from datetime import datetime  # Datetime information for file naming

# Defined functions and classes
def solve_heat_conduction_FPI(body_, tstep_, old_heat_, old_T_, old_c_p_, old_k_):
    ''' Evolves the time and temperature of the body_
    :param body_: An instance of the Planetesimal class, which we will take certain constants from and apply the
                  heat and time evolution to
    :param tstep_: The amount of time in the future we are solving the temperature for [sec]
    :param old_heat_: The heat power from the material in the current time [W / kg]
    :param old_T_: The temperature in the current time [K]
    :param old_c_p_: The specific heat in the current time [J / ( kg * K )]
    :param old_k_: The heat conductivity in the current time [W / ( m * K )]
    :return: Returns nothing (None) if succesful, as the changes have been applied to the Planetesimal class instance
             already. If there are too many fixed point iteration loops, returns True to signal convergence failure.
    '''
    new_heat_ = body_.decay_heat(body_.time + tstep_)  # [W / kg]
    old_Q = (old_heat_ / old_c_p_) * (tstep_ / 2)  # [K]
    num_radial_points_ = body_.num_radial_nodes
    new_T_previous_try = np.zeros(num_radial_points_)

    new_rho_ = body_.rho  # in this model, the density does not change over time
    old_rho_ = body_.rho
    new_r_ = body_.radial_points  # in this model, the radial points do not change over time
    old_r_ = body_.radial_points

    # Initializations before the Fixed Point Iteration loop that tries to solve the heat conduction

    old_A = np.zeros(num_radial_points_)
    old_B = np.zeros(num_radial_points_)
    old_C = np.zeros(num_radial_points_)
    new_A = np.zeros(num_radial_points_)
    new_B = np.zeros(num_radial_points_)
    new_C = np.zeros(num_radial_points_)

    banded_LHS = np.zeros((3, num_radial_points_))
    RHS = np.zeros(num_radial_points_)

    FPI_counter = 0
    while np.max(np.abs(new_T_previous_try - body_.T) / body_.T) > 5e-8:
        FPI_counter += 1
        if FPI_counter > 10:
            print("Conduction FPI Exceeded 10 Loops!")
            return True

        new_T_previous_try = body_.T

        new_c_p_ = body_.c_p
        new_k_ = body_.k
        new_Q = (new_heat_ / new_c_p_) * (tstep_ / 2)

        old_lambda = (old_k_ / (old_rho_ * old_c_p_)) * (tstep_ / 2)  # [m^2]
        new_lambda = (new_k_ / (new_rho_ * new_c_p_)) * (tstep_ / 2)

        for i in range(1, num_radial_points_ - 1):
            # All unitless
            old_A[i] = old_lambda[i] * (2 / (old_r_[i] * (old_r_[i + 1] - old_r_[i - 1]))
                                        + 2 / ((old_r_[i + 1] - old_r_[i]) * (old_r_[i + 1] - old_r_[i - 1])))
            new_A[i] = new_lambda[i] * (2 / (new_r_[i] * (new_r_[i + 1] - new_r_[i - 1]))
                                        + 2 / ((new_r_[i + 1] - new_r_[i]) * (new_r_[i + 1] - new_r_[i - 1])))
            old_B[i] = old_lambda[i] * (-2 / ((old_r_[i + 1] - old_r_[i]) * (old_r_[i] - old_r_[i - 1])))
            new_B[i] = new_lambda[i] * (-2 / ((new_r_[i + 1] - new_r_[i]) * (new_r_[i] - new_r_[i - 1])))
            old_C[i] = old_lambda[i] * (-2 / (old_r_[i] * (old_r_[i + 1] - old_r_[i - 1]))
                                        + 2 / ((old_r_[i] - old_r_[i - 1]) * (old_r_[i + 1] - old_r_[i - 1])))
            new_C[i] = new_lambda[i] * (-2 / (new_r_[i] * (new_r_[i + 1] - new_r_[i - 1]))
                                        + 2 / ((new_r_[i] - new_r_[i - 1]) * (new_r_[i + 1] - new_r_[i - 1])))

        old_N = (2 * old_lambda[0]) / (old_r_[1] ** 2)
        new_N = (2 * new_lambda[0]) / (new_r_[1] ** 2)

        # Now that we have all the variables for the Crank-Nicolson scheme, we'll put them in the right hand side
        #  vector and in the left hand side matrix (writing the sparse/tridiagonal matrix as a banded matrix for space)
        banded_LHS[0, 1] = -new_N
        banded_LHS[1, 0] = 1 + new_N
        for i in range(num_radial_points_ - 2):
            banded_LHS[0, i + 2] = -new_A[i + 1]
            banded_LHS[1, i + 1] = 1 - new_B[i + 1]
            banded_LHS[2, i] = -new_C[i + 1]
        banded_LHS[1, -1] = 1

        RHS[0] = old_N * old_T_[1] + (1 - old_N) * old_T_[0] + old_Q[0] + new_Q[0]
        for i in range(1, num_radial_points_ - 1):
            RHS[i] = old_A[i] * old_T_[i + 1] + (1 + old_B[i]) * old_T_[i] + old_C[i] * old_T_[i - 1] + old_Q[i] + new_Q[i]
        RHS[-1] = body_.T_surf

        # Now that we have the LHS matrix and RHS vector for the CN scheme, scipy solves for the temperatures of the
        #  next timestep. These temperatures are then saved to the planetesimal object.
        body_.T = solve_banded((1, 1), banded_LHS, RHS)

        body_.update_specific_heat()
        body_.update_heat_conductivity()


class Planetesimal:
    G = 6.67408e-11  # m^3 / (kg s^2)
    pi = np.pi

    def __init__(self, T_0, T_surf, rho, t_form, total_radius, num_radial_nodes, first_thickness, h_magnitude, h_tau, cp_multiplier, kappa_multiplier):
        ''' Initial or constant values for the body
        :param T_0: Initial temperature of entire body [K]
        :param T_surf: Surface temperature (held constant throughout evolution) [K]
        :param rho: Mass density of the material [kg / m^3]
        :param t_form: Time of formation from CAI [sec]
        :param total_radius: Total radius of the body [m]
        :param num_radial_nodes: Number of radial finite-difference nodes
        :param first_thickness: Thickness of the thinnest and outermost shell (surface) [m]
        :param h_magnitude: Magnitude of heat production from exponential nuclear decay [W / kg]
        :param h_tau: Decay rate for exponential nuclear decay (sec)
        '''

        # Define instance variables for the parameters that are constants
        self.num_radial_nodes = num_radial_nodes
        self.rho = rho
        self.T_surf = T_surf
        self.h_magnitude = h_magnitude
        self.h_tau = h_tau
        self.cp_multiplier = cp_multiplier / 100  # Convert from percent to decimal
        self.kappa_multiplier = kappa_multiplier / 100

        # Define initial values for instance variables from the parameters
        self.T = T_0 * np.ones(num_radial_nodes)
        self.time = t_form
        self.update_specific_heat()  # Specific heat [J / ( kg * K )]
        self.update_heat_conductivity()  # Heat conductivity [W / ( m * K )]

        # Create spacing of the radial points from a geometric series, according to thickness of the outer layer
        # and the number of shells. This creates more nodes near the surface than near the center.
        if geo_spacing:
            coefficients = np.zeros(num_radial_nodes)
            coefficients[-1] = total_radius / first_thickness - 1
            coefficients[-2] = - total_radius / first_thickness
            coefficients[0] = 1
            geometric_solved = np.roots(coefficients)
            thickening = geometric_solved.real[
                np.logical_and(abs(geometric_solved.imag) < 1e-15, geometric_solved.real > 1 + 1e-10)]
            depths = np.zeros(num_radial_nodes)
            depths[1] = first_thickness
            for i in range(2, num_radial_nodes):
                depths[i] = depths[i - 1] + thickening ** (i - 1) * first_thickness
            self.radial_points = np.abs(np.flip(np.around(total_radius - depths)))
        else:
            self.radial_points = np.linspace(0, total_radius, num_radial_nodes)

    def update_heat_conductivity(self):
        self.k =  self.c_p * self.rho * self.kappa_multiplier *( 1.29e-7 + (1.52e-4)/self.T )

    def update_specific_heat(self):
        self.c_p = self.cp_multiplier * (800 + 0.25*self.T - (1.5e7)/(self.T**2))

    def decay_heat(self, t):
        return np.sum(self.h_magnitude * np.exp(-t / self.h_tau))

# Begin main code
if __name__ == '__main__':

    """ ----- CONTROL PANEL START ----- """
    # Planetesimal settings
    T_0 = 200  # K
    T_s = 200  # K
    kappa_multiplier = 100  # percent (100 would be default)
    cp_multiplier = 100  # percent (100 would be default)
    # Note that for this specific model, the value of density does not matter
    ## as it cancels out of the equations of heat conduction
    density = 3400  # [kg/m^3]
    t_form_kyr = 2200  # kyr or ka
    t_form_seconds = t_form_kyr * 1e3 * 365.25 * 24 * 60 * 60  # [sec]
    tot_rad = 50_000  # meters
    num_shells = 301  # Recommend: 100 for smaller bodies and up to 300 for larger bodies
    geo_spacing = False  # Option for spacing shell positions out by a geometric sequence
    first_thick = 20  # m, for geo spacing only
    heat_multiplier = 100  # percent (100 would be default)
    h_mag = np.array([2.034e-7]) * heat_multiplier / 100  # [W / kg], 26 Al
    h_tau = np.array([1.0387e6]) * 365.25 * 24 * 60 * 60  # [sec], 26 Al

    # Simulation settings
    t_end_myr = 10  # This is the end of the simulation only IF the central temperature is less than 1.05 times the surface.
    end_of_simulation_time = t_end_myr * 1e6 * 365.25 * 24 * 60 * 60  # [sec]
    timestep = 1e1 * 365.25 * 24 * 60 * 60  # Initial timestep [sec]
    max_timestep = 1e3 * 365.25 * 24 * 60 * 60  # Maximum timestep [sec]

    # Data recording settings
    num_recorded_timesteps = 100  # Roughly the number of recorded timesteps, could be less based on max_timestep

    """ ----- CONTROL PANEL END ----- """

    # Additional data recording information
    if tot_rad % 1000 == 0 and tot_rad >= 3000:
        shells_to_record = [tot_rad, tot_rad - 500, tot_rad-1000, tot_rad-1500, tot_rad-2250, tot_rad-3000]
        k=0
        while (k<2 or shells_to_record[-1] % 5000 != 0) and shells_to_record[-1]!=0:
            # So it will do this at least two times, and then until it reaches a multiple of 5000, unless it hits 0
            shells_to_record.append(shells_to_record[-1] - 1000)
            k += 1
        while shells_to_record[-1] > 0:
            shells_to_record.append(shells_to_record[-1] - 5000)
    elif tot_rad >= 3000:
        less_rad = tot_rad - tot_rad % 1000
        print(less_rad)
        shells_to_record = [less_rad, less_rad - 500, less_rad-1000, less_rad-1500, less_rad-2250, less_rad-3000]
        k=0
        while (k<2 or shells_to_record[-1] % 5000 != 0) and shells_to_record[-1]!=0:
            # So it will do this at least two times, and then until it reaches a multiple of 5000, unless it hits 0
            shells_to_record.append(shells_to_record[-1] - 1000)
            k += 1
        while shells_to_record[-1] > 0:
            shells_to_record.append(shells_to_record[-1] - 5000)
    else:
        raise ValueError("Radius must be greater than 3000 for the shell recording to work")

    current_datetime = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    file_name = 'ItokawaResults_'+current_datetime+'-km{}-cpm{}-Ts{}-form{}-rad{}.txt'.format(kappa_multiplier, cp_multiplier, T_s,
                    t_form_seconds/(1e6 * 365.25 * 24 * 60 * 60), tot_rad/1000)  # Name of .txt file that will be saved
    figure_name = 'ItokawaResults_'+current_datetime+'.png'  # name of .png file that will be saved


    # Create an instance/object from the Planetesimal class
    parent = Planetesimal(T_0=T_0, T_surf=T_s, rho=density, t_form=t_form_seconds, total_radius=tot_rad,
                          num_radial_nodes=num_shells, first_thickness=first_thick, h_magnitude=h_mag, h_tau=h_tau, cp_multiplier=cp_multiplier, kappa_multiplier=kappa_multiplier)

    record_time = (end_of_simulation_time - t_form_seconds) / num_recorded_timesteps  # roughly how often it records
    shell_indices = [np.argmin(np.abs(parent.radial_points - shell)) for shell in shells_to_record]

    # Initializing some variables
    recorded_T = []
    recorded_time = []
    last_record = 0
    last_readout = 0
    max_temp = np.zeros(num_shells)

    # Main loop that evolves the temperature of the planetesimal class instance over time
    while parent.time < end_of_simulation_time or parent.T[0] > T_s*2:
        # Save the current properties of the class instance as 'old' so that we can revert the instance back to these
        #  in case the conduction for a time step is too big or doesn't converge, and we need to redo the timestep.
        old_heat = parent.decay_heat(parent.time)
        old_c_p = parent.c_p
        old_k = parent.k
        old_T = parent.T

        # Solves the conduction to find the new temperatures of the instance a timestep later. The conduction function
        #  will return True if it fails to converge, hence the output being labeled 'redo'.
        redo = solve_heat_conduction_FPI(body_=parent, tstep_=timestep, old_heat_=old_heat, old_T_=old_T,
                                         old_c_p_=old_c_p, old_k_=old_k)

        num_lowered = 0
        while np.max(np.abs(old_T - parent.T) / old_T) > 0.003 or redo == True:
            # This is saying that we tried to do too big of a timestep, and the resulting change in values was too large
            #   to be done in one timestep. So reset the body back to the 'old' state and try a lower timestep.

            parent.k = old_k
            parent.c_p = old_c_p
            parent.T = old_T

            num_lowered += 1
            print("Timestep Lowerer Activated! Number of activations this timestep:", num_lowered)
            timestep /= 1.5
            redo = solve_heat_conduction_FPI(body_=parent, tstep_=timestep, old_heat_=old_heat, old_T_=old_T,
                                              old_c_p_=old_c_p, old_k_=old_k)

        # Now that we've successfully evolved the temperature, we can update the time to match it.
        parent.time += timestep

        # If the temperature is too large then this model is no longer accurate and we should quit out:
        if np.max(parent.T) > 2000:
            file_name = "FAILED"+file_name
            figure_name = "FAILED"+figure_name
            break

        # If the change in temperature was very small, we will raise the size of the timestep
        if np.max(np.abs(old_T - parent.T) / old_T) < 0.0001 and timestep < max_timestep:
            print("- Timestep Raiser activated!")
            if timestep * 1.4 < max_timestep:
                timestep *= 1.4
            else:
                timestep = max_timestep

        # Most of the data produced and used in the simulation is overwritten and not saved, only sometimes recorded
        # here according to the num_recorded_timesteps
        if parent.time - last_record > record_time:
            last_record = parent.time
            recorded_T.append(parent.T[shell_indices])
            recorded_time.append(parent.time)

            if time_module.process_time()-last_readout > 1:
                last_readout = time_module.process_time()
                print("\nProcess time is at {:.2f} seconds".format(time_module.process_time()))
                print("Time Elapsed: {:.2f} Ma".format((parent.time - t_form_seconds) / (1e6*365.25*24*60*60)))
                print("Time After CAI: {:.2f} Ma".format(parent.time / (1e6*365.25*24*60*60)))
                print("Current Timestep: {:.2f} a".format(timestep / (365.25*24*60*60)))
                print("Center T: {:.1f} K or {:.1f} C".format(parent.T[0], parent.T[0] - 273.15))



        # Updates the maximum temperatures in each shell
        max_temp = np.maximum(max_temp, parent.T)


    # ---------- PLOTTING AND STORING RESULTS ----------
    print("\nFinished simulation")
    print("Array of Maximum Temperatures Throughout Body (center to surface): \n", np.round(max_temp, 1))

    temp_zone_bounds = (1073, 1273)
    temp_zone = parent.radial_points[np.logical_and(max_temp > temp_zone_bounds[0], max_temp < temp_zone_bounds[1])]
    if len(temp_zone) != 0:
        temp_zone_volume = (np.pi * 4 / 3) * (temp_zone[-1] ** 3 - temp_zone[0] ** 3)
        temp_zone_percent = temp_zone_volume / ((np.pi * 4 / 3) * tot_rad ** 3)
    else:
        temp_zone_volume = 0
        temp_zone_percent = 0

    # Plotting results
    for i in range(len(shells_to_record)):
        plt.semilogx([t / (365.25 * 24 * 60 * 60) for t in recorded_time], [snapshot[i] for snapshot in recorded_T])
    plt.ylabel("Temperature (K)")
    plt.xlabel("Time (annum)")
    plt.legend(["{:.2f} km".format(r / 1000) for r in parent.radial_points[shell_indices]])
    plt.savefig(figure_name)

    # Writing results to a file
    with open(file_name, 'w') as f:
        f.write("T_0 = {}".format(T_0) + ", T_s = {}".format(T_s) + ", Density = {}".format(density)
                + ", t_form = {}".format(t_form_seconds) + ", total_radi = {}".format(tot_rad)
                + ", num_radi_nodes = {}".format(num_shells) + ", geo_spacing = {}".format(geo_spacing)
                + ", first_thick = {}".format(first_thick) + ", h_mag = {}".format(h_mag)
                + ", h_tau = {}".format(h_tau) + ", h_multiplier = {}".format(heat_multiplier) +", max_timestep = {}".format(max_timestep) + ", kappa_multiiplier = {}".format(kappa_multiplier) + ", cp_mulitiplier = {}".format(cp_multiplier))
        f.write('\n')
        f.write('\nMax Temps')
        f.write('\n')
        f.write(' '.join(['{:.3e}'.format(x) for x in max_temp]))
        f.write('\n')
        f.write('\nRadial Points of Max Temps')
        f.write('\n')
        f.write('  '.join(['{:.3e}'.format(x) for x in parent.radial_points]))
        f.write('\n')
        f.write('\nVolume and Percent Volume in Temp Range {} to {}: vol:{}. %vol:{} '.format(
            temp_zone_bounds[0], temp_zone_bounds[1], temp_zone_volume, temp_zone_percent))
        f.write('\n')
        f.write('\n')
        f.write('time(sec)  ')
        for r in parent.radial_points[shell_indices]:
            f.write('T({:.2e} m) '.format(r))
        f.write('\n')
        for i in range(len(recorded_time)):
            f.write('{:.3e}  '.format(recorded_time[i]))
            f.write('     '.join(['{:.3e}'.format(x) for x in recorded_T[i]]))
            f.write('\n')
