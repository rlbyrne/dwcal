from dwcal import delay_weighted_cal as dwcal
import numpy as np


def test_grad(
    test_ant,
    test_freq,
    delta_gains,
    gains_init,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    weight_mat,
    data_visibilities,
    real_part=True,
    verbose=False,
):

    if real_part:
        real_imag_text = "real"
        multiplier = 1.0
    else:
        real_imag_text = "imag"
        multiplier = 1j

    if verbose:
        print("*******")
        print(f"Testing the gradient calculation, {real_imag_text} part")

    gains0 = np.copy(gains_init)
    gains0[test_ant, test_freq] -= multiplier * delta_gains / 2.0
    gains0_expanded = np.stack((np.real(gains0), np.imag(gains0)), axis=0).flatten()
    gains1 = np.copy(gains_init)
    gains1[test_ant, test_freq] += multiplier * delta_gains / 2.0
    gains1_expanded = np.stack((np.real(gains1), np.imag(gains1)), axis=0).flatten()
    gains_init_expanded = np.stack(
        (np.real(gains_init), np.imag(gains_init)), axis=0
    ).flatten()

    test_ind = test_ant * Nfreqs + test_freq
    if not real_part:
        test_ind += Nants * Nfreqs

    negloglikelihood0 = dwcal.cost_function_dw_cal(
        gains0_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        weight_mat,
        data_visibilities,
        verbose=False,
    )
    negloglikelihood1 = dwcal.cost_function_dw_cal(
        gains1_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        weight_mat,
        data_visibilities,
        verbose=False,
    )
    grad = dwcal.jac_dw_cal(
        gains_init_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        weight_mat,
        data_visibilities,
    )

    empirical_val = (negloglikelihood1 - negloglikelihood0) / delta_gains
    calculated_val = grad[test_ind]
    pass_condition = np.isclose(
        empirical_val, calculated_val, rtol=1e-06, atol=1e-06, equal_nan=False
    )
    if pass_condition:
        pass_text = "passed"
    else:
        pass_text = "FAILED!!!!!!!!!"
    if real_part:
        print(f"Grad calc., real, ant {test_ant}, freq {test_freq}: {pass_text}")
    else:
        print(f"Grad calc., imag, ant {test_ant}, freq {test_freq}: {pass_text}")
    if verbose:
        print(f"Empirical value: {empirical_val}")
        print(f"Calculated value: {calculated_val}")

    return pass_condition


def test_hess(
    test_ant,
    test_freq,
    delta_gains,
    gains_init,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    weight_mat,
    data_visibilities,
    real_part=True,
    verbose=False,
):

    if real_part:
        part1_text = "real"
        multiplier = 1.0
    else:
        part1_text = "imag"
        multiplier = 1j

    if verbose:
        print("*******")
        print(f"Testing the hessian calculation, {part1_text} part")

    gains0 = np.copy(gains_init)
    gains0[test_ant, test_freq] -= multiplier * delta_gains / 2.0
    gains0_expanded = np.stack((np.real(gains0), np.imag(gains0)), axis=0).flatten()
    gains1 = np.copy(gains_init)
    gains1[test_ant, test_freq] += multiplier * delta_gains / 2.0
    gains1_expanded = np.stack((np.real(gains1), np.imag(gains1)), axis=0).flatten()
    gains_init_expanded = np.stack(
        (np.real(gains_init), np.imag(gains_init)), axis=0
    ).flatten()

    test_ind = test_ant * Nfreqs + test_freq
    if not real_part:
        test_ind += Nants * Nfreqs

    grad0 = dwcal.jac_dw_cal(
        gains0_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        weight_mat,
        data_visibilities,
    )

    grad1 = dwcal.jac_dw_cal(
        gains1_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        weight_mat,
        data_visibilities,
    )

    hess = dwcal.hess_dw_cal(
        gains_init_expanded,
        Nants,
        Nfreqs,
        Nbls,
        model_visibilities,
        gains_exp_mat_1,
        gains_exp_mat_2,
        weight_mat,
        data_visibilities,
    )

    empirical_values = (grad1 - grad0) / delta_gains
    calc_values = hess[:, test_ind]
    pass_condition = np.isclose(
        empirical_values, calc_values, rtol=1e-06, atol=1e-06, equal_nan=False
    )
    if np.min(pass_condition):
        pass_text = "passed"
    else:
        pass_text = "FAILED!!!!!!!!!"
    print(
        f"Hess calc., {part1_text}, ant {test_ant}, freq {test_freq}: {pass_text}"
    )

    return np.min(pass_condition)


def test_derivative_calculations_all_baselines():

    Nants = 10
    Nbls = int((Nants ** 2 - Nants) / 2)
    Ntimes = 1
    Nfreqs = 5

    ant_1_array = np.zeros(Nbls, dtype=int)
    ant_2_array = np.zeros(Nbls, dtype=int)
    ind = 0
    for ant_1 in range(Nants):
        for ant_2 in range(ant_1 + 1, Nants):
            ant_1_array[ind] = ant_1
            ant_2_array[ind] = ant_2
            ind += 1

    # Format visibilities
    data_stddev = 6.0
    data_visibilities = np.random.normal(
        0.0,
        data_stddev,
        size=(Ntimes, Nbls, Nfreqs),
    ) + 1.0j * np.random.normal(
        0.0,
        data_stddev,
        size=(Ntimes, Nbls, Nfreqs),
    )
    model_visibilities = np.random.normal(
        0.0,
        data_stddev,
        size=(Ntimes, Nbls, Nfreqs),
    ) + 1.0j * np.random.normal(
        0.0,
        data_stddev,
        size=(Ntimes, Nbls, Nfreqs),
    )

    # Create gains expand matrices
    gains_exp_mat_1 = np.zeros((Nbls, Nants), dtype=int)
    gains_exp_mat_2 = np.zeros((Nbls, Nants), dtype=int)
    antenna_list = np.unique([ant_1_array, ant_2_array])
    for baseline in range(Nbls):
        gains_exp_mat_1[baseline, np.where(antenna_list == ant_1_array[baseline])] = 1
        gains_exp_mat_2[baseline, np.where(antenna_list == ant_2_array[baseline])] = 1

    # Initialize gains
    gain_init_noise = 0.1
    gains_init = np.random.normal(
        1.0,
        gain_init_noise,
        size=(Nants, Nfreqs),
    ) + 1.0j * np.random.normal(
        0.0,
        gain_init_noise,
        size=(Nants, Nfreqs),
    )

    weight_mat_stddev = 5.0
    weight_mat = np.random.normal(0.0, weight_mat_stddev, size=(Nbls, Nfreqs, Nfreqs))
    weight_mat += np.transpose(weight_mat, (0, 2, 1))  # Matrix must be Hermitian

    if Nfreqs == 1:
        test_freq = 0
        readout_freq = 0
    else:
        test_freq = np.random.randint(0, Nfreqs - 1)
        readout_freq = np.random.randint(0, Nfreqs - 1)
    delta_gains = 0.0001

    pass_cond = []
    for test_ant in range(Nants):
        for test_freq in range(Nfreqs):

            grad_pass_1 = test_grad(
                test_ant,
                test_freq,
                delta_gains,
                gains_init,
                Nants,
                Nfreqs,
                Nbls,
                model_visibilities,
                gains_exp_mat_1,
                gains_exp_mat_2,
                weight_mat,
                data_visibilities,
                real_part=True,
            )

            grad_pass_2 = test_grad(
                test_ant,
                test_freq,
                delta_gains,
                gains_init,
                Nants,
                Nfreqs,
                Nbls,
                model_visibilities,
                gains_exp_mat_1,
                gains_exp_mat_2,
                weight_mat,
                data_visibilities,
                real_part=False,
            )

            hess_pass_1 = test_hess(
                test_ant,
                test_freq,
                delta_gains,
                gains_init,
                Nants,
                Nfreqs,
                Nbls,
                model_visibilities,
                gains_exp_mat_1,
                gains_exp_mat_2,
                weight_mat,
                data_visibilities,
                real_part=True,
            )

            hess_pass_2 = test_hess(
                test_ant,
                test_freq,
                delta_gains,
                gains_init,
                Nants,
                Nfreqs,
                Nbls,
                model_visibilities,
                gains_exp_mat_1,
                gains_exp_mat_2,
                weight_mat,
                data_visibilities,
                real_part=True,
            )

            pass_cond.extend([grad_pass_1, grad_pass_2, hess_pass_1, hess_pass_2])

    if np.min(pass_cond):
        print("All tests passed successfully.")
    else:
        print("WARNING: Failed tests.")
