import delay_weighted_cal as dwcal
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
        print(f"Grad calc., real, ant {test_ant}: {pass_text}")
    else:
        print(f"Grad calc., imag, ant {test_ant}: {pass_text}")
    if verbose:
        print(f"Empirical value: {empirical_val}")
        print(f"Calculated value: {calculated_val}")

    return pass_condition


def test_hess(
    test_ant,
    test_freq,
    readout_ant,
    readout_freq,
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
    real_part1=True,
    real_part2=True,
    verbose=False,
):

    if real_part1:
        part1_text = "real"
        multiplier = 1.0
    else:
        part1_text = "imag"
        multiplier = 1j
    if real_part2:
        part2_text = "real"
    else:
        part2_text = "imag"

    if verbose:
        print("*******")
        print(f"Testing the hessian calculation, {part1_text}-{part2_text} part")

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
    if not real_part1:
        test_ind += Nants * Nfreqs
    readout_ind = readout_ant * Nfreqs + readout_freq
    if not real_part2:
        readout_ind += Nants * Nfreqs

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

    empirical_value = (grad1[readout_ind] - grad0[readout_ind]) / delta_gains
    calc_value = hess[readout_ind, test_ind]
    pass_condition = np.isclose(
        empirical_value, calc_value, rtol=1e-06, atol=1e-06, equal_nan=False
    )
    if pass_condition:
        pass_text = "passed"
    else:
        pass_text = "FAILED!!!!!!!!!"
    print(
        f"Hess calc., {part1_text}-{part2_text}, ants [{test_ant},{readout_ant}]: {pass_text}"
    )
    if verbose:
        print(f"Empirical value: {empirical_value}")
        print(f"Calculated value: {calc_value}")

    return pass_condition


def test_derivative_calculations():

    data, model = dwcal.get_test_data(
        model_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
        model_use_model=True,
        data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        debug_limit_freqs=None,
        use_antenna_list=[3, 4, 57, 70, 92, 110],
        use_flagged_baselines=False,
    )

    Nants = data.Nants_data
    Nbls = data.Nbls
    Ntimes = data.Ntimes
    Nfreqs = data.Nfreqs

    # Format visibilities
    data_visibilities = np.zeros((Ntimes, Nbls, Nfreqs), dtype=complex)
    model_visibilities = np.zeros((Ntimes, Nbls, Nfreqs), dtype=complex)
    flag_array = np.zeros((Ntimes, Nbls, Nfreqs), dtype=bool)
    for time_ind, time_val in enumerate(np.unique(data.time_array)):
        data_copy = data.copy()
        model_copy = model.copy()
        data_copy.select(times=time_val)
        model_copy.select(times=time_val)
        data_copy.reorder_blts()
        model_copy.reorder_blts()
        data_copy.reorder_freqs(channel_order="freq")
        model_copy.reorder_freqs(channel_order="freq")
        if time_ind == 0:
            metadata_reference = data_copy.copy(metadata_only=True)
        model_visibilities[time_ind, :, :] = np.squeeze(
            model_copy.data_array, axis=(1, 3)
        )
        data_visibilities[time_ind, :, :] = np.squeeze(
            data_copy.data_array, axis=(1, 3)
        )
        flag_array[time_ind, :, :] = np.max(
            np.stack(
                [
                    np.squeeze(model_copy.flag_array, axis=(1, 3)),
                    np.squeeze(data_copy.flag_array, axis=(1, 3)),
                ]
            ),
            axis=0,
        )

    # Create gains expand matrices
    gains_exp_mat_1 = np.zeros((Nbls, Nants), dtype=int)
    gains_exp_mat_2 = np.zeros((Nbls, Nants), dtype=int)
    antenna_list = np.unique(
        [metadata_reference.ant_1_array, metadata_reference.ant_2_array]
    )
    for baseline in range(metadata_reference.Nbls):
        gains_exp_mat_1[
            baseline, np.where(antenna_list == metadata_reference.ant_1_array[baseline])
        ] = 1
        gains_exp_mat_2[
            baseline, np.where(antenna_list == metadata_reference.ant_2_array[baseline])
        ] = 1

    # Initialize gains
    # gain_init_noise = 0.1
    # gains_init = np.random.normal(
    #    1.0, gain_init_noise, size=(Nants, Nfreqs),
    # ) + 1.0j * np.random.normal(0.0, gain_init_noise, size=(Nants, Nfreqs),)
    gains_init = np.full((Nants, Nfreqs), 1.01 + 0.01j, dtype="complex")

    weight_mat = dwcal.get_weighted_weight_mat(
        Nfreqs, Nbls, metadata_reference.uvw_array, metadata_reference.freq_array
    )

    test_ant = 3
    test_freq = 1
    readout_ant = 2
    readout_freq = 1
    delta_gains = 0.0001

    test_grad(
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

    test_grad(
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

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
        readout_freq,
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
        real_part1=True,
        real_part2=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
        readout_freq,
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
        real_part1=True,
        real_part2=False,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
        readout_freq,
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
        real_part1=False,
        real_part2=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
        readout_freq,
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
        real_part1=False,
        real_part2=False,
    )


def test_derivative_calculations_randomized():

    Nants = 10
    Nbls = int((Nants ** 2 - Nants) / 2)
    Ntimes = 2
    Nfreqs = 384

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

    test_ant = np.random.randint(0, Nants - 1)
    test_freq = np.random.randint(0, Nfreqs - 1)
    #readout_ant = np.random.randint(0, Nants - 1)
    readout_ant = test_ant
    readout_freq = np.random.randint(0, Nfreqs - 1)
    delta_gains = 0.0001

    test_grad(
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
        verbose=True,
    )

    test_grad(
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
        verbose=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
        readout_freq,
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
        real_part1=True,
        real_part2=True,
        verbose=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
        readout_freq,
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
        real_part1=True,
        real_part2=False,
        verbose=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
        readout_freq,
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
        real_part1=False,
        real_part2=True,
        verbose=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
        readout_freq,
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
        real_part1=False,
        real_part2=False,
        verbose=True,
    )

    # Test hess frequency diagonals
    test_hess(
        test_ant,
        test_freq,
        readout_ant,
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
        real_part1=True,
        real_part2=True,
        verbose=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
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
        real_part1=True,
        real_part2=False,
        verbose=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
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
        real_part1=False,
        real_part2=True,
        verbose=True,
    )

    test_hess(
        test_ant,
        test_freq,
        readout_ant,
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
        real_part1=False,
        real_part2=False,
        verbose=True,
    )


def test_derivative_calculations_all_baselines():

    Nants = 10
    Nbls = int((Nants ** 2 - Nants) / 2)
    Ntimes = 1
    Nfreqs = 1

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

        pass_cond.extend([grad_pass_1, grad_pass_2])

        for readout_ant in range(Nants):

            # if readout_ant == test_ant:  # Autocorrelations are excluded
            #    continue

            hess_pass_1 = test_hess(
                test_ant,
                test_freq,
                readout_ant,
                readout_freq,
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
                real_part1=True,
                real_part2=True,
            )

            hess_pass_2 = test_hess(
                test_ant,
                test_freq,
                readout_ant,
                readout_freq,
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
                real_part1=True,
                real_part2=False,
            )

            hess_pass_3 = test_hess(
                test_ant,
                test_freq,
                readout_ant,
                readout_freq,
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
                real_part1=False,
                real_part2=True,
            )

            hess_pass_4 = test_hess(
                test_ant,
                test_freq,
                readout_ant,
                readout_freq,
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
                real_part1=False,
                real_part2=False,
            )

            pass_cond.extend([hess_pass_1, hess_pass_2, hess_pass_3, hess_pass_4])

            if test_freq != readout_freq:
                # Test hess frequency diagonals
                hess_pass_5 = test_hess(
                    test_ant,
                    test_freq,
                    readout_ant,
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
                    real_part1=True,
                    real_part2=True,
                )

                hess_pass_6 = test_hess(
                    test_ant,
                    test_freq,
                    readout_ant,
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
                    real_part1=True,
                    real_part2=False,
                )

                hess_pass_7 = test_hess(
                    test_ant,
                    test_freq,
                    readout_ant,
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
                    real_part1=False,
                    real_part2=True,
                )

                hess_pass_8 = test_hess(
                    test_ant,
                    test_freq,
                    readout_ant,
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
                    real_part1=False,
                    real_part2=False,
                )

                pass_cond.extend([hess_pass_5, hess_pass_6, hess_pass_7, hess_pass_8])

    if np.min(np.array(pass_cond)) == True:
        print("All tests passed successfully.")
    else:
        print("WARNING: Failed tests.")


def test_calibration():

    data, model = dwcal.get_test_data(
        model_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
        model_use_model=True,
        data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
        data_use_model=True,
        obsid="1061316296",
        pol="XX",
        use_autos=False,
        debug_limit_freqs=None,
        use_antenna_list=[3, 4, 57, 70, 92, 110],
        use_flagged_baselines=False,
    )

    cal = dwcal.calibration_optimization(
        data,
        model,
        use_wedge_exclusion=False,
        log_file_path=None,
        apply_flags=False,
    )


if __name__ == "__main__":
    test_derivative_calculations_all_baselines()
