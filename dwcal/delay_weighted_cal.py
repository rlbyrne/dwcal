#!/usr/bin/python

import numpy as np
import matplotlib.pyplot as plt
import sys
import scipy
import scipy.optimize
import time
import pyuvdata


def get_test_data(
    model_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
    model_use_model=True,
    data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
    data_use_model=True,
    obsid="1061316296",
    pol="XX",
    use_autos=False,
    debug_limit_freqs=None,
    use_antenna_list=None,
    use_flagged_baselines=False,
):

    model = pyuvdata.UVData()
    if model_path.endswith(".uvfits"):
        if pol == "XX":
            pol_int = -5
        elif pol == "YY":
            pol_int = -6
        else:
            print("ERROR: Unknown polarization.")
            sys.exit(1)
        print(f"Loading model from {model_path}.")
        sys.stdout.flush()
        model.read_uvfits(model_path, polarizations=pol_int)
    else:
        model_filelist = [
            "{}/{}".format(model_path, file)
            for file in [
                "vis_data/{}_vis_{}.sav".format(obsid, pol),
                "vis_data/{}_vis_model_{}.sav".format(obsid, pol),
                "vis_data/{}_flags.sav".format(obsid),
                "metadata/{}_params.sav".format(obsid),
                "metadata/{}_settings.txt".format(obsid),
                "metadata/{}_layout.sav".format(obsid),
            ]
        ]
        if model_use_model:
            print(
                f"Loading model from {model_path}, using the FHD run's model visibilities."
            )
            sys.stdout.flush()
        else:
            print(
                f"Loading model from {model_path}, using the FHD run's data visibilities."
            )
            sys.stdout.flush()
        model.read_fhd(model_filelist, use_model=model_use_model)

    # Average across time
    model.downsample_in_time(n_times_to_avg=int(model.Ntimes))

    if debug_limit_freqs is not None:  # Limit frequency axis for debugging
        min_freq_channel = round(model.Nfreqs / 2 - debug_limit_freqs / 2)
        use_frequencies = model.freq_array[
            0, min_freq_channel : round(min_freq_channel + debug_limit_freqs)
        ]
        model.select(frequencies=use_frequencies)

    if use_antenna_list is not None:  # Use a subset of antennas
        model.select(antenna_nums=use_antenna_list)

    if not use_autos:  # Remove autocorrelations
        bl_lengths = np.sqrt(np.sum(model.uvw_array**2.0, axis=1))
        non_autos = np.where(bl_lengths > 0.01)[0]
        model.select(blt_inds=non_autos)

    if data_path != model_path or model_use_model != data_use_model:
        data = pyuvdata.UVData()

        if data_path.endswith(".uvfits"):
            if pol == "XX":
                pol_int = -5
            elif pol == "YY":
                pol_int = -6
            else:
                print("ERROR: Unknown polarization.")
                sys.exit(1)
            print(f"Loading data from {data_path}.")
            sys.stdout.flush()
            data.read_uvfits(data_path, polarizations=pol_int)
        else:
            data_filelist = [
                "{}/{}".format(data_path, file)
                for file in [
                    "vis_data/{}_vis_{}.sav".format(obsid, pol),
                    "vis_data/{}_vis_model_{}.sav".format(obsid, pol),
                    "vis_data/{}_flags.sav".format(obsid),
                    "metadata/{}_params.sav".format(obsid),
                    "metadata/{}_settings.txt".format(obsid),
                    "metadata/{}_layout.sav".format(obsid),
                ]
            ]
            if data_use_model:
                print(
                    f"Loading data from {data_path}, using the FHD run's model visibilities."
                )
                sys.stdout.flush()
            else:
                print(
                    f"Loading data from {data_path}, using the FHD run's data visibilities."
                )
                sys.stdout.flush()
            data.read_fhd(data_filelist, use_model=data_use_model)

        # Average across time
        data.downsample_in_time(n_times_to_avg=int(data.Ntimes))
        if debug_limit_freqs is not None:
            data.select(frequencies=use_frequencies)
        if use_antenna_list is not None:
            data.select(antenna_nums=use_antenna_list)
        if not use_autos:  # Remove autocorrelations
            bl_lengths = np.sqrt(np.sum(data.uvw_array**2.0, axis=1))
            non_autos = np.where(bl_lengths > 0.01)[0]
            data.select(blt_inds=non_autos)
    else:
        print("Using model for data")
        sys.stdout.flush()
        data = model.copy()

    # Ensure ordering matches between the data and model
    if np.max(np.abs(data.baseline_array - model.baseline_array)) > 0.0:
        data.reorder_blts()
        model.reorder_blts()
    if np.max(np.abs(data.freq_array - model.freq_array)) > 0.0:
        data.reorder_freqs(channel_order="freq")
        model.reorder_freqs(channel_order="freq")

    if not use_flagged_baselines:
        # Remove baselines with any flagged frequencies
        # This does not work if more that one time channel is used and flags are time-dependent
        flag_arr_combined = np.stack((model.flag_array, data.flag_array), axis=4)
        bl_flags = np.max(flag_arr_combined, axis=(1, 2, 3, 4))
        bl_inds_use = np.where(np.invert(bl_flags))[0]
        if len(bl_inds_use) < data.Nblts:
            print("use_flagged_baselines=False. Removing baselines with flags.")
            frac_data_removed = 1 - float(len(bl_inds_use)) / float(data.Nblts)
            print(f"Fraction of data removed: {frac_data_removed}")
            sys.stdout.flush()
            model.select(blt_inds=bl_inds_use)
            data.select(blt_inds=bl_inds_use)

    return data, model


def initialize_cal(data, antenna_list, gains=None):

    cal = pyuvdata.UVCal()
    cal.Nants_data = data.Nants_data
    cal.Nants_telescope = data.Nants_telescope
    cal.Nfreqs = data.Nfreqs
    cal.Njones = 1
    cal.Nspws = 1
    cal.Ntimes = 1
    cal.ant_array = antenna_list
    cal.antenna_names = data.antenna_names
    cal.antenna_numbers = data.antenna_numbers
    cal.cal_style = "sky"
    cal.cal_type = "gain"
    cal.channel_width = data.channel_width
    cal.freq_array = data.freq_array
    cal.gain_convention = "multiply"
    cal.history = ""
    cal.integration_time = np.mean(data.integration_time)
    cal.jones_array = np.array([-5])
    cal.spw_array = data.spw_array
    cal.telescope_name = data.telescope_name
    cal.time_array = np.array([np.mean(data.time_array)])
    cal.time_range = np.array([np.min(data.time_array), np.max(data.time_array)])
    cal.x_orientation = "east"
    if gains is None:  # Set all gains to 1
        cal.gain_array = np.full(
            (cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones),
            1,
            dtype=complex,
        )
    else:
        cal.gain_array = gains[:, np.newaxis, :, np.newaxis, np.newaxis]
    cal.flag_array = np.full(
        (cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones),
        False,
        dtype=bool,
    )  # Does not yet support flags
    cal.quality_array = np.full(
        (cal.Nants_data, cal.Nspws, cal.Nfreqs, cal.Ntimes, cal.Njones),
        1.0,
        dtype=float,
    )
    cal.ref_antenna_name = ""
    cal.sky_catalog = "GLEAM_bright_sources"
    cal.sky_field = "phase center (RA, Dec): ({}, {})".format(
        np.degrees(np.mean(data.phase_center_app_ra)),
        np.degrees(np.mean(data.phase_center_app_dec)),
    )

    if not cal.check():
        print("ERROR: UVCal check failed.")
        sys.exit(1)

    return cal


def cost_function_dw_cal(
    x,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    weight_mat,
    data_visibilities,
    verbose=True,
    lambda_val=None,
):

    if lambda_val is None:
        #lambda_val = float(Nbls)
        lambda_val = 1.

    gains = np.reshape(x, (2, Nants, Nfreqs))
    gains = gains[0, :, :] + 1.0j * gains[1, :, :]

    gains_expanded = np.matmul(gains_exp_mat_1, gains) * np.matmul(
        gains_exp_mat_2, np.conj(gains)
    )
    res_vec = model_visibilities - gains_expanded[np.newaxis, :, :] * data_visibilities
    weighted_part2 = np.squeeze(np.matmul(res_vec[:, :, np.newaxis, :], weight_mat))
    cost = np.real(np.sum(np.conj(np.squeeze(res_vec)) * weighted_part2))

    if lambda_val != 0.0:
        cost += lambda_val * np.sum(np.sum(np.angle(gains), axis=0) ** 2.0)

    if verbose:
        print(f"Cost func. value: {cost}")
        sys.stdout.flush()

    return cost


def jac_dw_cal(
    x,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    weight_mat,
    data_visibilities,
    lambda_val=None,
):

    if lambda_val is None:
        #lambda_val = float(Nbls)
        lambda_val = 1.

    gains = np.reshape(x, (2, Nants, Nfreqs))
    gains = gains[0, :, :] + 1.0j * gains[1, :, :]

    gains1_expanded = np.matmul(gains_exp_mat_1, gains)
    gains2_expanded = np.matmul(gains_exp_mat_2, gains)
    term1_part1 = gains1_expanded[np.newaxis, :, :] * data_visibilities
    term2_part1 = gains2_expanded[np.newaxis, :, :] * np.conj(data_visibilities)
    cost_term = (
        model_visibilities
        - gains1_expanded[np.newaxis, :, :]
        * np.conj(gains2_expanded[np.newaxis, :, :])
        * data_visibilities
    )
    weighted_part2 = np.squeeze(
        np.matmul(cost_term[:, :, np.newaxis, :], weight_mat), axis=2
    )
    term1 = np.sum(
        np.matmul(gains_exp_mat_2.T, term1_part1 * np.conj(weighted_part2)), axis=0
    )
    term2 = np.sum(np.matmul(gains_exp_mat_1.T, term2_part1 * weighted_part2), axis=0)
    grad = -2 * (term1 + term2)

    grad = np.stack((np.real(grad), np.imag(grad)), axis=0).flatten()

    if lambda_val != 0.0:
        regularization_term = (
            2
            * lambda_val
            * np.conj(gains)
            / np.abs(gains) ** 2.0
            * np.sum(np.angle(gains), axis=0)[np.newaxis, :]
        )
        regularization_term = np.stack(
            (np.imag(regularization_term), np.real(regularization_term)), axis=0
        ).flatten()
        grad += regularization_term

    return grad


def reformat_baselines_to_antenna_matrix(bl_array, gains_exp_mat_1, gains_exp_mat_2):
    # Reformat an array indexed in baselines into a matrix with antenna indices

    (Nbls, Nants) = np.shape(gains_exp_mat_1)
    antenna_matrix = np.zeros_like(
        bl_array[
            0,
        ],
        dtype=bl_array.dtype,
    )
    antenna_matrix = np.repeat(
        np.repeat(antenna_matrix[np.newaxis,], Nants, axis=0)[
            np.newaxis,
        ],
        Nants,
        axis=0,
    )
    antenna_numbers = np.arange(Nants)
    antenna1_num = np.matmul(gains_exp_mat_1, antenna_numbers)
    antenna2_num = np.matmul(gains_exp_mat_2, antenna_numbers)
    for bl_ind in range(Nbls):
        antenna_matrix[antenna1_num[bl_ind], antenna2_num[bl_ind],] = bl_array[
            bl_ind,
        ]
    return antenna_matrix


def hess_dw_cal(
    x,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    weight_mat,
    data_visibilities,
    lambda_val=None,
):

    if lambda_val is None:
        #lambda_val = float(Nbls)
        lambda_val = 1.

    gains = np.reshape(x, (2, Nants, Nfreqs))
    gains = gains[0, :, :] + 1.0j * gains[1, :, :]

    gains1_expanded = np.matmul(gains_exp_mat_1, gains)
    gains2_expanded = np.matmul(gains_exp_mat_2, gains)

    gains1_times_data = gains1_expanded[np.newaxis, :, :] * data_visibilities
    gains2_times_conj_data = gains2_expanded[np.newaxis, :, :] * np.conj(
        data_visibilities
    )
    term1 = np.sum(
        weight_mat[np.newaxis, :, :, :]
        * gains1_times_data[:, :, :, np.newaxis]
        * gains2_times_conj_data[:, :, np.newaxis, :],
        axis=0,
    )
    term1 = reformat_baselines_to_antenna_matrix(
        term1, gains_exp_mat_1, gains_exp_mat_2
    )
    term1 = np.transpose(term1, (1, 0, 2, 3))

    term2 = np.sum(
        np.conj(weight_mat[np.newaxis, :, :, :])
        * gains2_times_conj_data[:, :, :, np.newaxis]
        * gains1_times_data[:, :, np.newaxis, :],
        axis=0,
    )
    term2 = reformat_baselines_to_antenna_matrix(
        term2, gains_exp_mat_1, gains_exp_mat_2
    )

    # hess elements are ant_c, ant_d, freq_f0, freq_f1, and real/imag pair
    # The real/imag pairs are in order [real-real, real-imag, and imag-imag]
    hess = np.zeros((Nants, Nants, Nfreqs, Nfreqs, 3), dtype=float)
    hess[:, :, :, :, 0] = 2 * np.real(term1 + term2)
    hess[:, :, :, :, 1] = 2 * np.imag(term1 + term2)
    hess[:, :, :, :, 2] = -2 * np.real(term1 + term2)

    # Calculate frequency diagonals
    cost_term = (
        gains1_expanded[np.newaxis, :, :]
        * np.conj(gains2_expanded[np.newaxis, :, :])
        * data_visibilities
        - model_visibilities
    )
    weight_times_cost = np.einsum("ijk,jkl->ijl", cost_term, weight_mat)
    term3 = np.sum(np.conj(data_visibilities) * weight_times_cost, axis=0)
    term3 = reformat_baselines_to_antenna_matrix(
        term3, gains_exp_mat_1, gains_exp_mat_2
    )
    term4 = np.transpose(np.conj(term3), (1, 0, 2))
    terms3and4 = 2 * (term3 + term4)

    for freq in range(Nfreqs):
        hess[:, :, freq, freq, 0] += np.real(terms3and4[:, :, freq])
        hess[:, :, freq, freq, 1] -= np.imag(terms3and4[:, :, freq])
        hess[:, :, freq, freq, 2] += np.real(terms3and4[:, :, freq])

    # Calculate antenna diagonals
    gains1_times_data = gains1_expanded[np.newaxis, :, :] * data_visibilities
    gains2_times_conj_data = gains2_expanded[np.newaxis, :, :] * np.conj(
        data_visibilities
    )
    ant_diag_part1 = np.sum(
        weight_mat[np.newaxis, :, :, :]
        * gains1_times_data[:, :, np.newaxis, :]
        * np.conj(gains1_times_data[:, :, :, np.newaxis]),
        axis=0,
    )
    ant_diag_part1 = np.einsum("ij,jkl->ikl", gains_exp_mat_2.T, ant_diag_part1)
    ant_diag_part2 = np.sum(
        weight_mat[np.newaxis, :, :, :]
        * gains2_times_conj_data[:, :, np.newaxis, :]
        * np.conj(gains2_times_conj_data[:, :, :, np.newaxis]),
        axis=0,
    )
    ant_diag_part2 = np.einsum("ij,jkl->ikl", gains_exp_mat_1.T, ant_diag_part2)
    ant_diags = ant_diag_part1 + ant_diag_part2
    for ant_ind in range(Nants):
        hess[ant_ind, ant_ind, :, :, 0] = 2 * np.real(ant_diags[ant_ind, :, :])
        hess[ant_ind, ant_ind, :, :, 1] = 2 * np.imag(ant_diags[ant_ind, :, :])
        hess[ant_ind, ant_ind, :, :, 2] = 2 * np.real(ant_diags[ant_ind, :, :])

    if lambda_val != 0.0:  # Apply regularization

        im_part = np.imag(gains) / np.abs(gains) ** 2.0
        real_part = np.real(gains) / np.abs(gains) ** 2.0
        arg_sum = np.sum(np.angle(gains), axis=0)

        # Real-real derivative
        term1_rr = 2 * lambda_val * np.einsum("ik,jk->ijk", im_part, im_part)
        term2_rr = 4 * lambda_val * arg_sum * im_part * real_part

        # Real-imaginary derivative
        term1_ri = -2 * lambda_val * np.einsum("ik,jk->ijk", im_part, real_part)
        term2_ri = 2 * lambda_val * arg_sum * (im_part**2.0 - real_part**2.0)

        # Imaginary-imaginary derivative
        term1_ii = 2 * lambda_val * np.einsum("ik,jk->ijk", real_part, real_part)
        term2_ii = -1 * term2_rr

        hess[:, :, np.arange(Nfreqs), np.arange(Nfreqs), 0] += term1_rr
        hess[:, :, np.arange(Nfreqs), np.arange(Nfreqs), 1] += term1_ri
        hess[:, :, np.arange(Nfreqs), np.arange(Nfreqs), 2] += term1_ii
        for ant_ind in range(Nants):
            hess[ant_ind, ant_ind, np.arange(Nfreqs), np.arange(Nfreqs), 0] += term2_rr[
                ant_ind, :
            ]
            hess[ant_ind, ant_ind, np.arange(Nfreqs), np.arange(Nfreqs), 1] += term2_ri[
                ant_ind, :
            ]
            hess[ant_ind, ant_ind, np.arange(Nfreqs), np.arange(Nfreqs), 2] += term2_ii[
                ant_ind, :
            ]

    hess_reformatted = np.zeros((2, Nants * Nfreqs, 2, Nants * Nfreqs), dtype=float)
    hess_reformatted[0, :, 0, :] = np.transpose(
        hess[:, :, :, :, 0], (0, 2, 1, 3)
    ).reshape(Nants * Nfreqs, Nants * Nfreqs)
    hess_reformatted[0, :, 1, :] = np.transpose(
        hess[:, :, :, :, 1], (0, 2, 1, 3)
    ).reshape(Nants * Nfreqs, Nants * Nfreqs)
    hess_reformatted[1, :, 0, :] = np.transpose(
        hess[:, :, :, :, 1], (1, 3, 0, 2)
    ).reshape(Nants * Nfreqs, Nants * Nfreqs)
    hess_reformatted[1, :, 1, :] = np.transpose(
        hess[:, :, :, :, 2], (0, 2, 1, 3)
    ).reshape(Nants * Nfreqs, Nants * Nfreqs)
    del hess
    hess_reformatted = hess_reformatted.reshape(2 * Nants * Nfreqs, 2 * Nants * Nfreqs)

    return hess_reformatted


def cost_function_sky_cal(
    x,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    weight_mat,
    data_visibilities,
):

    gains = np.reshape(x, (2, Nants, Nfreqs))
    gains = (
        gains[
            0,
        ]
        + 1.0j
        * gains[
            1,
        ]
    )

    gains_expanded = np.matmul(gains_exp_mat_1, gains) * np.matmul(
        gains_exp_mat_2, np.conj(gains)
    )
    res_vec = model_visibilities - gains_expanded[np.newaxis, :, :] * data_visibilities

    cost = np.sum(np.abs(res_vec) ** 2)

    return cost


def get_weight_mat_identity(Nfreqs, Nbls):

    weight_mat = np.identity(Nfreqs)
    weight_mat = np.repeat(weight_mat[np.newaxis, :, :], Nbls, axis=0)
    weight_mat = weight_mat.reshape((Nbls, Nfreqs, Nfreqs))
    return weight_mat


def get_weighted_weight_mat(
    Nfreqs,
    Nbls,
    uvw_array,
    channel_width_hz,
    wedge_slope_factor=0.628479,
    wedge_delay_buffer=6.5e-8,
    downweight_frac=0.0131875,
):

    c = 3.0 * 10**8  # Speed of light
    bl_lengths = np.sqrt(np.sum(uvw_array**2.0, axis=1))
    delay_array = np.fft.fftfreq(Nfreqs, d=channel_width_hz)
    delay_weighting = np.ones((Nbls, Nfreqs))
    for delay_ind, delay_val in enumerate(delay_array):
        wedge_bls = np.where(
            wedge_slope_factor * bl_lengths / c + wedge_delay_buffer > np.abs(delay_val)
        )[0]
        delay_weighting[wedge_bls, delay_ind] = downweight_frac
    freq_weighting = np.fft.ifft(delay_weighting, axis=1)
    weight_mat = np.zeros((Nbls, Nfreqs, Nfreqs), dtype=complex)
    for freq_ind1 in range(Nfreqs):
        for freq_ind2 in range(Nfreqs):
            weight_mat[:, freq_ind1, freq_ind2] = freq_weighting[
                :, np.abs(freq_ind1 - freq_ind2)
            ]

    # Make normalization match identity matrix weight mat
    normalization_factor = Nfreqs * Nbls / np.sum(np.abs(weight_mat))
    weight_mat *= normalization_factor

    return weight_mat


def apply_calibration(
    cal,
    data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
    data_use_model=True,
    obsid="1061316296",
    pol="XX",
    debug_limit_freqs=None,
):

    data = pyuvdata.UVData()
    if data_path.endswith(".uvfits"):
        if pol == "XX":
            pol_int = -5
        elif pol == "YY":
            pol_int = -6
        else:
            print("ERROR: Unknown polarization.")
            sys.exit(1)
        data.read_uvfits(data_path, polarizations=pol_int)
    else:
        data_filelist = [
            "{}/{}".format(data_path, file)
            for file in [
                "vis_data/{}_vis_{}.sav".format(obsid, pol),
                "vis_data/{}_vis_model_{}.sav".format(obsid, pol),
                "vis_data/{}_flags.sav".format(obsid),
                "metadata/{}_params.sav".format(obsid),
                "metadata/{}_settings.txt".format(obsid),
                "metadata/{}_layout.sav".format(obsid),
            ]
        ]
        data.read_fhd(data_filelist, use_model=data_use_model)

    if debug_limit_freqs is not None:
        min_freq_channel = round(data.Nfreqs / 2 - debug_limit_freqs / 2)
        use_frequencies = data.freq_array[
            0, min_freq_channel : round(min_freq_channel + debug_limit_freqs)
        ]
        data.select(frequencies=use_frequencies)

    data_calibrated = pyuvdata.utils.uvcalibrate(
        data, cal, inplace=False, time_check=False
    )
    return data_calibrated


def newtons_method_optimizer(
    x0,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    weight_mat,
    data_visibilities,
    step_size=1.0,
    covergence_condition=0.001,
):

    n_iters = 0
    convergence_iters = 0
    while convergence_iters < 3:
        hess_mat = hess_dw_cal(
            x0,
            Nants,
            Nfreqs,
            Nbls,
            model_visibilities,
            gains_exp_mat_1,
            gains_exp_mat_2,
            weight_mat,
            data_visibilities,
        )
        hess_mat_inv = np.linalg.inv(hess_mat)
        del hess_mat
        jac = jac_dw_cal(
            x0,
            Nants,
            Nfreqs,
            Nbls,
            model_visibilities,
            gains_exp_mat_1,
            gains_exp_mat_2,
            weight_mat,
            data_visibilities,
        )
        x1 = x0 - step_size * np.matmul(hess_mat_inv, jac)
        del hess_mat_inv
        del jac
        cost = cost_function_dw_cal(
            x0,
            Nants,
            Nfreqs,
            Nbls,
            model_visibilities,
            gains_exp_mat_1,
            gains_exp_mat_2,
            weight_mat,
            data_visibilities,
        )
        print(f"Iteration {n_iters}, cost func value: {cost}")
        sys.stdout.flush()
        check_conv = np.max(np.abs(x1 - x0))
        if check_conv < covergence_condition:
            convergence_iters += 1
        else:
            convergence_iters = 0
        x0 = x1
        n_iters += 1

    return x1


def grad_descent_optimizer(
    x0,
    Nants,
    Nfreqs,
    Nbls,
    model_visibilities,
    gains_exp_mat_1,
    gains_exp_mat_2,
    weight_mat,
    data_visibilities,
    covergence_condition=1e-8,
):

    n_iters = 0
    convergence_iters = 0
    while convergence_iters < 10:
        jac = jac_dw_cal(
            x0,
            Nants,
            Nfreqs,
            Nbls,
            model_visibilities,
            gains_exp_mat_1,
            gains_exp_mat_2,
            weight_mat,
            data_visibilities,
        )
        if n_iters == 0:
            step_size = 1.0 / (2 * np.max(jac))  # Don't change gains by more than 1/2
        else:
            step_size = np.abs(np.matmul((x0 - x_prev).T, jac - jac_prev)) / np.sum(
                (jac - jac_prev) ** 2.0
            )
        print(step_size)
        x1 = x0 - step_size * jac
        cost = cost_function_dw_cal(
            x0,
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
        print(f"Iteration {n_iters}, cost func value: {cost}")
        sys.stdout.flush()
        check_conv = np.max(np.abs(x1 - x0))
        if check_conv < covergence_condition:
            convergence_iters += 1
        else:
            convergence_iters = 0

        jac_prev = jac
        x_prev = x0
        x0 = x1
        n_iters += 1

    return x1


def initialize_gains_from_calfile(
    gain_init_calfile,
    Nants,
    Nfreqs,
    antenna_list,
    antenna_names,
    time_ind=0,
    pol_ind=0,
):

    uvcal = pyuvdata.UVCal()
    uvcal.read_calfits(gain_init_calfile)
    gains_init = np.ones((Nants, Nfreqs), dtype=complex)
    cal_ant_names = np.array([uvcal.antenna_names[ant] for ant in uvcal.ant_array])
    for ind, ant in enumerate(antenna_list):
        ant_name = antenna_names[ant]
        cal_ant_ind = np.where(cal_ant_names == ant_name)[0][0]
        gains_init[ind, :] = uvcal.gain_array[cal_ant_ind, 0, :, time_ind, pol_ind]

    return gains_init


def calibration_optimization(
    data,
    model,
    use_wedge_exclusion=False,
    log_file_path=None,
    apply_flags=False,
    xtol=1e-8,
    gain_init_stddev=0.01,
    gain_init_calfile=None,
    use_newtons_method=False,
    use_grad_descent=False,
):

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

    if not np.max(flag_array):  # Check for flags
        apply_flags = False

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
    if gain_init_calfile is None:
        gains_init = np.ones((Nants, Nfreqs), dtype=complex)
    else:
        gains_init = initialize_gains_from_calfile(
            gain_init_calfile,
            Nants,
            Nfreqs,
            antenna_list,
            metadata_reference.antenna_names,
        )

    if gain_init_stddev != 0.0:
        gains_init += np.random.normal(
            0.0,
            gain_init_stddev,
            size=(Nants, Nfreqs),
        ) + 1.0j * np.random.normal(
            0.0,
            gain_init_stddev,
            size=(Nants, Nfreqs),
        )
    # Expand the initialized values
    x0 = np.stack((np.real(gains_init), np.imag(gains_init)), axis=0).flatten()

    start_weight_mat = time.time()
    if use_wedge_exclusion:
        print(f"use_wedge_exclusion=True: Generating wedge excluding covariance matrix")
        sys.stdout.flush()
        weight_mat = get_weighted_weight_mat(
            Nfreqs, Nbls, metadata_reference.uvw_array, metadata_reference.channel_width
        )
    else:
        print(f"use_wedge_exclusion=False: Covariance matrix is the identity")
        sys.stdout.flush()
        weight_mat = get_weight_mat_identity(Nfreqs, Nbls)
    end_weight_mat = time.time()
    print(
        f"Time generating covariance matrix: {(end_weight_mat - start_weight_mat)/60.} minutes"
    )
    sys.stdout.flush()

    if apply_flags:
        # Apply flagging
        print(f"Applying flags to the covariance matrix")
        sys.stdout.flush()
        # Flagging individual times is currently not supported
        flag_array_time_averaged = np.max(flag_array, axis=0)
        frac_flagged = float(np.sum(flag_array_time_averaged)) / float(
            np.prod(np.shape(flag_array_time_averaged))
        )
        print(f"Fraction of the data flagged: {frac_flagged}")
        sys.stdout.flush()
        flag_array_inverted = np.invert(flag_array_time_averaged)
        weight_mat *= (
            flag_array_inverted[:, :, np.newaxis]
            * flag_array_inverted[:, np.newaxis, :]
        )
    else:
        print(
            f"Warning: apply_flags is False. No flags are applied. Data and model may include zeroed visibilities"
        )
        sys.stdout.flush()

    # Minimize the cost function
    start_optimize = time.time()
    if use_newtons_method:
        gains_fit_flattened = newtons_method_optimizer(
            x0,
            Nants,
            Nfreqs,
            Nbls,
            model_visibilities,
            gains_exp_mat_1,
            gains_exp_mat_2,
            weight_mat,
            data_visibilities,
            step_size=1.0,
            covergence_condition=xtol,
        )
    elif use_grad_descent:
        gains_fit_flattened = grad_descent_optimizer(
            x0,
            Nants,
            Nfreqs,
            Nbls,
            model_visibilities,
            gains_exp_mat_1,
            gains_exp_mat_2,
            weight_mat,
            data_visibilities,
            covergence_condition=xtol,
        )
    else:  # Use scipy optimizer
        result = scipy.optimize.minimize(
            cost_function_dw_cal,
            x0,
            args=(
                Nants,
                Nfreqs,
                Nbls,
                model_visibilities,
                gains_exp_mat_1,
                gains_exp_mat_2,
                weight_mat,
                data_visibilities,
            ),
            method="Newton-CG",
            jac=jac_dw_cal,
            hess=hess_dw_cal,
            options={"disp": True, "xtol": xtol},
        )
        print(result.message)
        gains_fit_flattened = result.x
    end_optimize = time.time()
    print(f"Optimization time: {(end_optimize - start_optimize)/60.} minutes")
    sys.stdout.flush()

    gains_fit = np.reshape(gains_fit_flattened, (2, Nants, Nfreqs))
    gains_fit = (
        gains_fit[
            0,
        ]
        + 1.0j
        * gains_fit[
            1,
        ]
    )
    # Ensure that the phase of the gains is mean-zero for each frequency
    avg_angle = np.arctan2(
        np.mean(np.sin(np.angle(gains_fit)), axis=0),
        np.mean(np.cos(np.angle(gains_fit)), axis=0),
    )
    gains_fit *= np.cos(avg_angle) - 1j * np.sin(avg_angle)

    # Create cal object
    cal = initialize_cal(data, antenna_list, gains=gains_fit)

    return cal


def calibrate(
    model_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
    model_use_model=True,
    data_path="/Users/ruby/Astro/FHD_outputs/fhd_rlb_model_GLEAM_Aug2021",
    data_use_model=True,
    obsid="1061316296",
    pol="XX",
    use_autos=False,
    use_wedge_exclusion=False,
    cal_savefile=None,
    calibrated_data_savefile=None,
    log_file_path=None,
    debug_limit_freqs=None,  # Set to number of freq channels to use
    use_antenna_list=None,
    use_flagged_baselines=False,
    apply_flags=False,
    xtol=1e-8,
    gain_init_stddev=0.01,
    gain_init_calfile=None,
    use_newtons_method=False,
    use_grad_descent=False,
):

    if log_file_path is not None:
        stdout_orig = sys.stdout
        stderr_orig = sys.stderr
        sys.stdout = sys.stderr = log_file_new = open(log_file_path, "w")

    start = time.time()

    start_read_data = time.time()
    data, model = get_test_data(
        model_path=model_path,
        model_use_model=model_use_model,
        data_path=data_path,
        data_use_model=data_use_model,
        obsid=obsid,
        pol=pol,
        use_autos=use_autos,
        debug_limit_freqs=debug_limit_freqs,
        use_antenna_list=use_antenna_list,
        use_flagged_baselines=use_flagged_baselines,
    )
    end_read_data = time.time()
    print(f"Time reading data: {(end_read_data - start_read_data)/60.} minutes")
    sys.stdout.flush()

    cal = calibration_optimization(
        data,
        model,
        use_wedge_exclusion=use_wedge_exclusion,
        log_file_path=log_file_path,
        apply_flags=apply_flags,
        xtol=xtol,
        gain_init_stddev=gain_init_stddev,
        gain_init_calfile=gain_init_calfile,
        use_newtons_method=use_newtons_method,
        use_grad_descent=use_grad_descent,
    )

    if cal_savefile is not None:
        print(f"Saving calibration solutions to {cal_savefile}")
        sys.stdout.flush()
        cal.write_calfits(cal_savefile, clobber=True)

    # Apply calibration
    if calibrated_data_savefile is not None:
        calibrated_data = apply_calibration(
            cal,
            data_path=data_path,
            data_use_model=data_use_model,
            obsid=obsid,
            pol=pol,
            debug_limit_freqs=debug_limit_freqs,
        )
        print(f"Saving calibrated data to {calibrated_data_savefile}")
        sys.stdout.flush()
        calibrated_data.write_uvfits(calibrated_data_savefile)

    end = time.time()
    print(f"Total runtime: {(end - start)/60.} minutes")

    if log_file_path is not None:
        sys.stdout = stdout_orig
        sys.stderr = stderr_orig
        log_file_new.close()
        
