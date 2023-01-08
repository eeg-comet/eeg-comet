# TESS Algorithm (Matching with Microstate Maps) https://linkinghub.elsevier.com/retrieve/pii/S1053-8119(14)00243-2
# Authors: Raaj Chatterjee, Amin Kabir, 2022 ebrain lab

# This script requires the following
# Python version 3.7+ required to fetch template files
# Packages required (from pip install): mne
# Optional packages for visualization: ipython, pyvista, pyvistaqt, ipywidgets

# Basic steps for source localization:
# 1. Compute forward model
# 2. Compute noise covariance matrix
# 3. Compute inverse model to extract source time-courses

# Package imports
import os.path
import numpy as np
from scipy import stats
import pandas as pd
import pickle
import h5py
from joblib import Parallel, delayed 
import tempfile

# MNE imports
import mne
mne.viz.set_3d_backend("pyvista")
from mne.datasets import eegbci
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import (make_inverse_operator, apply_inverse_raw)


from mayavi import mlab
#mlab.init_notebook()

def load_average_mri(spacing):
    # Load Average MRI - warning, patient-specific MRI is more accurate
    # Download fsaverage files if they don't exist
    fs_dir = fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)

    # The files live in:
    subject = 'fsaverage'
    trans = 'fsaverage'  # MNE has a built-in fsaverage transformation
    # src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif') # Use default src space
    # ico3: 642 sources/hemisphere
    # oct5: 1026 sources/hemisphere
    # ico4 - 2562 sources/hemisphere
    # oct6: 4098 sources/hemisphere
    # ico5: 10242 sources/hemisphere
    src = mne.setup_source_space(subject, spacing=spacing,
                                 subjects_dir=subjects_dir,
                                 add_dist=False, n_jobs=-1)
    bem = os.path.join(fs_dir, 'bem', 'fsaverage-5120-5120-5120-bem-sol.fif') # Use default src space
    return src, bem, trans


def extract_forward_transform(eeg_info, spacing):
    # Load average mri data
    src, bem, trans = load_average_mri(spacing)

    # Check that the locations of EEG electrodes is correct with respect to MRI
    # mne.viz.plot_alignment(
    #     eeg_info, src=src, eeg=['original', 'projected'], trans=trans,
    #     show_axes=True, mri_fiducials=True, dig='fiducials')

    # Calculate forward solution
    fwd = mne.make_forward_solution(eeg_info, trans=trans, src=src,
                                    bem=bem, eeg=True, mindist=5.0, n_jobs=-1)
    leadfield = fwd['sol']['data']
    print("Leadfield size : %d sensors x %d dipoles" % leadfield.shape)

    fwd_fixed = mne.convert_forward_solution(fwd, surf_ori=True, force_fixed=True,
                                             use_cps=True)
    leadfield_fixed = fwd_fixed['sol']['data']
    print("Leadfield size : %d sensors x %d dipoles" % leadfield_fixed.shape)
    return fwd_fixed, leadfield_fixed, fwd


def load_eeg_data(eeg_info_path="eeg_info.pkl", eeg_data_path='test_study.h5'):
    with open(eeg_info_path, 'rb') as f:
        eeg_info = pickle.load(f)

    hf_group = h5py.File(eeg_data_path, 'r')
    study_name = list(hf_group.keys())[0]
    file_names = list(hf_group[study_name].keys())

    for filename in file_names:
        print(study_name, filename)
        data = np.asarray(hf_group[study_name][filename][:])*pow(10,6)

    return data, eeg_info


def load_microstate_topos(microstate_maps_path):
    microstate_maps = pd.read_csv(microstate_maps_path)

    map_a = microstate_maps['A'].values
    map_b = microstate_maps['B'].values
    map_c = microstate_maps['C'].values
    map_d = microstate_maps['D'].values
    map_e = microstate_maps['E'].values
    all_maps = np.vstack((map_a, map_b, map_c, map_d, map_e))
    return np.transpose(all_maps)


def load_eegbci_eeg():
    # Load sample data
    raw_fname, = eegbci.load_data(subject=1, runs=[6])
    raw = mne.io.read_raw_edf(raw_fname, preload=True)
    # Clean channel names to be able to use a standard 1005 montage
    new_names = dict(
        (ch_name,
         ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
        for ch_name in raw.ch_names)
    raw.rename_channels(new_names)
    # Read and set the EEG electrode locations, which are already in fsaverage's
    # space (MNI space) for standard_1020:
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage)
    raw.set_eeg_reference(projection=True)  # needed for inverse modeling
    return raw

def inverse_source(sensor_raw, fwd, inv_method):
    # (events, event_id) = mne.events_from_annotations(sensor_raw)
    # epochs = mne.Epochs(sensor_raw, events, event_id['T1'], -2, 5)
    #cov_method = 'empirical'
    noise_cov = mne.compute_raw_covariance(sensor_raw, method='auto')
    #mne.viz.plot_cov(noise_cov, sensor_raw.info)

    inverse_operator = make_inverse_operator(sensor_raw.info, fwd, noise_cov)
    #inv_method = "dSPM"
    snr = 3.
    lambda2 = 1. / snr ** 2

    stc = apply_inverse_raw(sensor_raw, inverse_operator, lambda2,
                        method=inv_method, pick_ori=None, verbose=True)
    #brain = stc.plot(surface='inflated', hemi='rh')
    return stc


def dummy_scalp_maps(size):
    # define random scalp map - replace with real maps
    maps = np.vander((1, 1.1, 1.2), size).transpose()
    return maps


def find_t_coeff(sample, maps):
    # Run least squares on the microstate maps,
    # Residuals not included but can be added (res = np.linalg.lstsq(...)[1])
    t_coeff = np.linalg.lstsq(maps, sample, rcond=None)[0]
    return t_coeff


def first_regression(sensor_time_series, maps):
    # Load maps
    # maps1 = dummy_scalp_maps(sensor_time_series.shape[0])
    # Extract T coefficients
    all_t_coeff = np.apply_along_axis(find_t_coeff, 0, sensor_time_series, maps).transpose()
    return all_t_coeff


def second_regression(t_coeff, source_time_series):
    # source time series should have dimensions Time x Sources
    # beta_coeff = np.linalg.lstsq(t_coeff, source_time_series, rcond=None)[0]
    beta_coeff = np.linalg.solve(t_coeff.T @ t_coeff, t_coeff.T @ source_time_series)
    #beta_coeff = sp_lstsq(t_coeff, source_time_series, lapack_driver='gelsy', check_finite=False)[0]
    return beta_coeff


def run_tess(eeg_info_path, eeg_data_path, microstate_maps_path, inv_method, nperm, spacing):
    # Load sensor time series - replace with real data
    # sensor_raw = load_eegbci_eeg()
    # Ensure data is channels x time points (could be improved)
    # if sensor_raw._data.shape[0] < sensor_raw._data.shape[1]:
    #     sensor_time_series = sensor_raw._data
    # else:
    # sensor_time_series = np.transpose(sensor_raw._data)
    eeg_data, eeg_info = load_eeg_data(eeg_info_path, eeg_data_path)
    eeg_raw = mne.io.RawArray(eeg_data, eeg_info)
    eeg_raw.set_eeg_reference('average', projection=True)
     # to extract proper inverse models
    (fwd, leadfield, fwd_free) = extract_forward_transform(eeg_info, spacing)
    source_time_series = np.matmul(np.transpose(eeg_data), leadfield)

    # Source inverse space run
    stc = inverse_source(eeg_raw, fwd_free, inv_method)
    source_time_series = stc.data
    source_time_series = np.transpose(source_time_series)

    # Run first (spatial) regression
    maps = load_microstate_topos(microstate_maps_path)
    t_coeff = first_regression(eeg_data, maps)

    # Run second (temporal) regression
    beta_coeff = second_regression(t_coeff, source_time_series)

    # Permutation of beta over t to determine significance
    #nperm = 20 # Change to 2000
    z_scores = np.zeros(beta_coeff.shape)
    beta_dist = np.zeros((beta_coeff.shape[0], beta_coeff.shape[1], nperm))
    bonferroni = beta_coeff.shape[1]
    t_shuffle = t_coeff
    path = tempfile.mkdtemp()
    beta_path = os.path.join(path,'beta_dist.mmap')
    beta_dist = np.memmap(beta_path, dtype=float, shape=(np.size(maps)[1], np.size(source_time_series)[1], nperm), mode='w+')
    
    def process(t_shuffle, source_time_series, ii):
        np.random.shuffle(t_shuffle)
        beta_dist[:,:,ii] = second_regression(t_shuffle, source_time_series)
        print(ii + " Parallel")
    '''
    #for ii in range(0, nperm):
    #    np.random.shuffle(t_shuffle)
    #    beta_dist[:, :, ii] = second_regression(t_shuffle, source_time_series)
    #    print(ii)
    '''
    Parallel(n_jobs =-1)(delayed(process)(t_shuffle, source_time_series, ii) for ii in range(0, nperm) )

    for idx, x in np.ndenumerate(beta_coeff):
        z_scores[idx] = stats.zscore(np.insert(beta_dist[idx[0]][idx[1]][:], 0, x))[0]

    p_values = bonferroni * stats.norm.sf(abs(z_scores))
    print(p_values)

    # Source visualization:
    mystc = mne.SourceEstimate(np.transpose(z_scores), [np.arange(2562), np.arange(2562)], 0, 1)

    mystc.subject = 'fsaverage'
    mystc.plot(subjects_dir=None, initial_time=1,
               clim=dict(kind='value', pos_lims=[3, 6, 9]),
               time_viewer=True)

def run_source_localization(study_name, eeg_info, eeg_data_path, microstate_maps,
                            inv_method, nperm, spacing):

    hf_group = h5py.File(eeg_data_path, 'r')
    file_names = list(hf_group[study_name].keys())

    counter = 0
    for filename in file_names:
        counter += 1
        print(study_name, filename)
        eeg_data = np.asarray(hf_group[study_name][filename][:]) * pow(10, 6)
        eeg_raw = mne.io.RawArray(eeg_data, eeg_info)
        ch_name = eeg_info.ch_names
        new_names = dict(
            (ch_name,
             ch_name.rstrip('.').upper().replace('Z', 'z').replace('FP', 'Fp'))
            for ch_name in eeg_raw.ch_names)
        eeg_raw.rename_channels(new_names)
        # Read and set the EEG electrode locations, which are already in fsaverage's
        # space (MNI space) for standard_1020:
        montage = mne.channels.make_standard_montage('standard_1020')
        eeg_raw.set_montage(montage)
        eeg_raw.set_eeg_reference('average', projection=True)

        # to extract proper inverse models
        (fwd, leadfield, fwd_free) = extract_forward_transform(eeg_info, spacing)
        source_time_series = np.matmul(np.transpose(eeg_data), leadfield)

        # Source inverse space run
        stc = inverse_source(eeg_raw, fwd_free, inv_method)
        source_time_series = stc.data
        source_time_series = np.transpose(source_time_series)
        # source_time_series = np.load('source_time_series.npy')

        # Run first (spatial) regression
        t_coeff = first_regression(eeg_data, microstate_maps)

        # Run second (temporal) regression
        beta_coeff = second_regression(t_coeff, source_time_series)

        # Permutation of beta over t to determine significance
        z_scores = np.zeros(beta_coeff.shape)
        beta_dist = np.zeros((beta_coeff.shape[0], beta_coeff.shape[1], nperm))
        bonferroni = beta_coeff.shape[1]
        t_shuffle = t_coeff

        for ii in range(0, nperm):
            print("Running ", str(nperm), "permutations ...")
            np.random.shuffle(t_shuffle)
            beta_dist[:, :, ii] = second_regression(t_shuffle, source_time_series)
            print(ii)

        for idx, x in np.ndenumerate(beta_coeff):
            z_scores[idx] = stats.zscore(np.insert(beta_dist[idx[0]][idx[1]][:], 0, x))[0]

        p_values = bonferroni * stats.norm.sf(abs(z_scores))
        #sig_ind = p_values < 0.05
        #sig_scores = np.multiply(z_scores, sig_ind)

        # Source visualization:
        for i in range(np.size(z_scores, 0)):
            z_scores[i, :] = 2. * (z_scores[i, :] - np.min(z_scores[i, :])) / np.ptp(z_scores[i, :]) - 1

        if counter == 1:
            avg_stc_data = z_scores
        else:
            avg_stc_data = avg_stc_data + z_scores
    avg_stc_data = avg_stc_data / counter
    print("Source localization is done!")
    return avg_stc_data


def visualize_sources(avg_stc_data, spacing):
    mystc = mne.SourceEstimate(np.transpose(avg_stc_data),
                       [np.arange(np.size(avg_stc_data, 1)/2), np.arange(np.size(avg_stc_data, 1)/2)], 0, 1)

    mystc.subject = 'fsaverage'
    brain = mystc.plot(subjects_dir=None,
                       initial_time=0,
                       cortex='high_contrast',
                       hemi='split',
                       surface='inflated',
                       clim=dict(kind='value', lims=[0, 0.5, 1]),
                       time_viewer=True,
                       background='white',
                       spacing=spacing,
                       colormap='jet',
                       smoothing_steps=15)
    input("Visualize Microstate Sources")
