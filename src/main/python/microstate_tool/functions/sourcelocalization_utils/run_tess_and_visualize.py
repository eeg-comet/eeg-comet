### Using this file for TESS Amin Paper


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
import pandas as pd
from scipy import stats
from fnmatch import fnmatch
import pickle
# from functions.data_utils.data_io import find_data, load_eegs, get_eeg_data
# from functions.sourcelocalization_utils.stc_io import stc_read, stc_write

import mne
# mne.viz.set_3d_backend("pyvista")

# from mayavi import mlab
#mlab.init_notebook()

from joblib import Parallel, delayed

def find_data(input_folder, extension, pattern='*'):
    list_path = []  # Initialize an empty list to store matching file paths
    list_filename = []

    # Walk through the directory tree rooted at input_folder
    for path, subdirs, files in os.walk(input_folder):
        for name in files:
            # Check if the file name matches the pattern with the specified extension
            if fnmatch(name, pattern + extension):
                # Add the matching file's full path to the list
                list_path.append(os.path.join(path, name))
                list_filename.append(name.split('.')[0])
    return list_path, list_filename

def load_eegs(self, filename, eeg_format, datatype, channel_location_dir='', chan2rm=[]):
    """
    Load EEG data from different formats and preprocess if needed.

    Inputs:
    - filename (str): Path to the EEG data file.
    - eeg_format (str): Format of the EEG data file (e.g., '.edf', '.fif', '.set').
    - datatype (str): Type of EEG data ('raw' for continuous data or 'epoched' for segmented data).
    - channel_location_dir (str): Directory path for channel location information (optional).
    - chan2rm (list): List of channel names to be removed (optional).

    Outputs:
    - eeg_data (mne.Raw or mne.Epochs): Loaded and optionally preprocessed EEG data.
    """

    if datatype == 'raw':

        # Load EEG data based on the specified format
        if eeg_format == ".vhdr":
            eeg = mne.io.read_raw_brainvision(filename, preload=True, verbose="WARNING")
        elif eeg_format == ".edf":
            eeg = mne.io.read_raw_edf(filename, preload=True, verbose="WARNING")
        elif eeg_format == ".bdf":
            eeg = mne.io.read_raw_bdf(filename, preload=True, verbose="WARNING")
        elif eeg_format == ".gdf":
            eeg = mne.io.read_raw_gdf(filename, preload=True, verbose="WARNING")
        elif eeg_format == ".cnt":
            eeg = mne.io.read_raw_cnt(filename, preload=True, verbose="WARNING")
        elif eeg_format == ".egi" or eeg_format == ".mff":
            eeg = mne.io.read_raw_egi(filename, preload=True, verbose="WARNING")
        elif eeg_format == ".set":
            eeg = mne.io.read_raw_eeglab(filename, preload=True, verbose="CRITICAL")
        elif eeg_format == ".data":
            eeg = mne.io.read_raw_nicolet(filename, preload=True, verbose="WARNING")
        elif eeg_format == ".nxe":
            eeg = mne.io.read_raw_eximia(filename, preload=True, verbose="WARNING")
        elif eeg_format == ".lay":
            eeg = mne.io.read_raw_persyst(filename, preload=True, verbose="WARNING")
        elif eeg_format == ".eeg":
            eeg = mne.io.read_raw_nihon(filename, preload=True, verbose="WARNING")
    elif datatype == 'epoched':
        if eeg_format == ".set":
            eeg = mne.io.read_epochs_eeglab(filename, verbose="WARNING")

    # Optional: Load channel locations if provided
    if os.path.isfile(channel_location_dir):
        montage = mne.channels.read_custom_montage(channel_location_dir)
        eeg.set_montage(montage, match_case=False, on_missing='warn', verbose="WARNING")
    elif channel_location_dir in mne.channels.get_builtin_montages():
        montage = mne.channels.make_standard_montage(channel_location_dir)
        eeg.set_montage(montage, match_case=False, on_missing='warn', verbose="WARNING")

    # Pick channels
    eeg = eeg.pick_types(meg=False, eeg=True, eog=False,
                         exclude=chan2rm, verbose="WARNING")

    # Add average reference projection
    eeg.set_eeg_reference('average', projection=True, verbose="WARNING")

    # Apply the added projection
    eeg.apply_proj(verbose="WARNING")

    return eeg


def stc_write(stc_data_subject_path, stc_file):
    # Write source time series to disk
    for idx, stc in enumerate(stc_file):
        filename = f'stc_{idx}'
        filepath = os.path.join(stc_data_subject_path, filename)
        stc.save(filepath, ftype='h5')


def stc_read(stc_data_subject_path):
    stc_list, _ = find_data(stc_data_subject_path, '.stc', pattern='*')
    # Read source time series from disk
    stc_file = []
    for stc_path in stc_list:
        stc = mne.read_source_estimate(stc_path)
        stc_file.append(stc)
    return stc_file


def load_average_mri(spacing='ico5'):
    """
    Load the standard template MRI subject "fsaverage" in MNE.
    The default "ico5" source space is used unless a different spacing is provided.

    Parameters:
        spacing (str): The spacing parameter for the source space. Default is 'ico5'.

    Returns:
        src (mne.SourceSpaces): The source space for the fsaverage subject.
        bem (str): The path to the BEM (Boundary Element Model) file for fsaverage subject.
        trans (str): The transformation for the fsaverage subject.
    """
    
    # Print information about using the standard template MRI subject -fsaverage-
    print('\nUsing the standard template MRI subject -fsaverage-')
    print('\nWarning, patient-specific MRI is more accurate!')

    # Download fsaverage files if they don't exist
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)
    subjects_dir = os.path.dirname(fs_dir)

    # The files live in:
    subject = "fsaverage"
    trans = "fsaverage"  # MNE has a built-in fsaverage transformation

    # Check if a different spacing is provided, then create a new source space
    if not spacing == 'ico5':
        src = mne.setup_source_space(subject,
                                     spacing=spacing,
                                     subjects_dir=subjects_dir,
                                     add_dist=False,
                                     n_jobs=-1)
    else:
        # Use the default 'ico5' source space
        src = os.path.join(fs_dir, 'bem', 'fsaverage-ico-5-src.fif')

    # Specify the path to the BEM file
    bem = os.path.join(fs_dir, "bem", "fsaverage-5120-5120-5120-bem-sol.fif")

    return src, bem, trans


def individual_mri(subjects_dir, subject, eeg_info, spacing):
    """
    Perform individual MRI coregistration for a specific subject.

    Parameters:
        subjects_dir (str): Path to the directory where subject-specific data is stored.
        subject (str): The subject identifier for whom the coregistration is being performed.
        eeg_info (mne.Info): The measurement info for the subject's raw data.

    Returns:
        src (mne.SourceSpaces): The source space for the subject's MRI.
        bem (mne.bem.ConductorModel): The BEM model for the subject.
        trans (mne.transforms.Transform): The transformation matrix obtained from the coregistration.
    """
    
    print(f"\nUsing the individual MRI subject: {subject}")
    print('\nThis may take some time to compute ...')

    # Setting up the source space
    src = mne.setup_source_space(subject=subject,
                                 spacing=spacing,
                                 subjects_dir=subjects_dir,
                                 n_jobs=-1)
    mne.write_source_spaces(subject + '-' + spacing + '-src.fif', src, overwrite=True)

    # Setting up the boundary-element model
    model = mne.make_bem_model(subject=subject, subjects_dir=subjects_dir, ico=spacing[-1])
    mne.write_bem_surfaces(subject + '-5120-5120-5120-bem.fif', model, overwrite=True)

    # Computing the BEM solution
    bem = mne.make_bem_solution(model, solver='mne')
    mne.write_bem_solution(subject + '-5120-5120-5120-bem-sol.fif', bem, overwrite=True)

    # Aligning coordinate frames

    # Get MNI fiducials for the subject
    fiducials = mne.coreg.get_mni_fiducials(subject=subject,
                                            subjects_dir=subjects_dir)
    
    # Perform coregistration based on fiducials and measurement info
    coreg = mne.coreg.Coregistration(info=eeg_info,
                                     subject=subject,
                                     subjects_dir=subjects_dir,
                                     fiducials=fiducials)
    coreg.fit_icp(n_iterations=20, nasion_weight=10.0, verbose=True)
    
    # Omit head shape points that are too close to the MRI surface
    coreg.omit_head_shape_points(distance=5.0 / 1000)  # distance is in meters
    
    # Compute distances between digitization points and MRI surface in millimeters
    dists = coreg.compute_dig_mri_distances() * 1e3
    print(f"Distance between HSP and MRI (mean/min/max): {np.mean(dists):.2f} mm / {np.min(dists):.2f} mm / {np.max(dists):.2f} mm")
    
    # Save the transformation matrix to a file
    trans_file = os.path.join(subjects_dir, subject, 'mri', 'transforms', f'{subject}-trans.fif')
    trans = coreg.trans
    mne.write_trans(trans_file, trans, overwrite=True)

    return src, bem, trans


def extract_forward_transform(src, bem, trans, data_type, eeg, eeg_info, inv_method):
    """
    Extract the forward solution and apply minimum-norm inverse to obtain the source time series.

    Parameters:
        src (mne.SourceSpaces): The source space.
        bem (mne.bem.ConductorModel): The Boundary Element Model (BEM) for the forward solution.
        trans (mne.transforms.Transform): The transformation matrix for coregistration.
        data_type (str): The type of data ('raw' or 'epoched') to process.
        eeg (mne.io.Raw): The raw data or epoched data.
        eeg_info (mne.Info): The measurement info for the raw or epoched data.
        inv_method (str): The inverse method to be used.

    Returns:
        stc (mne.SourceEstimate): The source estimate containing the inverse solution.
        stc_data (numpy.ndarray): The source time series data.
    """
    
    print('\nPerforming source localization ...')
    
    # Calculate the forward solution using the specified parameters
    fwd = mne.make_forward_solution(eeg_info, trans, src,
                                    bem, eeg=True, mindist=5.0, n_jobs=-1)

    # Compute noise covariance from the raw data
    noise_cov = mne.compute_covariance(eeg, method='auto', verbose=True, n_jobs=-1)
    # Regularize noise covariance to avoid singularity issues
    noise_cov = mne.cov.regularize(noise_cov, eeg_info,
                                   mag=0.1, grad=0.1, eeg=0.1, proj=True)

    # Create the inverse operator
    inverse_operator = mne.minimum_norm.make_inverse_operator(eeg_info, fwd, noise_cov)

    # Set the regularization parameter for the inverse solution based on the signal-to-noise ratio (snr)
    snr = 3.
    lambda2 = 1. / snr ** 2

    # Apply minimum-norm inverse to obtain the source time series
    if data_type == 'raw':
        # Process raw data
        stc = mne.minimum_norm.apply_inverse_raw(eeg, inverse_operator, lambda2,
                                                method=inv_method, pick_ori=None, verbose=True)

    elif data_type == 'epoched':
        # Process epoched data
        stc = mne.minimum_norm.apply_inverse_epochs(eeg, inverse_operator, lambda2,
                                                   method=inv_method, pick_ori=None, verbose=True)

    return stc


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


def second_regression(t_coeff, stc_data):
    # source time series should have dimensions Time x Sources
    # beta_coeff = np.linalg.lstsq(t_coeff, stc_data, rcond=None)[0]
    beta_coeff = np.linalg.solve(t_coeff.T @ t_coeff, t_coeff.T @ stc_data)
    #beta_coeff = sp_lstsq(t_coeff, stc_data, lapack_driver='gelsy', check_finite=False)[0]
    return beta_coeff


# TESS Algorithm for source localization
def tess_algorithm(eeg_data, microstate_maps, stc_data,
                   nperm):
    """
    Apply the TESS algorithm for source localization and save the results.
    https://linkinghub.elsevier.com/retrieve/pii/S1053-8119(14)00243-2
    
    Parameters:
        eeg_data (numpy.ndarray): The EEG data.
        microstate_maps (numpy.ndarray): The microstate maps.
        stc_data (numpy.ndarray): The source time series data.
        nperm (int): Number of permutations for significance testing.
        z_scores_path (str): The path to save the z-scores.
        p_values_path (str): The path to save the p-values.

    Returns:
        sum_z_scores (numpy.ndarray): The sum of z-scores (used for averaging later).
    """
    
    # z_scores_path = os.path.join(tess_path, "z_scores")
    # p_values_path = os.path.join(tess_path, "p_values")
    # if not os.path.exists(z_scores_path):
    #     os.makedirs(z_scores_path)
    # if not os.path.exists(p_values_path):
    #     os.makedirs(p_values_path)
    
    t_coeff = first_regression(eeg_data, microstate_maps)
    beta_coeff = second_regression(t_coeff, stc_data)

    # Permutation of beta over t to determine significance
    z_scores = np.zeros(beta_coeff.shape)
    beta_dist = np.zeros((beta_coeff.shape[0], beta_coeff.shape[1], nperm))
    bonferroni = beta_coeff.shape[1]
    t_shuffle = t_coeff
    # Parallelize this next for loop for speed
    # for ii in range(0, nperm):
    #     np.random.shuffle(t_shuffle)
    #     beta_dist[:, :, ii] = second_regression(t_shuffle, stc_data)
    #     if (ii % 50) == 0:
    #         print(ii)

    # How can I parallelize this?
    # https://stackoverflow.com/questions/9786102/how-do-i-parallelize-a-simple-python-loop

    for ii in range(0, nperm):
        np.random.shuffle(t_shuffle)
        beta_dist[:, :, ii] = second_regression(t_shuffle, stc_data)
        if (ii % 50) == 0:
            print(ii)

    for idx, x in np.ndenumerate(beta_coeff):
        z_scores[idx] = stats.zscore(np.insert(beta_dist[idx[0]][idx[1]][:], 0, x))[0]
    p_values = bonferroni * stats.norm.sf(abs(z_scores))

    # Save results for each file
    # np.save(os.path.join(z_scores_path, rawfilename + "_z_scores"), z_scores)
    # np.save(os.path.join(p_values_path, rawfilename + "_p_values"), p_values)

    # Filter z-scores
    significance = 0.005  # assuming the nperm=2000
    filtered_z_scores = (p_values < significance) * z_scores
    
    return p_values, z_scores, filtered_z_scores


def run_source_localization(preprocessed_data_path,
                            labelled_data_path,
                            localized_sources_path,
                            subjects_dir,
                            microstate_maps, inv_method, nperm, spacing,
                            source_localization_method, extension, data_type):
    """
    Run the source localization process with various options.
    
    Parameters:
    data_type (str): The type of data to process. Valid options are 'raw' or 'epoched'.
    preprocessed_data_path (str): The path to the preprocessed data files.
    labelled_data_path (str): The path to the directory containing labelled data.
    localized_sources_path (str): The path to save the localized source results.
    subjects_dir (str): Path to the directory where subject-specific data is stored.
    eeg_info (mne.Info): The measurement info for the EEG data.
    microstate_maps (numpy.ndarray): The microstate maps.
    inv_method (str): The inverse method to be used. Valid options are 'MNE', 'dSPM', 'sLORETA', or 'eLORETA'.
    nperm (int): Number of permutations for significance testing. Default is 2000.
    spacing (str): The spacing parameter for the source space. Valid options are 'ico3', 'ico4', or 'ico5'.
    - 'ico3': 642 sources/hemisphere
    - 'ico4': 2562 sources/hemisphere
    - 'ico5': 10242 sources/hemisphere
    source_localization_method (str): The source localization method. Valid options are 'avg' or 'tess'.
    
    Returns:
    None
    """
    
    stc_data_path = os.path.join(localized_sources_path, "stc_data")
    # Create a new directory because it does not exist
    if not os.path.exists(stc_data_path):
        os.makedirs(stc_data_path)
    

    list_eeg_path, list_eeg_names = find_data(preprocessed_data_path, extension, '*')

    counter = 0
    for eeg_path in list_eeg_path:
        counter += 1
        rawfilename = os.path.split(eeg_path)[1].split('.')[0]
        print("Source Localizing Microstates", rawfilename)
        print("\n", 100 * counter / len(list_eeg_path))

        # Load the EEG data
        # hf = h5py.File(filename, "r")
        # eeg_data = hf[list(hf.keys())[0]]
        eeg = load_eegs(eeg_path, extension, data_type)
        eeg_data = get_eeg_data(eeg, data_type)
        # eeg_data = np.asarray(eeg_data) * pow(10, 6)

        # Create new eeg structure
        #raw = load_eegs(filename, eeg_format, data_type, '', [])
        # raw = mne.io.RawArray(eeg_data, eeg_info)
        # raw.set_eeg_reference('average', projection=True)
        eeg_info = eeg.info

        if subjects_dir == 'fsaverage':
            # Clean channel names to be able to use a standard 1005 montage
            new_names = dict(
                (ch_name, ch_name.rstrip(".").upper().replace("Z", "z").replace("FP", "Fp"))
                for ch_name in eeg.ch_names
            )
            eeg.rename_channels(new_names)

            # Read and set the EEG electrode locations, which are already in fsaverage's
            # space (MNI space) for standard_1020:
            montage = mne.channels.make_standard_montage("standard_1005")
            eeg.set_montage(montage)
            eeg.set_eeg_reference(projection=True)  # needed for inverse modeling
            
            # Load Adult Template MRI
            src, bem, trans = load_average_mri(spacing)

            # Source inverse space run
            stc_file = extract_forward_transform(src, bem, trans, data_type, eeg, eeg_info, inv_method)

            # Export stc
            stc_data_subject_path = os.path.join(stc_data_path, 'fsaverage')
            if not os.path.exists(stc_data_subject_path):
                os.makedirs(stc_data_subject_path)
            stc_write(stc_data_subject_path, stc_file)

        else:
            subjects_list = [name for name in os.listdir(subjects_dir) if os.path.isdir(os.path.join(subjects_dir, name))]

            for subject in subjects_list:
                src, bem, trans = individual_mri(subjects_dir, subject, eeg_info, spacing)

                # Source inverse space run
                stc_file = extract_forward_transform(src, bem, trans, data_type, eeg, eeg_info, inv_method)

                # Export stc
                stc_data_subject_path = os.path.join(stc_data_path, subject)
                if not os.path.exists(stc_data_subject_path):
                    os.makedirs(stc_data_subject_path)
                stc_write(stc_data_subject_path, stc_file)


def identify_microstates_sources(stc_file, labelled_data_path,
                            localized_sources_path,
                            microstate_maps, nperm,
                            method):
    if method == 'avg':
        # Average sources over times matched with each microstate
        print('\nAveraging sources over times matched with each microstate ...')
        avg_sources_path = os.path.join(localized_sources_path, "avg_sources")
        if not os.path.exists(avg_sources_path):
            os.makedirs(avg_sources_path)

        list_segmented_data, _ = find_data(labelled_data_path, '.csv', pattern='*')
        for segmented_path in list_segmented_data:
            segment_data = pd.read_csv(segmented_path, header=0)
            filename, _ = os.path.splitext(os.path.basename(segmented_path))
            for m in segment_data['segmentation'].unique():
                sources_m_times = segment_data.index[segment_data['segmentation'] == m].tolist()
                sources_m = stc_file[sources_m_times, :]
                sources_m = np.mean(sources_m, axis=0)
                np.save(os.path.join(avg_sources_path, filename + "_sources_" + m), sources_m)


    elif method == 'tess':
        # TODO: modify
        """
        tess_path = os.path.join(localized_sources_path, "tess_sources")
        # TESS Algorithm
        # https://linkinghub.elsevier.com/retrieve/pii/S1053-8119(14)00243-2
        print('\nExtracting sources associated with each microstate',
              '\nusing the topographic electrophysiological state source-imaging (TESS) algorithm ...')
        if not os.path.exists(tess_path):
            os.makedirs(tess_path)

        p_values, z_scores, filtered_z_scores = tess_algorithm(rawfilename,
                                                             eeg_data,
                                                             microstate_maps,
                                                             stc_data,
                                                             nperm,
                                                             tess_path)

        if counter == 1:
            sum_z_scores = filtered_z_scores
        else:
            sum_z_scores = sum_z_scores + filtered_z_scores

        # Save the averaged z-scores
        avg_z_scores = np.divide(sum_z_scores, len(file_names))
        np.save(os.path.join(localized_sources_path, "avg_z_scores"), avg_z_scores)
        """
    else:
        print("\nError: Invalid source localization method. Please use 'avg' or 'tess'.")


def load_data_run_tess(parent_path):
    parent_path = '/Users/chattermac/Downloads/OutputEEGSource'
    list_eeg_path, list_eeg_name = find_data(os.path.join(parent_path, 'EEG Source', 'preprocessed_data'), '.set', '*')

    microstate_maps_df = pd.read_csv(os.path.join(parent_path, 'EEG Source', 'raw_features', 'microstate_maps.csv'), header=0)
    microstate_maps = np.asarray(microstate_maps_df.iloc[:, 1:])

    counter = 0
    for idx in range(len(list_eeg_path)):
        counter += 1
        print("Source Localizing ", list_eeg_name[idx])
        print("\n", 100 * counter / len(list_eeg_path))

        print('\nLoading EEG ...')
        eeg_path = list_eeg_path[idx]
        eeg = mne.io.read_raw_eeglab(eeg_path, preload=True)
        eeg.set_eeg_reference('average', projection=True)
        eeg.apply_proj()
        eeg_data = eeg.get_data()
        eeg_info = eeg.info

        stc_data = mne.read_source_estimate(os.path.join(parent_path, 'stc', f'{list_eeg_name[idx]}-stc.h5'))

        print(f'EEG data {eeg_data.shape}')
        print(f'STC data {stc_data.shape}')
        p, z, z_filt = tess_algorithm(eeg_data, microstate_maps, stc_data.data.T, 2000)

        print('\nTess Completed on File ...')

        np.save(os.path.join(parent_path, 'stc', f'{list_eeg_name[idx]}-filtered_zscore.npy'), z_filt)
        np.save(
            os.path.join(parent_path, 'stc', f'{list_eeg_name[idx]}-zscore.npy'),
            z)
        np.save(
            os.path.join(parent_path, 'stc', f'{list_eeg_name[idx]}-pval.npy'),
            p)


def normalize_and_average_data(parent_path):
    # find filtered z scores
    list_z_filt, list_z_filt_name = find_data(os.path.join(parent_path, 'stc'), '-filtered_zscore.npy', '*')

    znorm_path = os.path.join(parent_path, 'stc', 'zfilt')
    if not os.path.exists(znorm_path):
        os.makedirs(znorm_path)

    # normalize filtered z scores
    for idx in range(len(list_z_filt)):
        z_filt = np.load(list_z_filt[idx])
        range_z = np.max(z_filt) - np.min(z_filt)
        if range_z == 0:
            z_filt_norm = z_filt
        else:
            z_filt_norm = (z_filt - np.min(z_filt)) / range_z
        np.save(os.path.join(znorm_path, f'{list_z_filt_name[idx]}-normalized.npy'), z_filt_norm)

    # find normalized filtered z scores
    list_z_filt_normalized, list_z_filt_normalized_name = find_data(znorm_path, '-normalized.npy', '*')

    # find a way to average over the entire nparray of the list of z_filt_normalized
    z_filt_sum = np.zeros((z_filt_norm.shape[0], z_filt_norm.shape[1]))
    for idx in range(len(list_z_filt_normalized)):
        load_z_filt = np.load(list_z_filt_normalized[idx])
        z_filt_sum += load_z_filt

    z_filt_avg = z_filt_sum / len(list_z_filt_normalized)

    # I want to subtract the mean row from each row of z_filt_avg
    z_filt_avg_mean_subtracted = z_filt_avg - np.mean(z_filt_avg, axis=1)[:, None]

    # print the min and max of z_filt_avg_mean_subtracted, this can help tune the colormap limits
    print(f'min: {np.min(z_filt_avg_mean_subtracted)}')
    print(f'max: {np.max(z_filt_avg_mean_subtracted)}')

    # save the averaged z scores
    np.save(os.path.join(parent_path, f'avg_filtered_zscore.npy'), z_filt_avg)

    # save the averaged z scores mean subtracted
    np.save(os.path.join(parent_path, f"avg_filtered_zscore_mean_subtracted.npy"), z_filt_avg_mean_subtracted)


    mystc = mne.SourceEstimate(z_filt_avg_mean_subtracted.T,
                               vertices=[np.arange(z_filt_avg_mean_subtracted.shape[1] / 2), np.arange(z_filt_avg_mean_subtracted.shape[1] / 2)],
                               tmin=0, tstep=1, subject='fsaverage')

    # Load fsaverage files if they don't exist
    fs_dir = mne.datasets.fetch_fsaverage(verbose=True)

    # Initial time refers to which microstate is displayed (i.e. 0 = microstate A, 6 = microstate B)
    # Loop over all microstates by changing the initial time
    for initial_time in range(0, z_filt_avg_mean_subtracted.shape[0]):

        brain = mystc.plot(subjects_dir=os.path.dirname(fs_dir),
                           initial_time= initial_time,
                           views=['lateral', 'medial'],
                           cortex='low_contrast',
                           hemi='split',
                           surface='inflated',
                           clim=dict(kind='value', lims=[-0.15, 0, 0.15]),
                           time_viewer=False,
                           background='white',
                           spacing='ico4',
                           size=(1600, 800),
                           colormap='jet',
                           smoothing_steps=15,
                           colorbar=True)

        # brain.add_annotation("HCPMMP1_combined", borders=2)
        brain.save_image(filename=os.path.join(parent_path, f'avg_filtered_zscore_M_{initial_time}.png'))


parent_dir = "/Users/chattermac/Downloads/OutputEEGSource"
# load_data_run_tess(parent_dir)
normalize_and_average_data(parent_dir)