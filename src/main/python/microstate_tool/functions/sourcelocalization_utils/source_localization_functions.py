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

from functions.data_utils.data_io import DataIO
#from functions.sourcelocalization_utils.stc_io import stc_read, stc_write

import mne
# mne.viz.set_3d_backend("pyvista")

# from mayavi import mlab
#mlab.init_notebook()


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
def tess_algorithm(rawfilename, eeg_data, microstate_maps, stc_data,
                   nperm, tess_path):
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
    
    z_scores_path = os.path.join(tess_path, "z_scores")
    p_values_path = os.path.join(tess_path, "p_values")
    if not os.path.exists(z_scores_path):
        os.makedirs(z_scores_path)
    if not os.path.exists(p_values_path):
        os.makedirs(p_values_path)
    
    t_coeff = first_regression(eeg_data, microstate_maps)
    beta_coeff = second_regression(t_coeff, stc_data)

    # Permutation of beta over t to determine significance
    z_scores = np.zeros(beta_coeff.shape)
    beta_dist = np.zeros((beta_coeff.shape[0], beta_coeff.shape[1], nperm))
    bonferroni = beta_coeff.shape[1]
    t_shuffle = t_coeff
    for ii in range(0, nperm):
        np.random.shuffle(t_shuffle)
        beta_dist[:, :, ii] = second_regression(t_shuffle, stc_data)
        if (ii % 50) == 0:
            print(ii)
    for idx, x in np.ndenumerate(beta_coeff):
        z_scores[idx] = stats.zscore(np.insert(beta_dist[idx[0]][idx[1]][:], 0, x))[0]
    p_values = bonferroni * stats.norm.sf(abs(z_scores))

    # Save results for each file
    np.save(os.path.join(z_scores_path, rawfilename + "_z_scores"), z_scores)
    np.save(os.path.join(p_values_path, rawfilename + "_p_values"), p_values)

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

    data_io = DataIO()
    list_eeg_path, list_eeg_names = data_io.find_data(preprocessed_data_path, extension, '*')

    counter = 0
    for eeg_path in list_eeg_path:
        counter += 1
        rawfilename = os.path.split(eeg_path)[1].split('.')[0]
        print("Source Localizing Microstates", rawfilename)
        print("\n", 100 * counter / len(list_eeg_path))

        data_io = DataIO()
        # Load the EEG data
        eeg = data_io.load_eegs(eeg_path, extension, data_type)
        eeg_data = data_io.get_eeg_data(eeg, data_type)
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

