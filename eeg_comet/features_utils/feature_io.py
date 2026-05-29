"""I/O helpers for reading/writing microstate features and time-series."""

import os.path

import numpy as np
import pandas as pd

from eeg_comet.data_utils.safe_io import safe_pd_read_pickle


class FeatureIO:
    """Export/import helpers for calculated microstate features and related time-series."""

    def __init__(self) -> None:
        """Initialize a FeatureIO instance. The class is stateless; no setup required."""
        return

    @staticmethod
    def export_features(
        features_df, feature_type, feature_mode, output_folder, export_format=".csv"
    ):
        """Export calculated features to the specified file format.

        Args:
            features_df (pandas.DataFrame): The DataFrame containing the calculated features.
            feature_type (str): The type of the features.
            feature_mode (str): The mode of the features.
            output_folder (str): The folder path to save the exported file.
            export_format (str, optional): The format of the exported file. Defaults to '.csv'.
        """
        output_path = os.path.join(
            output_folder, f"{feature_type}_{feature_mode}_features{export_format}"
        )

        if export_format == ".csv":
            features_df.to_csv(output_path, index=False)
        elif export_format == ".pkl":
            features_df.to_pickle(output_path)
        elif export_format == ".hdf":
            features_df.to_hdf(output_path, key="features", mode="w")
        elif export_format == ".json":
            features_df.to_json(output_path, orient="records", lines=True)
        else:
            raise ValueError("Invalid export format.")

    @staticmethod
    def export_rof_data(
        rof_data_dict, feature_type, feature_mode, output_folder, export_format=".csv"
    ):
        """Export ROF baseline-corrected data to separate files for detailed analysis.

        Args:
            rof_data_dict (dict): Dictionary where keys are filenames and values are ROF data dictionaries
            feature_type (str): The type of the features (e.g., 'real', 'surrogate', 'random')
            feature_mode (str): The mode of the features (e.g., 'averaged', 'sliding')
            output_folder (str): The folder path to save the exported files
            export_format (str, optional): The format of the exported file. Defaults to '.csv'
        """
        if not rof_data_dict:
            return

        # Create comprehensive ROF data structure
        all_rof_data = []

        for filename, rof_data in rof_data_dict.items():
            baseline_corrected_rof = rof_data.get("baseline_corrected_rof", None)
            time_post_tms = rof_data.get("time_post_tms", None)
            microstates = rof_data.get("microstates", [])

            if baseline_corrected_rof is not None and time_post_tms is not None:
                # Create detailed ROF data for each microstate and time point
                for i, microstate in enumerate(microstates):
                    for j, time_point in enumerate(time_post_tms):
                        rof_value = baseline_corrected_rof[i, j]
                        all_rof_data.append(
                            {
                                "Filename": filename,
                                "Microstate": microstate,
                                "Time_ms": time_point,
                                "ROF_baseline_corrected": rof_value,
                            }
                        )

        if all_rof_data:
            # Convert to DataFrame
            rof_df = pd.DataFrame(all_rof_data)

            # Export the ROF data
            output_path = os.path.join(
                output_folder, f"{feature_type}_{feature_mode}_ROF_detailed{export_format}"
            )

            if export_format == ".csv":
                rof_df.to_csv(output_path, index=False)
            elif export_format == ".pkl":
                rof_df.to_pickle(output_path)
            elif export_format == ".hdf":
                rof_df.to_hdf(output_path, key="rof_data", mode="w")
            elif export_format == ".json":
                rof_df.to_json(output_path, orient="records", lines=True)
            else:
                raise ValueError("Invalid export format.")

    @staticmethod
    def export_rof_timeseries(rof_data_dict, output_folder, export_format=".csv"):
        """Export baseline-corrected ROF time-series for *all* recordings into a single file.

        Output format (wide):
            Filename | Time_ms | ROF_A | ROF_B | ...

        Args:
            rof_data_dict (dict): key → filename, value → ROF dictionary produced by
                                  FeatureHelper.compute_relative_occurrence_frequency
            output_folder (str): directory to write the file
            export_format (str): extension (".csv", ".pkl", etc.)
        """
        if not rof_data_dict:
            return

        rows = []
        all_microstates = set()
        # First gather all unique microstates to create consistent columns
        for data in rof_data_dict.values():
            all_microstates.update(data.get("microstates", []))

        all_microstates = sorted(list(all_microstates))

        for filename, data in rof_data_dict.items():
            time_ms = np.asarray(data.get("time_ms"))
            occ_clr_bc = data.get("occurrences_clr_bc", {})

            if time_ms is None or occ_clr_bc is None:
                continue  # Skip malformed entries

            # For each timepoint build a row
            for idx, t in enumerate(time_ms):
                row = {"Filename": filename, "Time_ms": float(t)}
                for ms in all_microstates:
                    # Fill NaN if microstate not present in this file
                    if ms in occ_clr_bc:
                        row[f"ROF_{ms}"] = float(occ_clr_bc[ms][idx])
                    else:
                        row[f"ROF_{ms}"] = np.nan
                rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows)
        df.sort_values(["Filename", "Time_ms"], inplace=True)

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"ROF_timeseries{export_format}")

        if export_format == ".csv":
            df.to_csv(output_path, index=False)
        elif export_format == ".pkl":
            df.to_pickle(output_path)
        elif export_format == ".hdf":
            df.to_hdf(output_path, key="rof", mode="w")
        elif export_format == ".json":
            df.to_json(output_path, orient="records", lines=True)
        else:
            raise ValueError("Invalid export format")

    @staticmethod
    def export_rtf_data(rtf_data_dict, output_folder, export_format=".csv"):
        """Export averaged, baseline-corrected RTF matrices for all recordings.

        Output format (wide):
        Filename | RTF_A_B | RTF_A_C | ...

        Args:
            rtf_data_dict (dict): key → filename, value → RTF dictionary produced by
                FeatureHelper.compute_relative_transition_frequency
            output_folder (str): directory to write the file
            export_format (str): extension (".csv", ".pkl", etc.)
        """
        if not rtf_data_dict:
            return

        rows = []
        all_microstates = set()
        # Collect microstate list
        for data in rtf_data_dict.values():
            all_microstates.update(data.get("microstates", []))

        all_microstates = sorted(list(all_microstates))

        for filename, data in rtf_data_dict.items():
            microstates = data.get("microstates", [])
            transition_averages_bc = data.get("transition_averages_bc", {})

            if not transition_averages_bc:
                continue

            # Prefer post_tms window; fallback to first non-baseline window if absent
            if "post_tms" in transition_averages_bc:
                mat = transition_averages_bc["post_tms"]
            else:
                alt_keys = [k for k in transition_averages_bc if k != "baseline"]
                if not alt_keys:
                    continue
                mat = transition_averages_bc[alt_keys[0]]

            row = {"Filename": filename}

            for i, from_state in enumerate(microstates):
                for j, to_state in enumerate(microstates):
                    if i == j:
                        continue  # skip self-transitions
                    key = f"RTF_{from_state}_{to_state}"
                    value = float(mat[i, j]) if isinstance(mat, np.ndarray) else np.nan
                    row[key] = value

            # Ensure consistent columns across rows by filling missing microstate pairs with NaN
            for i in all_microstates:
                for j in all_microstates:
                    if i == j:
                        continue
                    key = f"RTF_{i}_{j}"
                    if key not in row:
                        row[key] = np.nan

            rows.append(row)

        if not rows:
            return

        df = pd.DataFrame(rows)
        df.sort_values("Filename", inplace=True)

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"RTF_averages{export_format}")

        if export_format == ".csv":
            df.to_csv(output_path, index=False)
        elif export_format == ".pkl":
            df.to_pickle(output_path)
        elif export_format == ".hdf":
            df.to_hdf(output_path, key="rtf", mode="w")
        elif export_format == ".json":
            df.to_json(output_path, orient="records", lines=True)
        else:
            raise ValueError("Invalid export format")

    @staticmethod
    def export_variability_data(variability_data_dict, output_folder, export_format=".csv"):
        """Export sliding window variability features (SD and RMSSD) for all recordings.

        Output format (wide):
        Filename | DUR_SD_A | DUR_SD_B | ... | DUR_RMSSD_A | DUR_RMSSD_B | ... | COV_SD_A | ...

        Args:
            variability_data_dict (dict): key → filename, value → DataFrame with variability features in long format
            output_folder (str): directory to write the file
            export_format (str): extension (".csv", ".pkl", etc.)
        """
        if not variability_data_dict:
            return

        # Collect all DataFrames (they're in long format: Filename, Feature, Value)
        all_dfs = []
        for filename, df in variability_data_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                all_dfs.append(df)

        if not all_dfs:
            return

        # Concatenate all long-format DataFrames
        combined_long_df = pd.concat(all_dfs, ignore_index=True)
        
        # Pivot to wide format: Filename as rows, Feature names as columns
        combined_df = combined_long_df.pivot_table(
            index="Filename", 
            columns="Feature", 
            values="Value"
        ).reset_index()
        
        combined_df.sort_values("Filename", inplace=True)

        os.makedirs(output_folder, exist_ok=True)
        output_path = os.path.join(output_folder, f"real_sliding_variability_features{export_format}")

        if export_format == ".csv":
            combined_df.to_csv(output_path, index=False)
        elif export_format == ".pkl":
            combined_df.to_pickle(output_path)
        elif export_format == ".hdf":
            combined_df.to_hdf(output_path, key="variability", mode="w")
        elif export_format == ".json":
            combined_df.to_json(output_path, orient="records", lines=True)
        else:
            raise ValueError("Invalid export format")

    @staticmethod
    def import_features(file_path, import_format=".csv"):
        """Import calculated features from a file.

        Args:
            file_path (str): The path to the file containing the calculated features.
            import_format (str, optional): The format of the imported file. Defaults to '.csv'.

        Returns:
            pandas.DataFrame: The imported DataFrame containing the calculated features.
        """
        if import_format == ".csv":
            features = pd.read_csv(file_path)
        elif import_format == ".pkl":
            features = safe_pd_read_pickle(file_path)
        elif import_format == ".hdf":
            features = pd.read_hdf(file_path, key="features")
        elif import_format == ".json":
            features = pd.read_json(file_path, orient="records", lines=True)
        else:
            raise ValueError("Invalid import format.")
        return features
