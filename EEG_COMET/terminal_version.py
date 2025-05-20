import os
import argparse
from comet import COMET


def main():
    parser = argparse.ArgumentParser(description='EEG-COMET Terminal Application')
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--study', type=str, help='Override study name in config')
    parser.add_argument('--input', type=str, help='Override input folder in config')
    parser.add_argument('--output', type=str, help='Override output folder in config')

    # Analysis steps to run
    parser.add_argument('--preprocess', action='store_true', help='Run preprocessing')
    parser.add_argument('--cluster', action='store_true', help='Run microstate clustering')
    parser.add_argument('--label', action='store_true', help='Run microstate labeling')
    parser.add_argument('--backfit', action='store_true', help='Run microstate backfitting')
    parser.add_argument('--features', action='store_true', help='Run feature extraction')
    parser.add_argument('--source', action='store_true', help='Run source localization')
    parser.add_argument('--correlation', action='store_true', help='Running identifying microstate sources')
    parser.add_argument('--all', action='store_true', help='Run all analysis steps in sequence')

    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file not found: {args.config}")
        return

    print(f"Loading configuration from: {args.config}")

    # Create COMET instance with the config file
    comet = COMET(config_path=args.config,
                  study_name=args.study,
                  input_folder=args.input,
                  output_folder=args.output)

    # If no specific steps are selected but not --all, show help
    if not any([args.preprocess, args.cluster, args.label, args.backfit,
                args.features, args.source, args.correlation, args.all]):
        parser.print_help()
        print("\nNo analysis steps selected. Please specify steps to run or use --all.")
        return

    # Run selected analyses
    if args.preprocess or args.all:
        print("\n--- Running Preprocessing ---")
        comet.run_preprocessing()

    if args.cluster or args.all:
        print("\n--- Running Microstate Clustering ---")
        comet.run_clustering()

    if args.label or args.all:
        print("\n--- Running Microstate Labeling ---")
        comet.run_microstate_labeling()

    if args.backfit or args.all:
        print("\n--- Running Microstate Backfitting ---")
        comet.run_backfitting()

    if args.features or args.all:
        print("\n--- Running Feature Extraction ---")
        comet.run_feature_extraction()

    if args.source or args.all:
        print("\n--- Running Source Localization ---")
        comet.run_source_localization()

    if args.correlation or args.all:
        print("\n--- Running Identifying Microstate Sources ---")
        comet.run_identifying_microstate_sources()

    print("\nAll requested analyses completed.")


if __name__ == "__main__":
    main()
