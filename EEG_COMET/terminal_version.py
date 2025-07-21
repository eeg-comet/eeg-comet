#!/usr/bin/env python3
"""
EEG-COMET Terminal Version
Enhanced with fixed automatic k selection and consistent logging
"""

import os
import sys
import argparse
import warnings
from pathlib import Path

# Add the parent directory to the path to allow imports
sys.path.append(str(Path(__file__).parent))

from comet import COMET

# Silence TensorFlow warnings before any imports that might use it
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Hide INFO and WARNING messages
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
warnings.filterwarnings('ignore', category=UserWarning, module='.*tensorflow.*')


def display_welcome_message():
    """Display the welcome message for EEG-COMET terminal version"""
    print("=" * 80)
    print("🧠 EEG-COMET: Comprehensive Microstate Extraction Toolbox")
    print("=" * 80)
    print("Authors: Amin Kabir, Raaj Chatterjee, Faranak Farzan")
    print("Organization: SFU eBrain Lab (www.ebrainlab.ca)")
    print("GitHub: https://github.com/eBrainLab/EEG-Microstate-Feature-Extraction")
    print("=" * 80)
    print("🖥️  Terminal Version - Enhanced with Automatic K Selection")
    print("=" * 80)


def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description='EEG-COMET Terminal Application - Enhanced Version',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with automatic k selection
  python terminal_version.py --config my_config.ini --all --auto-k --kmin 2 --kmax 8

  # Run only preprocessing and clustering
  python terminal_version.py --config my_config.ini --preprocess --cluster

  # Use specific k value for clustering
  python terminal_version.py --config my_config.ini --cluster --k 4

  # Override study settings
  python terminal_version.py --config my_config.ini --study new_study --input /data --output /results --all
        """
    )

    # Required arguments
    parser.add_argument('--config', type=str, required=True,
                        help='Path to the configuration file')

    # Study override arguments
    parser.add_argument('--study', type=str,
                        help='Override study name in config')
    parser.add_argument('--input', type=str,
                        help='Override input folder in config')
    parser.add_argument('--output', type=str,
                        help='Override output folder in config')

    # Analysis steps
    parser.add_argument('--preprocess', action='store_true',
                        help='Run preprocessing')
    parser.add_argument('--cluster', action='store_true',
                        help='Run microstate clustering')
    parser.add_argument('--label', action='store_true',
                        help='Run microstate labeling')
    parser.add_argument('--backfit', action='store_true',
                        help='Run microstate backfitting')
    parser.add_argument('--features', action='store_true',
                        help='Run feature extraction')
    parser.add_argument('--source', action='store_true',
                        help='Run source localization')
    parser.add_argument('--correlation', action='store_true',
                        help='Run microstate source correlation')
    parser.add_argument('--all', action='store_true',
                        help='Run all analysis steps in sequence')

    # Clustering-specific arguments
    parser.add_argument('--auto-k', action='store_true',
                        help='Use automatic k selection for clustering')
    parser.add_argument('--k', type=int,
                        help='Specific number of clusters (overrides config)')
    parser.add_argument('--kmin', type=int, default=2,
                        help='Minimum k for automatic selection (default: 2)')
    parser.add_argument('--kmax', type=int, default=10,
                        help='Maximum k for automatic selection (default: 10)')
    parser.add_argument('--method', type=str,
                        choices=['kmeans', 'similarity', 'taahc'],
                        help='Clustering method (kmeans, similarity, taahc)')
    parser.add_argument('--repeats', type=int,
                        help='Number of clustering repetitions')

    # Verbosity options
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Enable verbose output')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress non-essential output')

    return parser


def validate_arguments(args):
    """Validate command line arguments"""
    errors = []

    # Check if config file exists
    if not os.path.exists(args.config):
        errors.append(f"Config file not found: {args.config}")

    # Check k selection arguments
    if args.auto_k and args.k:
        errors.append("Cannot specify both --auto-k and --k")

    if args.kmin >= args.kmax:
        errors.append("kmin must be less than kmax")

    # Check if at least one analysis step is selected
    analysis_steps = [args.preprocess, args.cluster, args.label, args.backfit,
                      args.features, args.source, args.correlation, args.all]
    if not any(analysis_steps):
        errors.append("No analysis steps selected. Use --all or specify individual steps.")

    # Check input/output directories if specified
    if args.input and not os.path.exists(args.input):
        errors.append(f"Input directory not found: {args.input}")

    if args.output and not os.path.exists(os.path.dirname(args.output)):
        parent_dir = os.path.dirname(args.output)
        if parent_dir and not os.path.exists(parent_dir):
            errors.append(f"Output parent directory not found: {parent_dir}")

    return errors


def configure_comet_for_clustering(comet, args):
    """Configure COMET instance for clustering based on arguments"""

    # Set clustering method
    if args.method:
        method_map = {
            'kmeans': 'Modified K-Means Clustering (Pascual-Marqui et al. 1995)',
            'similarity': 'Modified K-Means Clustering with Spatial Similarity',
            'taahc': 'Topographic Atomize and Agglomerate Hierarchical Clustering'
        }
        comet.clustering_method = method_map[args.method]

    # Set k selection parameters
    if args.auto_k:
        comet.number_of_maps = 'auto'
        comet.choose_number_of_maps = "auto"
        comet.kmin = args.kmin
        comet.kmax = args.kmax
        comet.stopping_mode = 'majority_vote'  # Use robust majority vote
        print(f"🔍 Automatic k selection enabled: k ∈ [{args.kmin}, {args.kmax}]")
        print(f"🎯 Using majority vote across all optimization methods")
        print(f"⚡ Using single repeat (n_inits=1) for optimization")
        print(f"📊 Using GFP peaks for optimization (ignoring use_percentages setting)")
    elif args.k:
        comet.number_of_maps = args.k
        comet.choose_number_of_maps = "user"
        print(f"📌 Using specified k value: {args.k}")

    # Set number of repetitions
    if args.repeats:
        comet.number_of_repeats = args.repeats
        print(f"🔄 Clustering repetitions: {args.repeats}")


def run_analysis_pipeline(comet, args):
    """Run the complete analysis pipeline"""

    # Determine which steps to run
    steps_to_run = []

    if args.all:
        steps_to_run = ['preprocess', 'cluster', 'label', 'backfit', 'features', 'source', 'correlation']
    else:
        if args.preprocess:
            steps_to_run.append('preprocess')
        if args.cluster:
            steps_to_run.append('cluster')
        if args.label:
            steps_to_run.append('label')
        if args.backfit:
            steps_to_run.append('backfit')
        if args.features:
            steps_to_run.append('features')
        if args.source:
            steps_to_run.append('source')
        if args.correlation:
            steps_to_run.append('correlation')

    print(f"\n📋 Analysis pipeline: {' → '.join(steps_to_run)}")
    print("=" * 60)

    # Run each step
    for step in steps_to_run:
        try:
            if step == 'preprocess':
                if not comet.done_preprocessing:
                    print("\n🔧 PREPROCESSING")
                    print("-" * 40)
                    comet.run_preprocessing()
                    print("[PREPROCESSING] Preprocessing completed successfully")
                else:
                    print("\n✅ Preprocessing already completed")

            elif step == 'cluster':
                if not comet.done_clustering:
                    print("\n🎯 MICROSTATE CLUSTERING")
                    print("-" * 40)

                    # Configure clustering parameters
                    configure_comet_for_clustering(comet, args)

                    # Run clustering
                    comet.run_clustering()

                    # Show optimization results if automatic k was used
                    if (comet.choose_number_of_maps == "auto" and
                            hasattr(comet, 'optimization_results') and
                            comet.optimization_results):
                        show_optimization_results(comet.optimization_results)

                    print("[CLUSTERING] Clustering completed successfully")
                else:
                    print("\n✅ Clustering already completed")

            elif step == 'label':
                if not comet.done_microstate_labeling:
                    print("\n🏷️  MICROSTATE LABELING")
                    print("-" * 40)
                    comet.run_microstate_labeling()
                    print("[LABELING] Microstate labeling completed successfully")
                else:
                    print("\n✅ Microstate labeling already completed")

            elif step == 'backfit':
                if not comet.done_backfitting:
                    print("\n📐 BACKFITTING")
                    print("-" * 40)
                    comet.run_backfitting()
                    print("[BACKFITTING] Backfitting completed successfully")
                else:
                    print("\n✅ Backfitting already completed")

            elif step == 'features':
                if not comet.done_extracting_features:
                    print("\n📊 FEATURE EXTRACTION")
                    print("-" * 40)
                    comet.extract_features()
                    print("[FEATURE EXTRACTION] Feature extraction completed successfully")
                else:
                    print("\n✅ Feature extraction already completed")

            elif step == 'source':
                if not comet.done_source_localization:
                    print("\n🧠 SOURCE LOCALIZATION")
                    print("-" * 40)
                    comet.source_localize_microstates()
                    print("[SOURCE LOCALIZATION] Source localization completed successfully")
                else:
                    print("\n✅ Source localization already completed")

            elif step == 'correlation':
                if not comet.done_identifying_microstate_sources:
                    print("\n🔗 SOURCE-MICROSTATE CORRELATION")
                    print("-" * 40)
                    comet.source_microstates_correlation()
                    print("[SOURCE LOCALIZATION] Source correlation completed successfully")
                else:
                    print("\n✅ Source correlation already completed")

        except Exception as e:
            print(f"❌ Error in {step}: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return False

    return True


def show_optimization_results(optimization_results):
    """Display optimization results in a formatted way"""
    print("\n" + "=" * 60)
    print("🔍 AUTOMATIC K SELECTION RESULTS")
    print("=" * 60)

    # Show individual method results
    for method_name, result in optimization_results.items():
        if method_name != 'majority_vote':
            print(f"📊 {result.method_name}: k = {result.optimal_k}")

    # Show majority vote result
    if 'majority_vote' in optimization_results:
        majority_result = optimization_results['majority_vote']
        print(f"\n🎯 FINAL DECISION (Majority Vote): k = {majority_result.optimal_k}")

        # Show vote breakdown
        from collections import Counter
        vote_counts = Counter(majority_result.scores)
        print("\n📈 Vote breakdown:")
        for k_value, count in sorted(vote_counts.items()):
            print(f"   k={k_value}: {count} votes")

    print("=" * 60)


def main():
    """Main function for terminal version"""
    # Display welcome message
    display_welcome_message()

    # Setup argument parser
    parser = setup_argument_parser()
    args = parser.parse_args()

    # Validate arguments
    validation_errors = validate_arguments(args)
    if validation_errors:
        print("❌ Validation errors:")
        for error in validation_errors:
            print(f"   • {error}")
        parser.print_help()
        return 1

    # Setup verbosity
    if args.quiet:
        import warnings
        warnings.filterwarnings('ignore')

    try:
        # Initialize COMET instance
        print(f"📂 Loading configuration from: {args.config}")

        comet = COMET(
            config_path=args.config,
            study_name=args.study,
            input_folder=args.input,
            output_folder=args.output
        )

        print(f"📋 Study: {comet.study_name}")
        print(f"📁 Input: {comet.input_folder}")
        print(f"💾 Output: {comet.output_folder}")

        # Run analysis pipeline
        success = run_analysis_pipeline(comet, args)

        if success:
            print("\n🎉 ANALYSIS COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📊 Results saved to: {comet.output_folder}")
            print(f"📝 Logs saved to: {comet.log_file_path}")
            return 0
        else:
            print("\n❌ ANALYSIS FAILED!")
            return 1

    except KeyboardInterrupt:
        print("\n⚠️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
