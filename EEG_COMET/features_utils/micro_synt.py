
from collections import Counter
import itertools
import random
import math
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


class MicroSynt:

    @staticmethod
    def remove_repetition_sequence(input_sequence):
        """
        Remove consecutive repetitions from a sequence.
        """
        if isinstance(input_sequence, list):
            input_sequence = "".join(input_sequence)

        no_permanence_sequence = ""
        prev_char = None

        for char in input_sequence:
            if char != prev_char:
                no_permanence_sequence += str(char)
                prev_char = char

        return no_permanence_sequence

    @staticmethod
    def generate_synthetic_sequence(input_sequence, method='random'):
        """
        Generate a synthetic sequence based on the input sequence.
        """
        if isinstance(input_sequence, list):
            input_sequence = "".join(input_sequence)

        synthetic_sequence = ''

        if method == 'surrogate':
            # Shuffle the entire sequence of characters
            shuffled_chars = list(input_sequence)
            random.shuffle(shuffled_chars)
            synthetic_sequence = ''.join(shuffled_chars)

        elif method == 'random':
            # Get unique letters from the input sequence
            unique_letters = list(set(input_sequence))

            # Generate synthetic sequence by randomly selecting from unique letters
            for _ in input_sequence:
                synthetic_sequence += random.choice(unique_letters)

        return synthetic_sequence

    @staticmethod
    def generate_theoretical_dictionary(input_sequence, word_size):
        """
        Generate a theoretical dictionary of non-repeating words.
        """
        # Count the number of unique characters in the input sequence
        unique_characters = sorted(set(input_sequence))

        # Generate all possible non-repeating words of the given size
        possible_words = itertools.product(unique_characters, repeat=word_size)

        # Filter out combinations with consecutive same letters
        theoretical_dictionary = []
        for word in possible_words:
            if not any(word[i] == word[i+1] for i in range(len(word) - 1)):
                theoretical_dictionary.append(''.join(word))

        return theoretical_dictionary

    @staticmethod
    def generate_real_dictionary(input_sequence, word_size):
        """
        Generate a real dictionary from an input sequence.
        """
        # Check if the input_sequence is a list, if so, join it into a string
        if isinstance(input_sequence, list):
            input_sequence = ''.join(input_sequence)

        # Initialize an empty dictionary to store the combinations and their counts
        real_dictionary, sequence_representation = {}, {}

        # Iterate through the input sequence
        for i in range(len(input_sequence) - word_size + 1):
            # Extract consecutive substring of given length
            combination = input_sequence[i:i+word_size]
            # Add combination to the dictionary and update its count
            if combination not in real_dictionary:
                real_dictionary[combination] = 1
            else:
                real_dictionary[combination] += 1

        # Generate sequence representation
        for word, count in real_dictionary.items():
            sequence_representation[word] = count

        # Sort the sequence representation dictionary
        sequence_representation = dict(sorted(sequence_representation.items()))

        return sorted(real_dictionary), sequence_representation

    @staticmethod
    def calculate_entropy(input_sequence):
        """
        Calculate the entropy of a sequence.
        """
        # Count the frequency of each microstate in the input sequence
        microstate_counts = Counter(input_sequence)

        # Total number of microstates
        total_microstates = len(input_sequence)

        # Initialize entropy
        entropy = 0

        # Calculate entropy
        for microstate, count in microstate_counts.items():
            probability = count / total_microstates
            entropy -= probability * math.log(probability)

        return entropy

    def sequence_analysis(self, input_sequence, word_size):
        """
        Analyze the entropy distribution in an input sequence.
        """
        # Check if the input_sequence is a list, if so, join it into a string
        if isinstance(input_sequence, list):
            input_sequence = ''.join(input_sequence)

        theoretical_dictionary = self.generate_theoretical_dictionary(input_sequence, word_size)
        real_dictionary, sequence_representation_real = self.generate_real_dictionary(input_sequence, word_size)

        # Compute entropy for each word in real_dictionary
        word_entropy = {}
        for word in real_dictionary:
            entropy = self.calculate_entropy(word)
            word_entropy[word] = entropy

        # Get unique entropy values and sort them (lowest first)
        unique_entropies_sorted = sorted(set(word_entropy.values()))

        # Convert sequence_representation to entropy classes
        entropy_representation_real = {f"EntropyClass{i + 1}": 0 for i in range(len(unique_entropies_sorted))}

        for word, count in sequence_representation_real.items():
            entropy = word_entropy[word]
            entropy_class = f"EntropyClass{unique_entropies_sorted.index(entropy) + 1}"
            entropy_representation_real[entropy_class] += count

        # Calculate percentages for each entropy class
        total_words_real = sum(entropy_representation_real.values())
        entropy_representation_percentage_real = {ec: count / total_words_real for ec, count in
                                                  entropy_representation_real.items()}

        # Convert sequence_representation for theoretical dictionary to entropy classes
        entropy_representation_theoretical = {f"EntropyClass{i + 1}": 0 for i in range(len(unique_entropies_sorted))}

        for word in theoretical_dictionary:
            entropy = self.calculate_entropy(word)
            entropy_class = f"EntropyClass{unique_entropies_sorted.index(entropy) + 1}"
            entropy_representation_theoretical[entropy_class] += 1

        # Calculate percentages for each entropy class
        total_words_theoretical = sum(entropy_representation_theoretical.values())
        entropy_representation_percentage_theoretical = {ec: count / total_words_theoretical for ec, count in
                                                         entropy_representation_theoretical.items()}

        return entropy_representation_percentage_real, entropy_representation_percentage_theoretical

    @staticmethod
    def calculate_representation_ratios(*entropy_dicts):
        """
        Calculate representation ratios between entropy distributions.
        """
        representation_ratios = {}
        first_entropy_dict = entropy_dicts[0]

        for class_name, entropy_value in first_entropy_dict.items():
            ratio_key = f'RepresentationRatio_{class_name}'
            second_entropy_value = entropy_dicts[1].get(class_name)
            if second_entropy_value is not None:
                representation_ratios[ratio_key] = entropy_value / second_entropy_value

        # Sort representation_ratios based on ratio_key
        sorted_representation_ratios = dict(sorted(representation_ratios.items(), key=lambda item: item[0]))

        return sorted_representation_ratios

    def surrogate_statistics(self, input_sequence, word_size, repeats=1000):
        """
        Generate surrogate statistics based on input sequences.
        """
        distributions = {}

        for _ in range(repeats):
            # Generate surrogate sequence
            surrogate_sequence = self.generate_synthetic_sequence(input_sequence, 'surrogate')
            surrogate_no_permanence_sequence = self.remove_repetition_sequence(surrogate_sequence)

            # Generate entropy representations
            surrogate_entropy_representation, surrogate_entropy_representation_theoretical = self.sequence_analysis(
                surrogate_no_permanence_sequence, word_size
            )

            # Calculate representation ratios
            surrogate_entropy_representation_ratio = self.calculate_representation_ratios(
                surrogate_entropy_representation, surrogate_entropy_representation_theoretical
            )

            # Append the ratio to the distribution for each class separately
            for key, value in surrogate_entropy_representation_ratio.items():
                # Rename the key
                new_key = key.replace('RepresentationRatio', 'Distribution')
                if new_key not in distributions:
                    distributions[new_key] = []
                distributions[new_key].append(value)

        return distributions

    @staticmethod
    def plot_violin_plots(distribution):
        """
        Plot violin plots for the distribution of entropy classes.
        """
        # Convert the distribution dictionary to a DataFrame
        df = pd.DataFrame.from_dict(distribution)

        # Melt the DataFrame to have all values in one column
        melted_df = df.melt(var_name='Entropy Class', value_name='Value')

        # Plot violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x='Entropy Class', y='Value', data=melted_df, inner=None, linewidth=0.8)
        sns.swarmplot(x='Entropy Class', y='Value', data=melted_df, color='k', alpha=0.5)  # Add swarmplot for individual points
        plt.title('Violin Plot for Distribution of Entropy Classes')
        plt.xlabel('Entropy Class')
        plt.ylabel('Entropy Representation Ratio')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def plot_representation_ratios(representation_ratios):
        """
        Plot representation ratios of different entropy classes in one figure as dots connected to each other.
        """
        # Extract entropy classes and ratios
        entropy_classes = list(representation_ratios.keys())
        ratios = list(representation_ratios.values())

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(entropy_classes, ratios, marker='o', linestyle='-')
        plt.title('Representation Ratios of Entropy Classes')
        plt.xlabel('Entropy Class')
        plt.ylabel('Representation Ratio')
        plt.xticks(rotation=45)
        plt.grid(True)
        plt.tight_layout()
        plt.show()


import numpy as np
import os.path
import fnmatch

main_folder = "C:/Users/amin_/Documents/Marian_stuff/Microstates_DEC2023/microstate_output/ANALYSIS1_HEALTHY_MDD/ANALYSIS1_HEALTHY_MDD_segmentation"
healthy_pattern = ["musc", "yd-c"]
mdd_pattern = ["yct", "yd-p"]

healthy_files = []
mdd_files = []

for root, dirs, files in os.walk(main_folder):
    for file in files:
        if file.endswith('.csv'):
            file_path = os.path.join(root, file)
            for pattern in healthy_pattern:
                if fnmatch.fnmatch(file, f'*{pattern}*'):
                    healthy_files.append(file_path)
                    break
            else:
                for pattern in mdd_pattern:
                    if fnmatch.fnmatch(file, f'*{pattern}*'):
                        mdd_files.append(file_path)
                        break


def get_segmentation_values(file_path):
    try:
        df = pd.read_csv(file_path)
        segmentation_values = df['segmentation'].tolist()
        return segmentation_values
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []

micro_synt = MicroSynt()
word_size = 3

def get_segmentation_values(file_path):
    try:
        df = pd.read_csv(file_path)
        segmentation_values = df['segmentation'].tolist()
        return segmentation_values
    except Exception as e:
        print(f"Error reading file {file_path}: {str(e)}")
        return []

def process_files(files):
    representation_ratios = []
    for file_path in files:
        segmentation_values = get_segmentation_values(file_path)
        input_sequence = segmentation_values
        input_no_permanence_sequence = micro_synt.remove_repetition_sequence(input_sequence)
        entropy_representation_real, entropy_representation_theoretical = micro_synt.sequence_analysis(
            input_no_permanence_sequence, word_size)
        entropy_representation_ratio = micro_synt.calculate_representation_ratios(
            entropy_representation_real, entropy_representation_theoretical)
        representation_ratios.append(entropy_representation_ratio['RepresentationRatio_EntropyClass1'])
    return input_sequence, representation_ratios

# Assuming you have the necessary functions imported from micro_synt module

# Process healthy files
input_sequence, healthy_representation_ratios = process_files(healthy_files)

# Process MDD files
input_sequence2, mdd_representation_ratios = process_files(mdd_files)

print("Healthy Representation Ratios (RepresentationRatio_EntropyClass1):")
print(np.mean(healthy_representation_ratios))

print("\nMDD Representation Ratios (RepresentationRatio_EntropyClass1):")
print(np.mean(mdd_representation_ratios))


from scipy.stats import ttest_ind

# Perform t-test
t_statistic, p_value = ttest_ind(healthy_representation_ratios, mdd_representation_ratios)
print("T-Statistic:", t_statistic)
print("P-Value:", p_value)


#######
# input_sequence = ['AAAAACCCCCCCCCCCCCBBBBBBBBBAAAAACCCCCCBBBBBBBBBBBBBBBB']
# input_sequence = ['A', 'A', 'A', 'A', 'A', 'A', 'A', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B', 'B']
# input_sequence = 'AAAAABBBBBAAABABACCCCCCCCCCCABABBBBBBACCCBAAAAAAABACABBBBBBBBBABACDDDDDDDDDDDDDDDDDAAAAAAAADDDDD'
word_size = 5

input_no_permanence_sequence = micro_synt.remove_repetition_sequence(input_sequence)
print(micro_synt.remove_repetition_sequence(input_sequence))
print(f"Surrogate: {micro_synt.generate_synthetic_sequence(input_sequence, 'surrogate')}")
print(f"Random: {micro_synt.generate_synthetic_sequence(input_sequence, 'random')}")

theoretical_dictionary = micro_synt.generate_theoretical_dictionary(input_no_permanence_sequence, word_size)
print(f"Theoretical Dictionary: {theoretical_dictionary}")

real_dictionary, sequence_representation_real = micro_synt.generate_real_dictionary(input_no_permanence_sequence, word_size)
print(f"Real Dictionary: {real_dictionary}")
print(f"Sequence Representation Real Dictionary: {sequence_representation_real}")

entropy_representation_real, entropy_representation_theoretical = micro_synt.sequence_analysis(input_no_permanence_sequence, word_size)
print(f"Sequence Analysis: {micro_synt.sequence_analysis(input_no_permanence_sequence, word_size)}")

representation_ratios = micro_synt.calculate_representation_ratios(entropy_representation_real, entropy_representation_theoretical)
print(f"Representation Ratios: {representation_ratios}")

distribution = micro_synt.surrogate_statistics(input_sequence, word_size)

# Find the maximum length among all lists
max_length = max(len(lst) for lst in distribution.values())

# Pad shorter lists with NaN values
for key in distribution:
    distribution[key] += [np.nan] * (max_length - len(distribution[key]))

# Convert the distribution to a numpy array for easy statistical analysis
distribution_array = np.array([distribution[key] for key in sorted(distribution.keys())])

# Calculate statistics for each entropy class
mean = np.nanmean(distribution_array, axis=1)
std_dev = np.nanstd(distribution_array, axis=1)
min_val = np.nanmin(distribution_array, axis=1)
max_val = np.nanmax(distribution_array, axis=1)

# Print statistics for each entropy class
for i, key in enumerate(sorted(distribution.keys())):
    print(f"Entropy Class {i+1}:")
    print("Mean:", mean[i])
    print("Standard Deviation:", std_dev[i])
    print("Minimum:", min_val[i])
    print("Maximum:", max_val[i])


# micro_synt.plot_violin_plots(distribution)

# Plot representation ratios
input_no_permanence_sequence = micro_synt.remove_repetition_sequence(input_sequence)
entropy_representation_real, entropy_representation_theoretical = micro_synt.sequence_analysis(
    input_no_permanence_sequence, word_size)
representation_ratios = micro_synt.calculate_representation_ratios(
    entropy_representation_real, entropy_representation_theoretical)
micro_synt.plot_representation_ratios(representation_ratios)

input_no_permanence_sequence = micro_synt.remove_repetition_sequence(input_sequence2)
entropy_representation_real, entropy_representation_theoretical = micro_synt.sequence_analysis(
    input_no_permanence_sequence, word_size)
representation_ratios = micro_synt.calculate_representation_ratios(
    entropy_representation_real, entropy_representation_theoretical)
micro_synt.plot_representation_ratios(representation_ratios)
