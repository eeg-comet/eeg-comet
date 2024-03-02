
import math
import random
import itertools
from collections import Counter


class FeatureHelper:

    @staticmethod
    def initialize_empty_window_data(input_sequence):
        """
        Create a dictionary with zero values for all elements in the input_sequence.
        """
        return {element: 0 for element in set(input_sequence)}

    @staticmethod
    def initialize_dynamic_windows(input_sequence, sampling_rate, window_size):
        """
        Divide the input sequence into fixed-size windows and initialize a list to store feature values for each window.
        """
        # Calculate the number of samples in each window
        window_size_samples = int(sampling_rate * window_size)

        # Calculate the total number of windows
        num_windows = len(input_sequence) // window_size_samples

        # Initialize a list with zeros for each window
        windows = [0.0] * num_windows
        return windows, window_size_samples

    @staticmethod
    def remove_repetition_sequence(input_sequence):
        """
        Remove consecutive repetitions from a sequence.
        """
        # If the input sequence is a list, join it into a string
        if isinstance(input_sequence, list):
            input_sequence = "".join(input_sequence)

        # Initialize an empty string to store the sequence with consecutive repetitions removed
        no_permanence_sequence = ""

        # Initialize a variable to store the previous character
        prev_char = None

        # Iterate through each character in the input sequence
        for char in input_sequence:
            # If the current character is different from the previous character, add it to the result string
            if char != prev_char:
                no_permanence_sequence += str(char)
                prev_char = char

        return no_permanence_sequence

    @staticmethod
    def compute_lempel_ziv_complexity(input_no_permanence_sequence):
        """
        Calculate Lempel-Ziv complexity for a single window using the LZ76 algorithm.
        """
        # Initialize variables
        n = len(input_no_permanence_sequence)
        i, k, l, c, k_max = 0, 1, 1, 1, 1

        # Iterate until a break condition is met
        while True:
            # Check if the current characters at indices i+k and l+k are different
            if input_no_permanence_sequence[i + k] != input_no_permanence_sequence[l + k]:
                # Update k_max if necessary
                if k > k_max:
                    k_max = k
                # Move to the next position i
                i += 1
                # Check if i reaches l (a new block)
                if i == l:
                    # Increment the complexity
                    c += 1
                    # Move l to the end of the current block
                    l += k_max
                    # Check if the end of the sequence is reached
                    if l + 1 > n - 1:
                        break
                    else:
                        # Reset indices and k_max for the next block
                        i = 0
                        k = 1
                        k_max = 1
                else:
                    # Reset k for the next comparison
                    k = 1
            else:
                # Increment k if the characters are the same
                k += 1
                # Check if the end of the sequence is reached
                if l + k > n - 1:
                    # Increment the complexity and exit the loop
                    c += 1
                    break

        # Calculate b (maximum complexity)
        b = 1.0 * n / math.log(n, 2)
        # Return the normalized complexity
        return c / b

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
            if not any(word[i] == word[i + 1] for i in range(len(word) - 1)):
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
            combination = input_sequence[i:i + word_size]
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
        Calculate the Shannon entropy of a sequence.
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
