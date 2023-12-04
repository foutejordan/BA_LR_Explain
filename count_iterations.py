from itertools import combinations

N_MAX = 4

def generate_all_modifications(binary_vector, n):
    """
    Generate all binary vectors with exactly n bits flipped from the input binary vector.
    """
    length = len(binary_vector)
    
    # Generate all combinations of indices to flip
    indices_to_flip_combinations = combinations(range(length), n)

    # Initialize a list to store all modified vectors
    modified_vectors = []

    # Iterate through each combination of indices
    for indices_to_flip in indices_to_flip_combinations:
        # Create a copy of the original binary vector
        modified_vector = binary_vector.copy()

        # Flip the bits at the specified indices
        for index in indices_to_flip:
            modified_vector[index] = 1 - modified_vector[index]

        # Append the modified vector to the list
        modified_vectors.append(modified_vector)

    return modified_vectors

vector = [0]*206

with open("possible-nb-modifications.txt", "w+") as file:
    for i in range(1, N_MAX+1):
        x = generate_all_modifications(vector, i)
        file.write(f"{i}: {len(x)}\n")
        