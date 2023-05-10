import random
import string
import collections

# Define the file to decrypt
ENCRYPTED_FILE = 'enc.txt'

# Define the size of the population
POPULATION_SIZE = 50

# Define the number of generations
NUM_GENERATIONS = 1000

# Define the mutation rate (probability of a gene mutating)
MUTATION_RATE = 0.1

# Define the length of the chromosomes
CHROMOSOME_LENGTH = 26

# Define the range of possible gene values
GENE_RANGE = string.ascii_uppercase


# Load the encrypted file
def load_encrypted_file(filename):
    with open(filename, 'r') as f:
        return f.read().upper()


# Define the fitness function
def fitness(decryption_key, encrypted_text, dictionary_file, letter_freq_file):
    # Decrypt the text using the decryption key
    decrypted_text = encrypted_text.translate(str.maketrans(decryption_key, GENE_RANGE))

    # Load the letter frequencies
    letter_freq = {}
    with open(letter_freq_file) as f:
        for line in f:
            line = line.strip()
            if line:
                prob, letter = line.split()
                letter_freq[letter] = float(prob)

    # Calculate the score based on letter frequency and dictionary lookup
    score = 0
    num_valid_words = 0
    with open(dictionary_file) as f:
        valid_words = set(line.strip() for line in f)
        for word in decrypted_text.split():
            if word.lower() in valid_words:
                score += 1
                num_valid_words += 1
            for letter in word:
                score += letter_freq.get(letter.lower(), 0)

    # Normalize the score by the length of the decrypted text
    fitness_score = score / len(decrypted_text)

    # Penalize solutions with fewer valid words
    if num_valid_words < len(decrypted_text.split()) / 2:
        fitness_score *= 0.5

    return fitness_score

# Generate the initial population
def generate_population(size):
    population = []
    for i in range(size):
        chromosome = ''.join(random.sample(GENE_RANGE, CHROMOSOME_LENGTH))
        population.append(chromosome)
    return population


# Select parents from the population using roulette wheel selection
def select_parents(population, fitness_scores):
    total_fitness = sum(fitness_scores)
    selection_probs = [score / total_fitness for score in fitness_scores]
    parent1 = random.choices(population, weights=selection_probs)[0]
    parent2 = random.choices(population, weights=selection_probs)[0]
    return parent1, parent2


# Crossover the selected parents to create offspring using one-point crossover
def crossover(parent1, parent2):
    crossover_point = random.randint(0, CHROMOSOME_LENGTH - 1)
    offspring1 = parent1[:crossover_point] + parent2[crossover_point:]
    offspring2 = parent2[:crossover_point] + parent1[crossover_point:]
    return offspring1, offspring2


# Mutate the offspring by randomly swapping two genes
def mutate(offspring):
    if random.random() < MUTATION_RATE:
        mutation_point1, mutation_point2 = random.sample(range(CHROMOSOME_LENGTH), 2)
        offspring_list = list(offspring)
        offspring_list[mutation_point2] = offspring[mutation_point1]
        offspring_list[mutation_point1] = offspring[mutation_point2]
        offspring = ''.join(offspring_list)
    return offspring


# Run the genetic algorithm
def genetic_algorithm(dictionary_file):
    # Load the encrypted file
    encrypted_text = load_encrypted_file(ENCRYPTED_FILE)

    # Generate the initial population
    population = generate_population(POPULATION_SIZE)

    # Iterate over the generations
    for generation in range(NUM_GENERATIONS):
        # Evaluate the fitness of each chromosome in the population
        fitness_scores = [fitness(chromosome, encrypted_text, dictionary_file) for chromosome in population]

        # Select the parents for crossover
        parent1, parent2 = select_parents(population, fitness_scores)

        # Create offspring via crossover
        offspring1, offspring2 = crossover(parent1, parent2)

        # Mutate the offspring
        offspring1 = mutate(offspring1)
        offspring2 = mutate(offspring2)

        # Replace the least fit members of the population with the offspring
        fitness_scores = [fitness(chromosome, encrypted_text, dictionary_file) for chromosome in population]
        min_fitness_index = fitness_scores.index(min(fitness_scores))
        population[min_fitness_index] = offspring1
        min_fitness_index = fitness_scores.index(min(fitness_scores))
        population[min_fitness_index] = offspring2

    # Return the best decryption key
    fitness_scores = [fitness(chromosome, encrypted_text, dictionary_file) for chromosome in population]
    best_index = fitness_scores.index(max(fitness_scores))
    best_decryption_key = population[best_index]
    return best_decryption_key


# Decrypt the file using the best decryption key found by the genetic algorithm
def decrypt_file(decryption_key):
    encrypted_text = load_encrypted_file(ENCRYPTED_FILE)
    decrypted_text = encrypted_text.translate(str.maketrans(decryption_key, GENE_RANGE))
    print(decrypted_text)


# Run the genetic algorithm and decrypt the file
dictionary_file = 'dict.txt'
letter_freq_file = 'Letter_freq.txt'
best_decryption_key = genetic_algorithm(dictionary_file, letter_freq_file)
decrypt_file(best_decryption_key)
