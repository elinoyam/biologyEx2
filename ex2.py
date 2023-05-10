import random
import json

GENE_RANGE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
NUM_TOP_PARENTS = 5
NUM_CHILDREN = 2
ENCRYPTED_FILE = 'encrypted.txt'


def load_dictionary(file):
    """Load dictionary file."""
    with open(file) as f:
        words = set(f.read().split())
    return words


def load_letter_freq(file):
    """Load letter frequency file."""
    with open(file) as f:
        letter_prob = {}
        for line in f:
            prob, letter = line.strip().split()
            letter_prob[letter] = float(prob)
    return letter_prob


def generate_initial_population(population_size, key_length):
    """Generate an initial population of decryption keys."""
    population = []
    for i in range(population_size):
        key = list(GENE_RANGE)
        random.shuffle(key)
        population.append(''.join(key))
    return population


def select_parents(population, encrypted_text, dictionary_file, letter_freq_file):
    """Select two parents from the population using fitness-proportionate selection."""
    fitnesses = [fitness(key, encrypted_text, dictionary_file, letter_freq_file) for key in population]
    total_fitness = sum(fitnesses)
    parent1 = random.choices(population, weights=[f/total_fitness for f in fitnesses])[0]
    parent2 = random.choices(population, weights=[f/total_fitness for f in fitnesses])[0]
    return parent1, parent2


def crossover(parent1, parent2):
    """Perform crossover to produce two children."""
    crossover_point = random.randint(1, len(parent1)-1)
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


def mutate(key, mutation_rate):
    """Mutate the decryption key."""
    if random.random() < mutation_rate:
        index1, index2 = random.sample(range(len(key)), 2)
        key[index1], key[index2] = key[index2], key[index1]
    return ''.join(key)


def fitness(key, encrypted_text, dictionary_file, letter_freq_file):
    """Calculate the fitness of a decryption key."""
    # Load dictionary and letter frequency
    dictionary = load_dictionary(dictionary_file)
    letter_freq = load_letter_freq(letter_freq_file)

    # Decrypt the text
    decrypted_text = encrypted_text.translate(str.maketrans(key, GENE_RANGE))

    # Calculate the fitness
    num_valid_words = sum(1 for word in decrypted_text.split() if word in dictionary)
    letter_counts = {letter: 0 for letter in GENE_RANGE}
    for letter in decrypted_text:
        if letter in GENE_RANGE:
            letter_counts[letter] += 1
    letter_freq_decrypted = {letter: count/len(decrypted_text) for letter, count in letter_counts.items()}
    fitness_score = num_valid_words + sum(min(letter_freq.get(letter, 0), letter_freq_decrypted.get(letter, 0)) for letter in GENE_RANGE)
    return fitness_score


def genetic_algorithm(dictionary_file, letter_freq_file, population_size=100, key_length=26, mutation_rate=0.1,
                      max_generations=1000):
    # Load the dictionary file and letter frequency file
    dictionary = load_dictionary_file(dictionary_file)
    letter_freq = load_letter_frequency_file(letter_freq_file)

    # Generate an initial population of decryption keys
    population = generate_population(population_size, key_length)

    # Evaluate the initial population
    fitness_scores = evaluate_population(population, letter_freq, dictionary)

    # Track the best decryption key and its fitness score
    best_key = population[0]
    best_fitness = fitness_scores[0]

    # Start the main loop of the genetic algorithm
    for generation in range(max_generations):
        # Select the parents for the next generation
        parents = select_parents(population, fitness_scores)

        # Generate the offspring for the next generation
        offspring = generate_offspring(parents, population_size)

        # Mutate the offspring
        offspring = mutate_offspring(offspring, mutation_rate)

        # Evaluate the offspring
        offspring_fitness_scores = evaluate_population(offspring, letter_freq, dictionary)

        # Select the survivors for the next generation
        population, fitness_scores = select_survivors(population, fitness_scores, offspring, offspring_fitness_scores)

        # Update the best decryption key and its fitness score
        if fitness_scores[0] > best_fitness:
            best_key = population[0]
            best_fitness = fitness_scores[0]

        # Print the progress every 10 generations
        if generation % 10 == 0:
            print(f"Generation {generation}: Best Fitness = {best_fitness:.2f}")

    # Return the best decryption key
    return best_key


def decrypt_file(decryption_key):
    encrypted_text = load_encrypted_file(ENCRYPTED_FILE)
    decrypted_text = encrypted_text.translate(str.maketrans(decryption_key, GENE_RANGE))
    print(decrypted_text)


# Run the genetic algorithm and decrypt the file
dictionary_file = 'dict.txt'
letter_freq_file = 'Letter_freq.txt'
best_decryption_key = genetic_algorithm(dictionary_file, letter_freq_file)
decrypt_file(best_decryption_key)
