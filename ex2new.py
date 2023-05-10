import random

# Load the input files
with open('dict.txt', 'r') as f:
    dictionary = f.read().splitlines()

with open('Letter_freq.txt', 'r') as f:
    letter_freq = {}
    for line in f:
        freq, letter = line.strip().split('\t')
        letter_freq[letter] = float(freq)

with open('Letter2_freq.txt', 'r') as f:
    letter2_freq = {}
    for line in f:
        if '#REF!' not in line and line != '\n':
            freq, letter2 = line.strip().split()
            letter2_freq[letter2] = float(freq)


# Define the fitness function
def fitness_function(key, dictionary, letter_freq, letter2_freq):
    # Decrypt the message using the key
    decrypted_message = decrypt(key, "enc.txt")

    # Calculate the fitness score
    score = 0
    for word in decrypted_message.split():
        if word in dictionary:
            score += 1
    for i in range(len(decrypted_message) - 1):
        if decrypted_message[i:i + 2] in letter2_freq:
            score += letter2_freq[decrypted_message[i:i + 2]]
    for letter in letter_freq:
        score += abs(decrypted_message.count(letter) / len(decrypted_message) - letter_freq[letter])
    return score


# Define the decryption function
def decrypt(key, enc_text):
    # Convert the key to a dictionary that maps encrypted letters to their corresponding decrypted letters
    key_dict = {}
    for i in range(26):
        key_dict[key[i]] = chr(i + ord('a'))

    # Decrypt the message
    decrypted_message = ''
    for c in enc_text:
        if c.isalpha():
            decrypted_message += key_dict[c]
        else:
            decrypted_message += c

    return decrypted_message


# Define the genetic algorithm function
def genetic_algorithm(dictionary, letter_freq, letter2_freq):
    # Set the genetic algorithm parameters
    population_size = 100
    mutation_rate = 0.01
    generations = 100

    # Initialize the population with random keys
    population = [list('ABCDEFGHIJKLMNOPQRSTUVWXYZ') for i in range(population_size)]
    for i in range(population_size):
        random.shuffle(population[i])

    # Iterate over the generations
    for generation in range(generations):
        # Calculate the fitness scores for all individuals
        fitness_scores = [fitness_function(key, dictionary, letter_freq, letter2_freq) for key in population]

        # Select the best individuals for reproduction
        selected_population = []
        for i in range(population_size // 2):
            # Choose two parents randomly from the population
            parent1 = random.choices(population, weights=fitness_scores)[0]
            parent2 = random.choices(population, weights=fitness_scores)[0]

            # Create two offspring by crossover
            offspring1 = parent1.copy()
            offspring2 = parent2.copy()
            crossover_point = random.randint(0, len(parent1))
            offspring1[:crossover_point] = parent2[:crossover_point]
            offspring2[:crossover_point] = parent1[:crossover_point]

            # Mutate the offspring
            for offspring in [offspring1, offspring2]:
                for j in range(len(offspring)):
                    if random.random() < mutation_rate:
                        k = random.randint(0, len(offspring) - 1)
                        offspring[j], offspring[k] = offspring[k], offspring[j]

            # Add the offspring to the selected population
            selected_population.append(offspring1)
            selected_population.append(offspring2)

        # Replace the old population with the selected population
        population = selected_population

        # Print the best key and fitness score of the current generation
        best_key = population[fitness_scores.index(max(fitness_scores))]
        print(
            f"Generation {generation + 1}: Best fitness score = {max(fitness_scores)}, Best key = {''.join(best_key)}")
        return ''.join(best_key)


best_key = genetic_algorithm(dictionary, letter_freq, letter2_freq)
decrypted_message = decrypt(best_key, "enc.txt")
print(decrypted_message)
