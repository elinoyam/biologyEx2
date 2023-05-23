import random

# Load dictionary
with open("dict.txt", "r") as dict_file:
    dictionary = set(word.strip() for word in dict_file)

# Load letter frequencies
with open("Letter_Freq.txt", "r") as freq_file:
    letter_frequencies = {}
    letter_frequencies_lines = freq_file.read().splitlines()
    for line in letter_frequencies_lines:
        if line.strip() != "":
            freq, letter = line.strip().split()
            letter_frequencies[letter.lower()] = float(freq)

# Load encoded text
with open("enc.txt", "r") as enc_file:
    encoded_text = enc_file.read().strip()

# Define genetic algorithm parameters
POPULATION_SIZE = 100
MUTATION_RATE = 0.1
FITNESS_THRESHOLD = 0.99
LETTERS = "abcdefghijklmnopqrstuvwxyz"
EPSILON = 0.002

# Generate initial population
def generate_population():
    population = []
    for _ in range(POPULATION_SIZE):
        decryption_key = list(LETTERS)
        random.shuffle(decryption_key)
        population.append({letter: decrypted for letter, decrypted in zip(LETTERS, decryption_key)})
    return population

# Fitness function
def fitness(decryption_key, include_frequency=True):
    letter_count = {}
    decrypted_text = ""
    for letter in encoded_text:
        decrypted_letter = decryption_key.get(letter, letter).replace(".", "").replace(",", "").replace(";", "")
        decrypted_text += decrypted_letter
        if letter in LETTERS:
            letter_count[decrypted_letter] = letter_count.get(decrypted_letter, 0) + 1

    # every letter that doesn't appear in the text is assigned a frequency of 0
    for letter in LETTERS:
        if letter not in letter_count:
            letter_count[letter] = 0

    # clean_decrypted_text = decrypted_text.replace(".", "").replace(",", "").replace(";", "")
    decrypted_words = decrypted_text.split()

    # calculate the fitness by the number of words in the decrypted text that are in the dictionary
    word_count = len(decrypted_words)
    correct_word_count = sum(1 for word in decrypted_words if word in dictionary)

    # result = correct_word_count / word_count

    # calculate the appearance percentage of each letter in the decrypted text
    if include_frequency:
        letter_appearance = {letter: count / len(decrypted_text) for letter, count in letter_count.items()}
        correct_letter_appearance = sum(1 for letter in letter_appearance if abs(letter_appearance[letter] - float(letter_frequencies[letter])) < EPSILON)
        # result = (correct_word_count + correct_letter_appearance) / (word_count + len(letter_appearance))

    return  0.9 * (correct_word_count / word_count) + 0.1 * (correct_letter_appearance / len(letter_appearance))

# Crossover
def crossover(parent1, parent2):
    # create a child from crossing over the parents at a random point in the decryption key dictionary
    # while making sure that the child's decryption key is valid (i.e. no duplicate values)
    child1, child2 = {}, {}
    crossover_point = random.randint(0, len(LETTERS) - 1)
    for i in range(crossover_point):
        child1[LETTERS[i]] = parent1[LETTERS[i]]
        child2[LETTERS[i]] = parent2[LETTERS[i]]
    for i in range(crossover_point, len(LETTERS)):
        child1[LETTERS[i]] = parent2[LETTERS[i]]
        child2[LETTERS[i]] = parent1[LETTERS[i]]

    # check if the child's decryption key is valid
    # if not, swap one of the child's duplicate values with a letter that doesn't exist in the child's decryption key
    for child in [child1, child2]:
        letters_in_child = set()
        letters_to_check = [char for char in LETTERS]
        for key, letter in child.items():
            if letter not in letters_in_child:
                letters_in_child.add(letter)
                letters_to_check.remove(letter)
            else: # letter that was already in the child's decryption key - swap it with a letter that wasn't seen yet
                child[key] = random.choice(letters_to_check)
                letters_in_child.add(child[key])
                letters_to_check.remove(child[key])

    return child1, child2

# Mutation
def mutate(decryption_key):
    for letter in decryption_key:
        if random.random() < MUTATION_RATE:
            swap_key = random.choice(list(decryption_key.keys()))
            decryption_key[letter], decryption_key[swap_key] = decryption_key[swap_key], decryption_key[letter]

# Genetic algorithm
def genetic_algorithm():
    population = generate_population()
    generation = 1
    include_freq = True

    # while True:
    for _ in range(200):
        # Evaluate fitness of the population
        fitness_scores = [fitness(decryption_key, include_freq) for decryption_key in population]

        # Check for convergence
        best_fitness = max(fitness_scores)
        # if best_fitness >= 0.45:
        #     include_freq = False
        if best_fitness >= FITNESS_THRESHOLD:
            break

        # Print best fitness
        print(f'generation {generation}: best fitness = {best_fitness}')
        new_population = []

        # add to the new population the best 10% of the population according to the fitness scores
        sorted_population = [x for _, x in sorted(zip(fitness_scores, population), key=lambda pair: pair[0], reverse=True)]
        best_population = sorted_population[:round(POPULATION_SIZE * 0.3)]
        new_population.extend(best_population)

        # crossover 95% of the population in random pairs
        while len(new_population) < POPULATION_SIZE:
            # Select parents for reproduction
            parent1, parent2 = random.choices(best_population, k=2)
            child1, child2 = crossover(parent1, parent2)
            new_population.append(child1)
            new_population.append(child2)

        # mutate 20% of the population
        mutants = random.choices(new_population, k=round(POPULATION_SIZE*0.4))
        # mutants = random.choices(new_population[round(POPULATION_SIZE * 0.3):], k=round(POPULATION_SIZE*0.4))
        for mutant in mutants :
            mutate(mutant)

        # replace the old population with the new population
        population = new_population
        generation += 1

    # Get the best decryption key and decrypted text
    best_index = max(range(POPULATION_SIZE), key=lambda i: fitness_scores[i])
    best_decryption_key = population[best_index]
    decrypted_text = "".join(best_decryption_key.get(letter, letter) for letter in encoded_text)

    # Save decrypted text to file
    with open("plain.txt", "w") as plain_file:
        plain_file.write(decrypted_text)

    # Save swapping table to file
    with open("perm.txt", "w") as perm_file:
        for letter, decrypted_letter in best_decryption_key.items():
            perm_file.write(f"{letter} {decrypted_letter}\n")

    print("Decryption completed successfully!")

# Run the genetic algorithm
genetic_algorithm()
