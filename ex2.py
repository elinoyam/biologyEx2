import random
from collections import Counter,defaultdict

# Global variables
POPULATION_SIZE = 100
MAX_GENERATIONS = 150
MUTATION_RATE = 1
REPLACEMENT_RATE = 0.15
REPLACEMENT_SIZE = int(POPULATION_SIZE * REPLACEMENT_RATE)
EPSILON = 0.001
NO_IMPROVEMENT_THRESHOLD = 15
ELITE_SELECTION = 0.2
OPTIMIZATION_SWAPS_COUNT = 1
DO_DARWIN = False # set to True to use Darwin's optimization
DO_LAMARCK = True # set to True to use Lamarck's optimization
FITNESS_COUNTER = 0
LETTERS = "abcdefghijklmnopqrstuvwxyz"
LETTERS_COUNT = 26

with open("dict.txt", "r") as dict_file:
    dictionary = set(word.strip() for word in dict_file)

with open("Letter_Freq.txt", "r") as freq_file:
    letter_frequencies = {}
    letter_frequencies_lines = freq_file.read().splitlines()
    for line in letter_frequencies_lines:
        if line.strip() != "":
            freq, letter = line.strip().split()
            letter_frequencies[letter.lower()] = float(freq)

with open("Letter2_Freq.txt", "r") as freq2_file:
    letter2_frequencies = {}
    letter_frequencies_lines = freq2_file.read().splitlines()
    for line in letter_frequencies_lines:
        if line.strip() != "":
            freq, letter = line.split("\t")
            if freq != '':
                letter2_frequencies[letter.lower()] = float(freq)

with open("enc.txt", "r") as enc_file:
    encoded_text = enc_file.read().strip()


def letters_freq_in_text(text):
    freq_map = Counter(text)
    text_len = len(text)
    freq_map = {letter: count / text_len for letter, count in freq_map.items()}
    return freq_map

def letters2_freq_in_text(text):
    freq_map = defaultdict(int)
    text_len = len(text)
    for i in range(len(text) - 1):
        letter_pair = text[i:i + 2]
        freq_map[letter_pair] += 1

    for letter_pair in freq_map:
        freq_map[letter_pair] /= text_len

    return freq_map

def fitness(decryption_key):
    global FITNESS_COUNTER
    FITNESS_COUNTER += 1
    decrypted_text = decrypt_text(decryption_key)
    decrypted_text_set = set(decrypted_text.split(" "))

    letter_freq = letters_freq_in_text(decrypted_text)
    letter_pair_freq = letters2_freq_in_text(decrypted_text)
    letters_matching = 0

    words_matching = len(decrypted_text_set.intersection(dictionary))

    for letter in letter_frequencies:
        if letter in letter_freq and letter in letter_frequencies:
            letters_matching += (1 - abs(letter_freq[letter] - letter_frequencies[letter])) ** 2

    for letter_pair in letter2_frequencies:
        if letter_pair in letter_pair_freq and letter_pair in letter2_frequencies:
            letters_matching += (1 - abs(letter_pair_freq[letter_pair] - letter2_frequencies[letter_pair])) ** 2

    return words_matching + letters_matching


def decrypt_text(decryption_key):
    decrypted_text = ""
    for letter in encoded_text:
        if letter in decryption_key:
            decrypted_text += decryption_key[letter]
        else:
            decrypted_text += letter
    return decrypted_text


def generate_population():
    LETTERS = "abcdefghijklmnopqrstuvwxyz"
    population = []
    for _ in range(POPULATION_SIZE):
        decryption_key = list(LETTERS)
        random.shuffle(decryption_key)
        population.append({letter: decrypted for letter, decrypted in zip(LETTERS, decryption_key)})
    return population


def crossover(parent1, parent2):
    # create a child from crossing over the parents at a random point in the decryption key dictionary
    # while making sure that the child's decryption key is valid (i.e. no duplicate values)
    child = {}
    crossover_point = random.randint(0, LETTERS_COUNT - 1)
    for i in range(crossover_point):
        child[LETTERS[i]] = parent1[LETTERS[i]]
    for i in range(crossover_point, len(LETTERS)):
        child[LETTERS[i]] = parent2[LETTERS[i]]

    # check if the child's decryption key is valid
    # if not, swap one of the child's duplicate values with a letter that doesn't exist in the child's decryption key
    letters_in_child = set()
    letters_to_check = [char for char in LETTERS]
    for key, letter in child.items():
        if letter not in letters_in_child:
            letters_in_child.add(letter)
            letters_to_check.remove(letter)
        else:  # letter that was already in the child's decryption key - swap it with a letter that wasn't seen yet
            child[key] = random.choice(letters_to_check)
            letters_in_child.add(child[key])
            letters_to_check.remove(child[key])

    return child

def mutate(decryption_key):
    # swap two random letters in the decryption key
    letter1, letter2 = random.choices(LETTERS, k=2)
    decryption_key[letter1], decryption_key[letter2] = decryption_key[letter2], decryption_key[letter1]

    return decryption_key


def select_parents(population, fitness_scores,tournament_size = 5):
    parents = []
    for i in range(2):
        #select random individuals from the population to compete in the tournament
        tournament = random.sample(range(len(population)), tournament_size)
        #select the individual with the highest fitness score from the tournament to be a parent
        winner = tournament[0]
        for j in tournament:
            if fitness_scores[j] > fitness_scores[winner]:
                winner = j
        parents.append(population[winner])
    return parents


def replace_population(population, offspring, fitness_scores):
    worst_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:3 * REPLACEMENT_SIZE]

    # Find indices of individuals with best fitness scores from the existing population
    best_population_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[
                              :REPLACEMENT_SIZE]

    # Find indices of individuals with best fitness scores from the offspring
    best_offspring_indices = sorted(range(len(offspring)), key=lambda i: fitness(offspring[i]),
                                    reverse=True)[:REPLACEMENT_SIZE]

    # replacing 3*REPLACEMENT_SIZE worst individuals with the best individuals from the population, offspring and
    # the best individuals from the offspring. keeping REPLACEMT_SIZE times best individual.
    # saves REPLACEMENT_SIZE top indices from population, and REPLACEMENT_SIZE top indices from offspring
    for i in range(REPLACEMENT_SIZE):
        population[worst_indices[i]] = population[best_population_indices[0]]
        population[worst_indices[i + REPLACEMENT_SIZE]] = population[best_population_indices[i]]
        population[worst_indices[i + 2 * REPLACEMENT_SIZE]] = offspring[best_offspring_indices[i]]

    return population


def optimized_mutation(solution, solution_fitness, number_of_swaps):
    for _ in range(number_of_swaps):
        solution = mutate(solution)

    return solution


def darwin_mutation(population, fitness_scores):
    # for each solution in the population, create a mutated version of it
    mutations_fitness = [fitness(optimized_mutation(solution, score, OPTIMIZATION_SWAPS_COUNT)) for solution, score in zip(population, fitness_scores)]

    # if the mutated version has better fitness than the original solution, replace the original fitness with the mutated version
    darwin_fitness = [max(fitness_scores[i], mutations_fitness[i]) for i in range(len(fitness_scores))]

    return darwin_fitness


def lamarck_mutation(population, fitness_scores):
    # for each solution in the population, create a mutated version of it
    mutations = [optimized_mutation(solution, score, OPTIMIZATION_SWAPS_COUNT) for solution, score in zip(population, fitness_scores)]
    lamarck_fitness = []

    # if the mutated version has better fitness than the original solution, replace the original solution and fitness with the mutated version
    for i in range(len(population)):
        fitness_score = fitness(mutations[i])
        if fitness_score > fitness_scores[i]:
            population[i] = mutations[i]
            lamarck_fitness.insert(i, fitness_score)
        else:
            lamarck_fitness.insert(i, fitness_scores[i])

    return lamarck_fitness


def genetic_algorithm():
    population = generate_population()
    no_change = 0
    prev_best_fitness = 0
    best_solution_fitness = 0

    for generation in range(MAX_GENERATIONS):
        if generation > 0:
            elite_size = int(POPULATION_SIZE * ELITE_SELECTION)  # Select top ELITE_SELECTION of the population
            elite = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i], reverse=True)[:elite_size]
            next_generation = [population[i] for i in elite]
            while len(next_generation) < POPULATION_SIZE:
                parent1, parent2 = select_parents(population, fitness_scores)
                offspring = crossover(parent1, parent2)
                if not DO_LAMARCK and not DO_DARWIN and random.random() < MUTATION_RATE:
                    offspring = mutate(offspring)
                next_generation.append(offspring)

            # Replace population with next generation
            ######decide which one of the two methods we prefer -----------------------------
            population = replace_population(population, next_generation, fitness_scores)
            #population = next_generation

        fitness_scores = [fitness(decryption_key) for decryption_key in population]
        if DO_DARWIN:
            fitness_scores = darwin_mutation(population, fitness_scores)
        if DO_LAMARCK:
            fitness_scores = lamarck_mutation(population, fitness_scores)


        # find the index of the best solution in the population
        best_solution_fitness = max(fitness_scores)
        best_solution_index = fitness_scores.index(best_solution_fitness)
        best_solution = population[best_solution_index]

        # if the best solution didn't change for 10 generations, stop
        print(f'generation {generation}: best fitness = {best_solution_fitness}')

        if best_solution_fitness - prev_best_fitness < EPSILON:
            no_change += 1
        else:
            no_change = 0

        if no_change == NO_IMPROVEMENT_THRESHOLD:
            break

        prev_best_fitness = best_solution_fitness # save the best fitness score for the next generation

    return best_solution


best_decryption_key = genetic_algorithm()
decrypted_text = decrypt_text(best_decryption_key)
# Save decrypted text to file
with open("plain.txt", "w") as plain_file:
    plain_file.write(decrypted_text)
with open("perm.txt", "w") as perm_file:
    for letter, decrypted_letter in best_decryption_key.items():
        perm_file.write(f"{letter}\t{decrypted_letter}\n")

expected_decryption_key = {'a': 'y', 'b': 'x', 'c': 'i', 'd': 'n', 'e': 't', 'f': 'o', 'g': 'z', 'h': 'j', 'i': 'c',
                           'j': 'e',
                           'k': 'b', 'l': 'l', 'm': 'd', 'n': 'u', 'o': 'k', 'p': 'm', 'q': 's', 'r': 'v', 's': 'p',
                           't': 'q',
                           'u': 'r', 'v': 'h', 'w': 'w', 'x': 'g', 'y': 'a', 'z': 'f'}
total_keys = len(expected_decryption_key)

correct = sum(1 for letter in expected_decryption_key if expected_decryption_key[letter] == best_decryption_key[letter])
print(f"amount of correct keys: {correct} / {total_keys}")
print(f"correctness percentage: {correct / total_keys * 100}%")
print(f"number of calls to fitness function is: {FITNESS_COUNTER}")

