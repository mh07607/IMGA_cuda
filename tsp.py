from types import FunctionType
from typing import List
import numpy as np
import copy
import random
from tqdm import tqdm
from evolutionary_algorithm import Individual, EvolutionaryAlgorithm
import matplotlib.pyplot as plt

'''Fetching Data'''
def read_and_convert_to_dict(file_path):
	data_dict = {}
	city_list = []
	with open(file_path, 'r') as file:
		for line in file:
			# Split the line into parts
			parts = line.strip().split()

			try:
				# Extract key and coordinates
				key = int(parts[0])
				# adding the city to the list as well to keep record
				city_list.append(key)
				coordinates = tuple(map(float, parts[1:]))
				# Create dictionary entry
				data_dict[key] = coordinates
			except:
				continue
	return city_list, data_dict
file_path = 'data/qa194.tsp'  # Replace with the path to your text file
city_list, city_dict = read_and_convert_to_dict(file_path)


def get_distance(x: tuple, y: tuple):
	# x and y are two 2D points each not 2 coordinates of one 2D point.
	return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**(1/2)

def distance(individual):
	distance = 0
	num_individuals = len(individual)
	for i in range(len(individual)):
		distance += get_distance(city_dict[individual[i]], city_dict[individual[(i+1) % num_individuals]])
	return distance

class TSP_Path(Individual):
	def __init__(self, genome):
		fitness = distance(genome)
		super().__init__(genome, fitness)
	
	def mutate(self) -> None:
		# mutation defined by reversing orders for now and to be changed
		# porposed algorithm would be to randomly swich 10 neighboring places such that it's neighbors have less distance as compared with the one
		rand_index1 = random.randint(0, len(self.genome)-1)
		rand_index2 = random.randint(0, len(self.genome)-1)

		self.genome[rand_index1], self.genome[rand_index2] = self.genome[rand_index2], self.genome[rand_index1]
		self.fitness = distance(self.genome)


def random_intercity_paths(population_size: int) -> List[TSP_Path]:
	population = []
	for i in range(population_size):
		genome = copy.deepcopy(city_list)
		np.random.shuffle(genome)
		population.append(TSP_Path(genome))
	return population


def TSP_random_length_crossover(parent1: TSP_Path, parent2: TSP_Path):
	length = len(parent1.genome)
	start = random.randint(1, length-3)
	end = random.randint(start, length-2)

	offspring1 = [None] * length
	offspring2 = [None] * length

	offspring1[start:end+1] = parent1.genome[start:end+1]
	offspring2[start:end+1] = parent2.genome[start:end+1]

	pointer = end + 1
	parent1_pointer = end + 1
	parent2_pointer = end + 1

	while None in offspring1:
		if parent2.genome[parent2_pointer] not in offspring1:
			offspring1[pointer % length] = parent2.genome[parent2_pointer]
			pointer += 1
		parent2_pointer = (parent2_pointer + 1) % length

	pointer = 0

	while None in offspring2:
		if parent1.genome[parent1_pointer] not in offspring2:
			offspring2[pointer % length] = parent1.genome[parent1_pointer]
			pointer += 1
		parent1_pointer = (parent1_pointer + 1) % length

	offspring1 = TSP_Path(offspring1)
	offspring2 = TSP_Path(offspring2)

	return offspring1, offspring2


class TSP_EvolutionaryAlgorithm(EvolutionaryAlgorithm):
	def run(self, num_iterations: int=10, num_generations: int=1000) -> tuple:
		best_fitnesses = [[] for _ in range(num_iterations)]
		average_fitnesses = [[] for _ in range(num_iterations)]
		x_offset = num_generations // 20

		for j in range(num_iterations):
			for i in tqdm(range(num_generations), desc='Iteration '+str(j+1)):
				self.run_generation()
				if(i % x_offset == 0):
					best_individual, average_fitness = self.get_average_and_best_individual()
					print("\nAverage fitness: ", average_fitness, ", Best value: ", best_individual.fitness)
					best_fitnesses[j].append(best_individual.fitness)
					average_fitnesses[j].append(average_fitness)

			self.population = self.initial_population_function(self.population_size)


		return best_individual, best_fitnesses, average_fitnesses


'''Test Run'''
# tsp = TSP_EvolutionaryAlgorithm(
#       initial_population_function = random_intercity_paths,
#       parent_selection_function = 'truncation',
#       survivor_selection_function = 'random',
#       cross_over_function = TSP_random_length_crossover,
#       population_size = 100,
#       mutation_rate = 0.5,
#       num_offsprings=20
#       )
# tsp.run()



''' Generating Graphs '''
selection_pairs = [
                    ('fitness', 'random', 100, 0.5, 50),
                    ('binary', 'truncation', 100, 0.9, 20), 
                    ('truncation', 'truncation', 100, 0.9, 50), 
                    ('random', 'random', 100, 0.5, 10),
                    ('fitness', 'rank', 100, 0.5, 50),
                    ('truncation', 'random', 100, 0.5, 50),
                    ('rank', 'binary', 100, 0.5, 100),
                  ]

num_generations = 400
num_iterations = 1
x_offset = num_generations // 20


for parent_selection, survivor_selection, population_size, mutation_rate, num_offsprings in selection_pairs:
	print(parent_selection, survivor_selection)
	tsp = TSP_EvolutionaryAlgorithm(
		initial_population_function = random_intercity_paths,
		parent_selection_function = parent_selection,
		survivor_selection_function = survivor_selection,
		cross_over_function = TSP_random_length_crossover,
		population_size = population_size,
		mutation_rate = mutation_rate,
		num_offsprings=num_offsprings
	)

	best_individual, best_fitnesses, average_fitnesses = tsp.run(num_generations=num_generations, num_iterations=num_iterations)
	best_fitnesses = np.array(best_fitnesses).T.tolist()
	average_fitnesses = np.array(average_fitnesses).T.tolist()
	x = []
	y1 = []
	y2 = []

	for i in range(len(best_fitnesses)):
		x.append(i * x_offset)
		y1.append(np.average(best_fitnesses[i]))
		y2.append(np.average(average_fitnesses[i]))

	plt.figure()
	plt.plot(x, y1, label='Average best fitness')
	plt.plot(x, y2, label='Average average fitness')

	plt.xlabel('Number of generations')
	plt.ylabel('Average average/best fitness values')
	plt.title(parent_selection + ', ' +survivor_selection + ', ' +
			str(population_size) + ', ' +
			str(mutation_rate) + ', ' +
			str(num_offsprings))
	plt.legend()

	plt.savefig(parent_selection+'_'+survivor_selection+'.png')  # Save as PNG
	print(best_individual.genome)
	# Plot path of best individual in the last iteration. Commented out because plots are not good
	# path_coordinates = [city_dict[city] for city in best_individual.genome]
	# path_coordinates.append(path_coordinates[0])
	# x_coords, y_coords = zip(*path_coordinates)
	# plt.figure()
	# plt.plot(x_coords, y_coords, marker='o', linestyle='-')
	# plt.xlabel('X-coordinate')
	# plt.ylabel('Y-coordinate')
	# plt.title('Parent selection: ' + parent_selection + '\n' +
	#         'Survivor selection: ' + survivor_selection + '\n' +
	#         'Population size: ' + str(population_size) + '\n' +
	#         'Mutation rate: ' + str(mutation_rate) + '\n' +
	#         'Number of offsprings: ' + str(num_offsprings))
	# plt.savefig('data/tsp_analysis/'+parent_selection+'_'+survivor_selection+'_path.png')
