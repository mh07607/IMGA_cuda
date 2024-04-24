import numpy as np
from random import randint, shuffle, uniform


def read_graph_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

        # Initialize variables to store graph size and edges
        n_vertices = 0
        edges = []

        # Iterate through the lines in the file
        for line in lines:
            # Skip comment lines
            if line.startswith('c'):
                continue
            # Parse graph size from the 'p edge' line
            elif line.startswith('p edge'):
                parts = line.split()
                n_vertices = int(parts[2])
            # Parse edge definitions
            elif line.startswith('e'):
                parts = line.split()
                v1 = int(parts[1])
                v2 = int(parts[2])
                edges.append((v1, v2))

        # Create an adjacency matrix representation of the graph
        graph = np.zeros((n_vertices, n_vertices), dtype=int)
        for edge in edges:
            v1, v2 = edge
            graph[v1 - 1][v2 - 1] = 1
            graph[v2 - 1][v1 - 1] = 1

        return graph
def printGraph(graph):
    print([row for row in graph])

class GraphColoringGeneticAlgorithm:
    def __init__(self, graph, population_size=60, mutation_chances=[0.65, 0.5, 0.15], num_generations=50):
        self.graph = graph
        self.population_size = population_size
        self.mutation_chances = mutation_chances
        self.num_generations = num_generations
        self.n = len(graph)
        print(self.n)
        self.max_colors = self.get_max_colors()
        self.fitness_array = np.zeros(self.population_size)

    def get_max_colors(self): 
        max_num_colors = 0
        for row in self.graph:
            curr_max = sum(row)
            if curr_max > max_num_colors:
                max_num_colors = curr_max
        return max_num_colors

    def create_chromosome(self):
        ## Each solution is represented in a 1D array where the index of each item in that array maps to the index of a vertex in the graph
        return np.random.randint(1, self.max_colors + 1, size=self.n) 

    def create_population(self):
        return np.array([self.create_chromosome() for _ in range(self.population_size)])

    def calc_fitness(self, chromosome):
        penalty = 0
        for vertex1 in range(self.n):
            for vertex2 in range(vertex1, self.n):
                if self.graph[vertex1][vertex2] == 1 and chromosome[vertex1] == chromosome[vertex2]:
                    penalty += 1
        return penalty

    def truncation_selection(self, population):
        fitness_result = [self.calc_fitness(chromosome) for chromosome in population]
        sorted_chromosomes = []
        delete = 45
        for _ in range(delete):
            best = fitness_result[0]
            best_idx = 0
            for j in range(len(fitness_result)):
                if fitness_result[j] < best:
                    best = fitness_result[j]
                    best_idx = j
            fitness_result[best_idx] = 0
            sorted_chromosomes.append(population[best_idx])
        return sorted_chromosomes

    def tournament_selection(self, population):
        new_population = []
        for _ in range(2):
            shuffle(population)
            for i in range(0, self.population_size - 1, 2):
                if self.calc_fitness(population[i]) < self.calc_fitness(population[i + 1]):
                    new_population.append(population[i])
                else:
                    new_population.append(population[i + 1])
        return new_population
    
    def best_fitness(self):
        return min(self.fitness_array)

    def one_point_crossover(self, parent1, parent2):
        split_point = randint(2, self.n - 2)
        child1 = np.concatenate((parent1[:split_point], parent2[split_point:]))
        child2 = np.concatenate((parent2[:split_point], parent1[split_point:]))
        return child1, child2

    def mutation(self, chromosome, chance):
        possible = uniform(0, 1)
        if chance <= possible:
            for vertex1 in range(self.n):
                for vertex2 in range(vertex1, self.n):
                    if self.graph[vertex1][vertex2] == 1 and chromosome[vertex1] == chromosome[vertex2]:
                        chromosome[vertex1] = randint(1, self.max_colors)
        return chromosome

    
    def run_generation(self,parent_selection, survivor_selection, population, offspring, num_gen = 1):
        parents = []
        for i in range(num_gen):
            if parent_selection == 1:
                parents = self.random_selection(offspring)
            elif parent_selection == 2:
                parents = self.binary_tournament_selection(offspring)
            elif parent_selection == 3:
                parents = self.fitness_proportional_selection(offspring)
            elif parent_selection == 4:
                parents = self.rank_selection(offspring)
            elif parent_selection == 5:
                parents = self.truncation_selection(offspring)
            
            for j in range(0,offspring,2):
                selected_parents = random.sample(parents, 2)
                self.island += self.crossover(selected_parents[0], selected_parents[1], self.mutation_rate)
            
            if survivor_selection == 1:
                self.island = self.random_selection(population)
            elif survivor_selection == 2:
                self.island = self.binary_tournament_selection(population)
            elif survivor_selection == 3:
                self.island = self.fitness_proportional_selection(population)
            elif survivor_selection == 4:
                self.island = self.rank_selection(population)
            elif survivor_selection == 5:
                self.island = self.truncation_selection(population)

            print("the total populatin size is ", len(self.island))

            # print("the best fit for the ", i, " generation is: ", self.best_fitness())

    # def evolve(self):
    #     max_num_colors = self.max_colors
    #     check_count = 0
    #     failed_colors = 0

    #     print(f'Trying to color with {max_num_colors} colors')
    #     while True:
    #         #create population
    #         population = self.create_population() 
    #         for i in range(self.population_size):
    #             self.fitness_array[i] = self.calc_fitness(population[i])

    #         #parent selection

            




    #         best_fitness = self.calc_fitness(population[0])
    #         fittest = population[0]

    #         generation = 0
    #         num_generations = 50

    #         while best_fitness != 0 and generation != num_generations:
    #             generation += 1

    #             population = self.tournament_selection(population)

    #             if len(population) % 2 != 0:
    #                 population.pop()

    #             children_population = []
    #             shuffle(population)
    #             for i in range(0, len(population) - 1, 2):
    #                 child1, child2 = self.one_point_crossover(population[i], population[i + 1])
    #                 children_population.append(child1)
    #                 children_population.append(child2)

    #             for chromosome in children_population:
    #                 if generation < 200:
    #                     chromosome = self.mutation(chromosome, self.mutation_chances[0])
    #                 elif generation < 400:
    #                     chromosome = self.mutation(chromosome, self.mutation_chances[1])
    #                 else:
    #                     chromosome = self.mutation(chromosome, self.mutation_chances[2])

    #             for i in range(len(population), self.population_size):
    #                 population.append(self.create_chromosome())

    #             population = children_population
    #             best_fitness = self.calc_fitness(population[0])
    #             fittest = population[0]
    #             for individual in population:
    #                 if self.calc_fitness(individual) < best_fitness:
    #                     best_fitness = self.calc_fitness(individual)
    #                     fittest = individual

    #             if best_fitness == 0:
    #                 break

    #         if best_fitness == 0:
    #             print(f'{max_num_colors} colors succeeded! Trying {max_num_colors - 1} colors')
    #             max_num_colors -= 1
    #             check_count = 0
    #         else:
    #             if check_count != 2 and max_num_colors > 1:
    #                 failed_colors = max_num_colors

    #                 if check_count == 0:
    #                     print(f'{max_num_colors} failed. For safety, checking for improvement with {max_num_colors} colors again')
    #                 if check_count == 1:
    #                     print(f'{max_num_colors} failed. For safety, checking for improvement with {max_num_colors - 1} colors')
    #                     max_num_colors -= 1

    #                 check_count += 1
    #                 continue
    #             if max_num_colors > 1:
    #                 print(f'Graph is {failed_colors + 1} colorable')
    #             else:
    #                 print(f'Graph is {max_num_colors + 1} colorable')
    #             break


class IslandModels:

    def __init__(self, num_islands, population_size_per_island, num_generations, migration_interval, migration_rate, mutation_rate, tournament_size, migration_stratergy):
        self.num_islands = num_islands  
        self.population_size_per_island = population_size_per_island 
        self.num_generations = num_generations 
        self.migration_interval = migration_interval  # informing the number of genererations after migrations occurs
        self.migration_rate = migration_rate # number of individuals transfered during each migration
        self.mutation_rate = mutation_rate 
        self.tournament_size = tournament_size

        self.islands = [Island(population_size_per_island, mutation_rate) for _ in range(num_islands)]
        self.migration_stratergy = migration_stratergy

    def lcs_distance(seq1, seq2):
        
        # 2D array to store the lengths of LCS for subsequences
        m, n = len(seq1), len(seq2)
        lcs_matrix = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Compute the LCS lengths for all subproblems
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    lcs_matrix[i][j] = lcs_matrix[i - 1][j - 1] + 1
                else:
                    lcs_matrix[i][j] = max(lcs_matrix[i - 1][j], lcs_matrix[i][j - 1])
        

        return lcs_matrix[m][n]



    def migrate(self, stratergy:int = 1):
        '''Migrate individuals between islands based on the specified migration strategy.'''
        '''stratergy: int, migration strategy to use. 1 for ring based migration, 2 for best individual migration, 3 for gene based similarity migration. Default is 1.'''
        # Select migrants from each island
        if stratergy == 1: #ring based migration
            migrants_per_island = int(self.population_size_per_island * self.migration_rate)
            migrants = [island.truncation_selection(migrants_per_island) for island in self.islands]
            
            # Migrate between islands in a circular fashion
            for i in range(self.num_islands):
                target_island = (i + 1) % self.num_islands
                self.islands[target_island].island.extend(migrants[i])
                # After migration, trim each island's population to maintain size
                self.islands[target_island].island = self.islands[target_island].truncation_selection(self.population_size_per_island)

        elif stratergy == 2: # best individual migration

            for i in range(self.num_islands):
            # Find the best individual in the island
                best_individual = min(self.islands[i].island, key=lambda chromo: chromo.get_distance())
                
                # Migrate the best individual to the next island
                target_island = (i + 1) % self.num_islands
                self.islands[target_island].island.append(best_individual)
                
                # Trim the population size of the target island
                self.islands[target_island].island = self.islands[target_island].truncation_selection(self.population_size_per_island)

        elif stratergy == 3: # gene based similarity migration

            for i in range(self.num_islands):
                # Calculate the centroid of the current island population
                centroid = np.mean([chromo.indi for chromo in self.islands[i].island], axis=0)
                print("HEY:",centroid)
                
                # Calculate the genetic similarity between individuals in the current island and individuals in other islands
                similarity_scores = []
                for j in range(self.num_islands):

                    if j != i:  # Exclude the current island

                        # Calculate the centroid of the other island population
                        other_centroid = np.mean([chromo.indi for chromo in self.islands[j].island], axis=0)

                        
                        # # Calculate the Euclidean distance between centroids
                        # distance = np.sqrt(np.sum((centroid - other_centroid) ** 2))

                        # Calculate the LCS distance between the two populations
                        distance = self.lcs_distance(centroid, other_centroid)

                        
                        similarity_scores.append((j, distance))
                    
                
                similarity_scores.sort(key=lambda x: x[1]) # Sort islands based on genetic similarity
                
                # Migrate individuals from the most genetically similar island
                target_island = similarity_scores[0][0]
                migrants_per_island = int(self.population_size_per_island * self.migration_rate)
                migrants = self.islands[i].truncation_selection(migrants_per_island)
                self.islands[target_island].island.extend(migrants)
                self.islands[target_island].island = self.islands[target_island].truncation_selection(self.population_size_per_island)


    def evolve(self):
        for generation in range(self.num_generations):
            # Evolve each island
            for island_index, island in enumerate(self.islands):
                # Assuming the arguments are in order: parent selection method, survivor selection method, population size, offspring size
                Island.run_generation(2, 5, self.population_size_per_island, self.tournament_size)
                best_fitness = Island.best_fitness()
                print(f"Best fitness for Island {island_index + 1} in generation {generation + 1}: {best_fitness}")
            
            # Perform migration at the specified interval
            
            if (generation + 1) % self.migration_interval == 0:
                self.migrate()
                print(f"Migration occurred at generation {generation + 1}.")

    
        
# Set parameters
num_islands = 10
population_size_per_island = 100
num_generations = 1000
migration_rate = 0.3
mutation_rate = 0.2
tournament_size = 10

migration_stratergy = 3
migration_interval = None

if migration_stratergy == 1:
    migration_interval = 10
if migration_stratergy == 2 or migration_stratergy == 3:
    migration_interval = 25





if __name__ == '__main__':
    file_path = 'queen11_11.col'  
    graph = read_graph_from_file(file_path)
    ####HERE, WHAT EXACTLY IS TOURNAMENT SIZE####  
    model = IslandModels(num_islands, population_size_per_island, num_generations, migration_interval, migration_rate, mutation_rate, tournament_size, migration_stratergy)
    model.evolve()



    