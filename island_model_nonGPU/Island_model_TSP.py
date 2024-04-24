import numpy as np
import copy
import random


city_list = []
city_dict = {}

def read_and_convert_to_dict(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
            # Find the line index where node coordinates start
            node_coord_start = lines.index("NODE_COORD_SECTION\n") + 1
            # Iterate over lines starting from node_coord_start to extract city coordinates
            for line in lines[node_coord_start:]:
                if line.strip() == "EOF":
                    break
                city = int(line.strip().split()[0])
                city_list.append(city)
                city_dict[city] = tuple(map(float, line.strip().split()[1:]))
        return city_dict


class chromosome():
    def __init__(self, comb: list): # comb is a list of the cities in the order they are to be visited
        self.indi = comb[:]

        self.distance = 0
        self.chromosome_distance(self.indi) # allowing a new distance value to be generated


    def city_distance(self, x: tuple, y: tuple):
    # x and y are two 2D points each not 2 coordinates of one 2D point.
        return ((x[0]-y[0])**2 + (x[1]-y[1])**2)**0.5
    
    
    def get_distance(self):
        return self.distance
    

    def chromosome_distance(self, cities: list):
        for i in range(len(cities)-1):
            self.distance += self.city_distance(city_dict[cities[i]], city_dict[cities[i+1]])
            
    def mutation(self, mutation_rate:int):
        random_probability = random.random()
        if random_probability <= mutation_rate:     # Check if probability lies within the mutation rate
            # Perform swap mutation
            chromosome_length = len(self.indi)
            swap_index = random.randint(0, chromosome_length - 1)
            swap_index2 = random.randint(0, chromosome_length -1)
            self.indi[swap_index2], self.indi[swap_index] = self.indi[swap_index], self.indi[swap_index2]

    
    
         
class Island():
    
    def __init__(self, population_size_per_island, mutation_rate):
        # how the lists are structured:
        # islands = [island_1, island_2, .....]
        #island_1 = [choromosome1, chormosome2, ....] i.e the population of the island
         
        self.population_size = population_size_per_island #
        self.mutation_rate = mutation_rate 
        self.island = self.initialize_island()

    #randomly shuffling the cities to create a chromosome, and doing this len(population_size) times 
    def initialize_island(self): 
        island = []
        for _ in range(self.population_size):
            city_lst = copy.deepcopy(city_list)
            np.random.shuffle(city_list)
            choromo1 = chromosome(city_lst)
            island.append(choromo1)

        return island

    # would done idividually to each island and the chromosomes in it
    def crossover(self, chrom1, chrom2, mut_rate):
        # scheme to be implemented using crossover scheme where a randomly selected set of parent 1 would displace the randomly selected set of parent 2

        #start should be between 25th and 50th percentile of len of chromomosome
        #end should be between 75th and 100th percentile of len of chromomosome
        len_chromosome = len(chrom1.indi)
        start = random.randint(len_chromosome * 20 // 100, len_chromosome * 40 // 100) 
        end = random.randint(len_chromosome * 60 // 100, len_chromosome * 80 // 100)

        # start = random.randint(1, len_chromosome // 2)
        # end = random.randint(len_chromosome // 2 + 1, len_chromosome - 2)

        parent1 = chrom1.indi
        parent2 = chrom2.indi

        offspring1 = [None] * 194
        offspring2 = [None] * 194

        offspring1[start:end+1] = parent1[start:end+1]
        offspring2[start:end+1] = parent2[start:end+1]
        pointer = end + 1
        parent1_pointer = end + 1
        parent2_pointer = end + 1
        while pointer != start:

            if parent2[parent2_pointer] not in offspring1:    
                offspring1[pointer] = parent2[parent2_pointer]
                pointer += 1
            parent2_pointer += 1
            if parent2_pointer == len(offspring1):
                parent2_pointer = 0
            if pointer == len(offspring1):
                pointer = 0
        pointer = end + 1

        while pointer != start:

            if parent1[parent1_pointer] not in offspring2: 
                offspring2[pointer] = parent1[parent1_pointer]
                pointer += 1
            parent1_pointer += 1
            if parent1_pointer == len(offspring2):
                parent1_pointer = 0
            if pointer == len(offspring2):
                pointer = 0

        offspring1 = chromosome(offspring1)
        offspring2 = chromosome(offspring2)
        offspring1.mutation(mut_rate)
        offspring2.mutation(mut_rate)

        return [offspring1, offspring2] 

    def total_fitness(self):
        total = 0
        for i in self.island:
            total += i.get_distance()
        return total
    
    def best_fitness(self):
        best = self.island[0].get_distance()
        for i in self.island[1:]:
            if i.get_distance() < best:
                best = i.get_distance()
        return best
    
    def avg_fitnness(self):
        total = self.total_fitness()
        return total / self.population_size
    
    def random_selection(self, size):
        result = []
        for _ in range(size):
            rand_num = random.randint(0, self.population_size - 1)
            result.append(self.island[rand_num])
        return result

    def truncation_selection(self, size):
        result = []
        result = copy.deepcopy(self.island)
        result.sort(key=lambda k : k.get_distance())
        return result[:size]
    
    def binary_tournament_selection(self, size):
        result= []
        for i in range(size):
            ind1, ind2 = random.sample(self.island, 2)
            selected = ind1 if ind1.get_distance() < ind2.get_distance() else ind2
            result.append(selected)

        return result
    
    def fitness_proportional_selection(self,size):
        total_fitness = sum(chromo.distance for chromo in self.island)
        selection_probs = [chromo.distance / total_fitness for chromo in self.island]
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=selection_probs)
        return [self.island[i] for i in selected_indices[:size]]
    


    def rank_selection(self, size):
        self.island.sort(key=lambda chromo: chromo.distance)
        ranks = np.arange(1, self.population_size + 1)
        total_rank = np.sum(ranks)
        selection_probs = ranks / total_rank
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=selection_probs)
        return [self.island[i] for i in selected_indices[:size]]

    # settnig the number of generation to a deault of 1 rn 
    def run_generation(self,parent_sel, survivor_sel, popul, offsp, num_gen = 1 ):
        
        parents = []
        for i in range(num_gen):
            if parent_sel == 1:
                parents = self.random_selection(offsp)
            elif parent_sel == 2:
                parents = self.binary_tournament_selection(offsp)
            elif parent_sel == 3:
                parents = self.fitness_proportional_selection(offsp)
            elif parent_sel == 4:
                parents = self.rank_selection(offsp)
            elif parent_sel == 5:
                parents = self.truncation_selection(offsp)
            
            for j in range(0,offsp,2):
                selected_parents = random.sample(parents, 2)
                self.island += self.crossover(selected_parents[0], selected_parents[1], self.mutation_rate)
            
            if survivor_sel == 1:
                self.island = self.random_selection(popul)
            elif survivor_sel == 2:
                self.island = self.binary_tournament_selection(popul)
            elif survivor_sel == 3:
                self.island = self.fitness_proportional_selection(popul)
            elif survivor_sel == 4:
                self.island = self.rank_selection(popul)
            elif survivor_sel == 5:
                self.island = self.truncation_selection(popul)

            print("the total populatin size is ", len(self.island))

            # print("the best fit for the ", i, " generation is: ", self.best_fitness())



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
                island.run_generation(2, 5, self.population_size_per_island, self.tournament_size)
                best_fitness = island.best_fitness()
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


#bet fitness achieved 22398.0

# Example usage:
file_path = 'qa194.tsp'  # Replace with the path to your text file
result_dict = read_and_convert_to_dict(file_path)

# Initialize island model
model = IslandModels(num_islands, population_size_per_island, num_generations, migration_interval, migration_rate, mutation_rate, tournament_size, migration_stratergy)

# Run evolution
model.evolve()


