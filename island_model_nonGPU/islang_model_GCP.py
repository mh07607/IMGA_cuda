import numpy as np
import random
import copy


def read_graph_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        n_vertices = 0
        edges = []
        for line in lines:
            if line.startswith('c'):
                continue
            elif line.startswith('p edge'):
                parts = line.split()
                n_vertices = int(parts[2])
            elif line.startswith('e'):
                parts = line.split()
                v1 = int(parts[1])
                v2 = int(parts[2])
                edges.append((v1, v2))
        graph = np.zeros((n_vertices, n_vertices), dtype=int)
        for edge in edges:
            v1, v2 = edge
            graph[v1 - 1][v2 - 1] = 1
            graph[v2 - 1][v1 - 1] = 1

        return graph
    

def printGraph(graph):
    print([row for row in graph])



class Island:
    def __init__(self, graph, population_size, mutation_rate):
        self.graph = graph
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.n = len(graph)
        self.maxNumColors = self.get_max_colors()
        self.island = self.initialize_island()  #represents population, an array of chromosomes, where each chromosome is as defined in create_chromosome
        self.bestFitness = self.calc_fitness(self.island[0]) 
        self.fittest = self.island[0]
        self.checkCount = 0
        self.failedColors = 0

        
        

        


    def initialize_island(self):
        return np.array([self.create_chromosome() for _ in range(self.population_size)])  #create population

    def create_chromosome(self):
        ## Each solution is represented in a 1D array where the index of each item in that array maps to the index of a vertex in the graph
        return np.random.randint(1, self.maxNumColors + 1, size=self.n) 

    def get_max_colors(self): 
        max_num_colors = 0
        for row in self.graph:
            curr_max = sum(row)
            if curr_max > max_num_colors:
                max_num_colors = curr_max
        return max_num_colors

    def calc_fitness(self, chromosome):
        penalty = 0
        for vertex1 in range(self.n):
            for vertex2 in range(vertex1, self.n):
                if self.graph[vertex1][vertex2] == 1 and chromosome[vertex1] == chromosome[vertex2]:
                    penalty += 1
        return penalty
    
    def truncation_selection(self, size):
        result = []
        result = copy.deepcopy(self.island)
        result = sorted(result, key=lambda k: self.calc_fitness(k))
        return result[:size]

    def random_selection(self, size):
        result = []
        for _ in range(size):
            rand_num = random.randint(0, self.population_size - 1)
            result.append(self.island[rand_num])
        return result
    
    def tournament_selection(self, size):
        selections = []
        participants = []
        for i in range(size):
            ele = random.choice(self.island)
            
            participants.append(ele)
        for _ in range(2):
            winner = min(participants, key=lambda chromosome: self.calc_fitness(chromosome))
            selections.append(winner)
        return selections
        

    
    def fitness_proportional_selection(self,size):
        total_fitness = sum(self.calc_fitness(chromo) for chromo in self.island)
        selection_probs = [self.calc_fitness(chromo) / total_fitness for chromo in self.island]
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=selection_probs)
        return [self.island[i] for i in selected_indices[:size]]
    


    def rank_selection(self, size):
        self.island = sorted(self.island, key=lambda chromo: self.calc_fitness(chromo))
        ranks = np.arange(1, self.population_size + 1)
        total_rank = np.sum(ranks)
        selection_probs = ranks / total_rank
        selected_indices = np.random.choice(range(self.population_size), size=self.population_size, replace=True, p=selection_probs)
        return [self.island[i] for i in selected_indices[:size]]
    
    def best_fitness(self):
        temp = []
        for i in self.island:
            temp.append(self.calc_fitness(i))

        return min(temp)

    def crossover(self, parent1, parent2): #one point crossover
        split_point = random.randint(2, self.n - 2)
        child1 = np.concatenate((parent1[:split_point], parent2[split_point:]))
        child2 = np.concatenate((parent2[:split_point], parent1[split_point:]))
        return child1, child2
    
    # def crossover(self, chrom1, chrom2, mut_rate):
    #     # scheme to be implemented using crossover scheme where a randomly selected set of parent 1 would displace the randomly selected set of parent 2
    #     #start should be between 25th and 50th percentile of len of chromomosome
    #     #end should be between 75th and 100th percentile of len of chromomosome

    #     len_chromosome = self.n
    #     start = random.randint(len_chromosome * 20 // 100, len_chromosome * 40 // 100) 
    #     end = random.randint(len_chromosome * 60 // 100, len_chromosome * 80 // 100)

    #     # start = random.randint(1, len_chromosome // 2)
    #     # end = random.randint(len_chromosome // 2 + 1, len_chromosome - 2)

    #     parent1 = chrom1
    #     parent2 = chrom2

    #     offspring1 = [None] * len_chromosome
    #     offspring2 = [None] * len_chromosome

    #     offspring1[start:end+1] = parent1[start:end+1]
    #     offspring2[start:end+1] = parent2[start:end+1]
    #     pointer = end + 1
    #     parent1_pointer = end + 1
    #     parent2_pointer = end + 1
    #     while pointer != start:

    #         if parent2[parent2_pointer] not in offspring1:    
    #             offspring1[pointer] = parent2[parent2_pointer]
    #             pointer += 1
    #         parent2_pointer += 1
    #         if parent2_pointer == len(offspring1):
    #             parent2_pointer = 0
    #         if pointer == len(offspring1):
    #             pointer = 0
    #     pointer = end + 1

    #     while pointer != start:

    #         if parent1[parent1_pointer] not in offspring2: 
    #             offspring2[pointer] = parent1[parent1_pointer]
    #             pointer += 1
    #         parent1_pointer += 1
    #         if parent1_pointer == len(offspring2):
    #             parent1_pointer = 0
    #         if pointer == len(offspring2):
    #             pointer = 0

        
    #     offspring1 = self.mutation(offspring1, mutation_rate)
    #     offspring2 = self.mutation(offspring2, mutation_rate)

    #     return [offspring1, offspring2] 

    def mutation(self, chromosome, chance):
        possible = random.uniform(0, 1)
        if chance <= possible:
            for vertex1 in range(self.n):
                for vertex2 in range(vertex1, self.n):
                    if self.graph[vertex1][vertex2] == 1 and chromosome[vertex1] == chromosome[vertex2]:
                        chromosome[vertex1] = random.randint(1, self.maxNumColors)
        return chromosome
    


    
    def run_generation(self, parent_selection, survivor_selection, population, offspring, island_num):  
        flag = 1

        #create new population with a reduced chromosome according to current maxNumColours
        self.island = self.initialize_island()
        self.bestFitness = self.calc_fitness(self.island[0]) 
        self.fittest = self.island[0]




        print(f'Trying to color island {island_num + 1} with {self.maxNumColors} colors')
        tries = 0
        while self.bestFitness != 0 and tries < 100:
            tries += 1
            # Parent Selection
            parents = []
            if parent_selection == 1:
                parents = self.random_selection(offspring)
            elif parent_selection == 2:
                parents = self.tournament_selection(offspring) #here offspring is the tournament size
            elif parent_selection == 3:
                parents = self.fitness_proportional_selection(offspring)
            elif parent_selection == 4:
                parents = self.rank_selection(offspring)
            elif parent_selection == 5:
                parents = self.truncation_selection(offspring)

            #Crossover
            child1, child2 =  self.crossover(parents[0], parents[1])
            
            #Mutation
            child1 = self.mutation(child1, self.mutation_rate)
            child2 = self.mutation(child2, self.mutation_rate)

            #Add children to the population
            self.island = np.append(self.island, [child1], axis=0)
            self.island = np.append(self.island, [child2], axis=0)

            # Survivor Selection
            if survivor_selection == 1:
                self.island = self.random_selection(population)
            elif survivor_selection == 2:
                self.island = self.fitness_proportional_selection(population)
            elif survivor_selection == 3:
                self.island = self.rank_selection(population)
            elif survivor_selection == 4:
                self.island = self.truncation_selection(population)

            
            for individual in self.island:
                temp = self.calc_fitness(individual)
                if(temp < self.bestFitness):
                    self.bestFitness = temp
                    self.fittest = individual
            

            if self.bestFitness == 0:
                    break
            
        if tries >= 100:
            print("Tries Exhausted!")
        if self.bestFitness == 0:
            print(f'{self.maxNumColors} colors succeeded! Trying {self.maxNumColors - 1} colors')
            self.maxNumColors -= 1
            self.checkCount = 0
        else:
            if self.checkCount != 2 and self.maxNumColors > 1:
                failedColors = self.maxNumColors
                if self.checkCount == 0:
                    print(f'{self.maxNumColors} failed. For safety, checking for improvement with {self.maxNumColors} colors again')
                if self.checkCount == 1:
                    print(f'{self.maxNumColors} failed. For safety, checking for improvement with {self.maxNumColors - 1} colors')
                    self.maxNumColors -= 1
                self.checkCount += 1

            if self.maxNumColors > 1 and self.checkCount == 2:
                print(f'Graph is {failedColors + 1} colorable')
            elif self.maxNumColors <= 1 and self.checkCount == 2:
                print(f'Graph is {self.maxNumColors + 1} colorable')

            flag = 0

        return flag
            
    


class IslandModels:

    def __init__(self, num_islands, population_size_per_island, num_generations, migration_interval, migration_rate, mutation_rate, tournament_size, migration_stratergy, graph, survivor_method, parent_method):
        self.num_islands = num_islands  
        self.population_size_per_island = population_size_per_island 
        self.num_generations = num_generations 
        self.migration_interval = migration_interval  # informing the number of genererations after migrations occurs
        self.migration_rate = migration_rate # number of individuals transfered during each migration
        self.mutation_rate = mutation_rate 
        self.tournament_size = tournament_size
        self.survivor_method = survivor_method
        self.parent_method = parent_method
        self.graph = graph

        self.islands = [Island( graph, population_size_per_island, mutation_rate) for _ in range(num_islands)]
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
                best_individual =  min(self.islands[i].island, key=lambda chromosome: self.islands[i].calc_fitness(chromosome))
                
                # Migrate the best individual to the next island
                target_island = (i + 1) % self.num_islands
                self.islands[target_island].island.append(best_individual)
                
                # Trim the population size of the target island
                self.islands[target_island].island = self.islands[target_island].truncation_selection(self.population_size_per_island)

    def evolve(self):
        
        checker = np.ones(len(self.islands))
        island_best_solutions = [1000] * (len(self.islands) + 1)
        for generation in range(self.num_generations):
            # Evolve each island
            for island_index, island in enumerate(self.islands):
                # Assuming the arguments are in order: parent selection method, survivor selection method, population size, offspring size
                if checker[island_index] == 1:
                    temp = island.run_generation(self.parent_method, self.survivor_method, self.population_size_per_island, self.tournament_size, island_num=island_index)
                    checker[island_index] = temp
                best_fitness = island.maxNumColors

                if best_fitness < island_best_solutions[island_index + 1]:
                    island_best_solutions[island_index + 1] = best_fitness
                print(f"Best fitness for Island {island_index + 1} in generation {generation + 1}: {best_fitness}")
            
            # Perform migration at the specified interval
            
            if (generation + 1) % self.migration_interval == 0:
                self.migrate()
                print(f"Migration occurred at generation {generation + 1}.")
        
        print(f"Best overall fitness is:{min(island_best_solutions)}")

    
        
# Set parameters
num_islands = 3
population_size_per_island = 10
num_generations = 200
migration_rate = 0.5
mutation_rate = 0.2
tournament_size = 10

migration_stratergy = 1 #3 doesn't exist yet
migration_interval = None

if migration_stratergy == 1:
    migration_interval = 10
if migration_stratergy == 2 or migration_stratergy == 3:
    migration_interval = 25

if __name__ == '__main__':
    file_path = '/Users/asadullahchaudhry/Github HU/IMGA_cuda/island_model_nonGPU/queen11_11.col'  
    graph = read_graph_from_file(file_path)
    ####HERE, WHAT EXACTLY IS TOURNAMENT SIZE####  
    survivor_method = 4
    parent_method = 2
    model = IslandModels(num_islands, population_size_per_island, num_generations, migration_interval, migration_rate, mutation_rate, tournament_size, migration_stratergy, graph, survivor_method, parent_method)
    model.evolve()
    
#refrence: https://github.com/soumildatta/GeneticGraphColoring/blob/main/geneticColoring.py