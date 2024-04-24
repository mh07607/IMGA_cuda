import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import bisect as bi

global option_1
global option_2

class Graph:
    def __init__(self, filename):
        with open(filename, 'r') as file:
            lines = file.readlines()
        
        edge_lines = [line.strip().split() for line in lines if line.startswith('e')]
        edges = [(int(edge[1]), int(edge[2])) for edge in edge_lines]
        
        graph = nx.Graph()
        graph.add_edges_from(edges)

        print("Number of nodes:", graph.number_of_nodes())
        print("Number of edges:", graph.number_of_edges())
        self.graph = graph
        self.getGraph()

    def getGraph(self):
        return self.graph
    

class AntColonyOptimization:
    def __init__(self, graph, num_ants, max_iterations, alpha, beta, evaporation_rate, Q=1.0):
        self.graph = graph
        self.num_ants = num_ants
        self.max_iterations = max_iterations
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.evaporation_rate = evaporation_rate

        self.num_nodes = nx.number_of_nodes(graph)
        self.pheromone_matrix = self.initialize_pheromone_matrix()
        self.adj_matrix = self.initialise_adjacency_matrix()
        max_degree = max(self.graph.degree(), key=lambda x: x[1])[1]
        self.Colors = [i for i in range(1,max_degree+1)] # Create colors list
        self.Ants = self.create_ants() # Create ants colony
        self.Colouring = np.zeros(self.num_ants)
        self.best_fitness_history = np.zeros(self.max_iterations)
        self.avg_fitness_history = np.zeros(self.max_iterations)
                                            
    # create a pheromone matrix with init pheromone values: 1 if nodes not adjacent, 0 if adjacent
    def initialize_pheromone_matrix(self):
        matrix = np.ones((self.num_nodes, self.num_nodes), float)
        for node in self.graph:  
            for adj_node in self.graph.neighbors(node):
                matrix[node - 1, adj_node - 1] = 0
        return matrix
    
    def initialise_adjacency_matrix(self):
        adj_matrix = np.zeros((self.num_nodes, self.num_nodes), int)
        for node in self.graph.nodes():
            for adj_node in self.graph.neighbors(node):
                adj_matrix[node-1, adj_node-1] = 1

        return adj_matrix

    def create_ants(self):
        Ants_List = []
        for i in range(self.num_ants):
            Ants_List.append(Ant(self.graph, self.alpha, self.beta, self.Colors, self.Q))
        return Ants_List
    
    def update_pheromone_matrix(self):
        summation = np.zeros((self.num_nodes,self.num_nodes))
        for ant in range(self.num_ants):
            summation = summation + self.Ants[ant].helper_updater()
        self.pheromone_matrix = ((1 - self.evaporation_rate) *self.pheromone_matrix ) + summation
    
            
    def optimize(self): #run the ACO Algorithm
        bestSoFar = float('inf')
        average = []
        for iteration in range(self.max_iterations):
            # Ant Colony Optimization loop
            for i in range(self.num_ants):
                self.Ants[i].initialise()
                self.Colouring[i] = self.Ants[i].colouring(self.pheromone_matrix)

            self.update_pheromone_matrix()
            average.append(np.mean(self.Colouring))
            self.avg_fitness_history[iteration] = sum(average)/len(average)
       
            min_fitness = np.min(self.Colouring)
            if min_fitness < bestSoFar:
                bestSoFar = min_fitness
            self.best_fitness_history[iteration] = bestSoFar
            
            print(f"Iteration {iteration}")
            print(f"Best Fitness:-  {round(bestSoFar,3)} ")
            print(f"Average Fitness:- {round(self.avg_fitness_history[iteration],3)}")
        




        #plot the results
        Iterations = [i for i in range(self.max_iterations)]
        
        fig, (ax1,ax2) = plt.subplots(1, 2)
        ax1.plot(Iterations,self.best_fitness_history,'tab:red')
        ax1.set_xlabel("Iterations")
        ax1.set_title("Best Fitness So Far")
        ax1.set_ylabel("Best Fitness So Far")

        ax2.plot(Iterations,self.avg_fitness_history,'tab:blue')
        ax2.set_xlabel("Iterations")
        ax2.set_title("Average Fitness So Far")
        ax2.set_ylabel("Average Fitness So Far")
        plt.show()
        #also save the plots
        plt.savefig('ACO_Results.png')

class Ant:
    def __init__(self, graph : nx, alpha, beta, colours : list, Q):
        self.graph = graph
        self.num_nodes = nx.number_of_nodes(graph)
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.option_1 = 1 #for first node
        self.option_2 = 1 #for desireability

        self.colours_available = [] #total available colours
        self.colours = colours #list of colours
        self.colours_map = {} #mapping of colours to set of nodes
        self.colour_of_nodes = {} #dictionary of relation between each node to a colour

        self.possible_nodes = [] #list of nodes to visit
        self.distance = 0 #num colours used

    def initialise(self):
        self.colours_available = self.colours.copy()
        self.possible_nodes = self.graph.nodes()
        
        for i in range(self.num_nodes):
            self.colour_of_nodes[i] = None # 0 to num_nodes - 1, 0 means uncoloured
        for c in self.colours_available:
            self.colours_map[c] = set() #initialise colours map to empty sets 1 to max_degree + 1
    
    def getHeuristic(self, node, W: set, B: set, Option):
        if Option == 1:
            return len(B.intersection(set(self.graph.neighbors(node))))
        elif Option == 2:
            return len(set(self.graph.neighbors(node)).intersection(B.union(W)))
        else:
            return len(W) - len(W.intersection(set(self.graph.neighbors(node))))
            
           
    def getFirstNode(self, set_of_nodes, Option):
        if Option == 1:
            return random.choice(list(set_of_nodes))
        elif Option == 2:
            degrees = {node: self.graph.degree(node) for node in set_of_nodes}
            return max(degrees, key=degrees.get)
        else:
            print("Invalid Option")
    
    def getPheromoneTrail(self, node, q, pheromone_matrix):
        pheromoneSum = 0
        for n in self.colours_map[q]:
            pheromoneSum += pheromone_matrix[node - 1][n-1]
        return pheromoneSum / len(self.colours_map[q])
    
    def helper_updater(self):
        temp = np.zeros((self.num_nodes, self.num_nodes))
        for v1 in range(self.num_nodes):
            for v2 in range(self.num_nodes):
                if self.colour_of_nodes[v1] == self.colour_of_nodes[v2] and (v1 != v2):
                    temp[v1, v2] = self.Q / self.distance
        return temp
    
    def CalculateProbabilities(self, set_nodes : list, pheromone_trails, heuristic_values: set):
        #checks if there are any vertices with zero desirability (all desirabilities are zero). If so, it sets their desirabilities to 1
        if all(value == 0 for value in heuristic_values.values()):
            for node in heuristic_values:
                heuristic_values[node] = 1 

        top = np.zeros(len(set_nodes))
        set_nodes = list(set_nodes)
        bottom = 0 #sum of top
        for i in range(len(set_nodes)):
            top[i] = pheromone_trails[set_nodes[i]] ** self.alpha * heuristic_values[set_nodes[i]] ** self.beta
            bottom += top[i]
        probabilities = np.cumsum(top/bottom)
        #find the index where a randomly generated value between 0 and 1 would be inserted into the cumulative probabilities array so list remains sorted
        index = bi.bisect_left(probabilities,random.uniform(0,1))
        return set_nodes[index] 
    
    def colouring(self, pheromone_matrix):
        distance = 0 #num colours used
        set_unvisited_nodes = set(self.possible_nodes)
        num_coloured_nodes = 0 #num nodes coloured

        while num_coloured_nodes < self.num_nodes:
            num_coloured_nodes += 1
            distance += 1
            first = self.getFirstNode(set_unvisited_nodes, self.option_1)  #get first node   
            self.colours_map[distance].add(first)
            self.colour_of_nodes[first] = distance

            #find neighbours of first (unvisited) node
            neighbours = set(self.graph.neighbors(first)).intersection(set_unvisited_nodes)

            set_uncoloured_impossible = set()
            while set_unvisited_nodes.difference((neighbours.union({first}))):
                num_coloured_nodes += 1
                set_uncoloured_impossible = set_uncoloured_impossible.union(neighbours) #add neighbours to set of uncoloured nodes
                set_unvisited_nodes = set_unvisited_nodes.difference(neighbours.union({first})) #remove neighbours from unvisited nodes

                desireability = {node: self.getHeuristic(node, set_unvisited_nodes, set_uncoloured_impossible, self.option_2) for node in set_unvisited_nodes}
                phenoTrail = {node: self.getPheromoneTrail(node, distance, pheromone_matrix) for node in set_unvisited_nodes}
            
                first = self.CalculateProbabilities(set_unvisited_nodes, phenoTrail, desireability)
                neighbours = set(self.graph.neighbors(first)).intersection(set_unvisited_nodes)
                self.colours_map[distance].add(first)
                self.colour_of_nodes[first] = distance

            set_unvisited_nodes = set_uncoloured_impossible.union(neighbours)
        
        self.distance = distance
        return distance


# Example usage:
file_1 = 'queen11_11.col'
file_2 = "le450_15b.col"
graph = Graph(file_2)
#colony = AntColonyOptimization(graph.graph, num_ants=20, max_iteratios=100, alpha=0.8, beta=0.8, evaporation_rate=0.1)
colony = AntColonyOptimization(graph.graph, num_ants=20, max_iterations=100, alpha=1.5, beta=1.5, evaporation_rate=0.5)
colony.optimize()

#print("Vertex colors:", best_solution)
