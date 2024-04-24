import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import bisect as bi

"""IN PROGRESS"""
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
    

