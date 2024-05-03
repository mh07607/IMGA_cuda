from types import FunctionType
from typing import List
from tqdm import tqdm
from abc import ABC, abstractmethod
import random
import copy
import numpy as np
import threading

class Individual():
  def __init__(self, genome, fitness):
    self.genome = genome
    self.fitness = fitness


  @abstractmethod
  def mutate(self) -> None:
    pass


class EvolutionaryAlgorithm():
  def __init__(self, 
               initial_population_function: FunctionType,
               parent_selection_function: str,
               survivor_selection_function: str,
               cross_over_function: FunctionType,
               population_size: int = 100,
               mutation_rate: float = 0.5,
               num_offsprings: int = 10):
    # __getattr__ can be used in taichi to get the function from the string
    selection_functions_string_map = {'truncation': self.truncation_selection,
                                      'random': self.random_selection,
                                      'binary': self.binary_tournament_selection,
                                      'rank': self.rank_selection,
                                      'fitness': self.fitness_proportional_selection}
    self.initial_population_function: FunctionType = initial_population_function
    self.population: List[Individual] = self.initial_population_function(population_size)
    self.population_size: int = population_size
    self.mutation_rate: float = mutation_rate
    self.parent_selection_function: FunctionType = selection_functions_string_map[parent_selection_function]
    self.survivor_selection_function: FunctionType = selection_functions_string_map[survivor_selection_function]
    self.cross_over_function: FunctionType = cross_over_function
    self.num_offsprings: int = num_offsprings


  ## selection functions
  def random_selection(self, num_selections: int) -> List[Individual]:
    survivors = []
    for i in range(num_selections):
      random_int = random.randint(0, len(self.population)-1)
      survivors.append(self.population[random_int])
    return survivors


  
  def truncation_selection(self, num_selections: int) -> List[Individual]:
    result = []
    result = copy.deepcopy(self.population)
    result.sort(key=lambda k : k.fitness)
    return result[:num_selections]
  

  def binary_tournament_selection(self, num_selections: int) -> List[Individual]:
    result= []
    for i in range(num_selections):
        ind1, ind2 = random.sample(self.population, 2)
        selected = ind1 if ind1.fitness < ind2.fitness else ind2
        result.append(selected)
    return result

  # relative fitness instead of absolute fitness
  def rank_selection(self, num_selections: int) -> List[Individual]:
    self.population.sort(key=lambda individual: individual.fitness, reverse=True)
    ranks = np.arange(1, len(self.population) + 1)
    total_rank = np.sum(ranks)
    selection_probs = ranks / total_rank
    selected_indices = np.random.choice(range(len(self.population)), size=num_selections, replace=True, p=selection_probs)
    return [self.population[i] for i in selected_indices]
  

  def fitness_proportional_selection(self, num_selections: int) -> List[Individual]:
    total_fitness = sum(1/individual.fitness for individual in self.population)
    selection_probs = [(1/individual.fitness) / total_fitness for individual in self.population]
    selected_indices = np.random.choice(range(len(self.population)), size=num_selections, replace=True, p=selection_probs)
    return [self.population[i] for i in selected_indices]


  def get_average_and_best_individual(self) -> tuple[Individual, float]:
    best_individual = self.population[0]
    cumulative_fitness = 0
    for individual in self.population:
      if(individual.fitness < best_individual.fitness):
        best_individual = individual
      cumulative_fitness += individual.fitness
    average_fitness = cumulative_fitness/len(self.population)
    return best_individual, average_fitness


  def get_total_fitness(self) -> float:
    total_fitness = 0
    for individual in self.population:
      total_fitness += individual.fitness
    return total_fitness


  def run_generation(self) -> None:
    parents = self.parent_selection_function(self.num_offsprings)

    # creating offspring
    for k in range(0, self.num_offsprings-1, 2):
      offspring1, offspring2 = self.cross_over_function(parents[k], parents[k+1])
      rand_num1, rand_num2 = random.randint(0,100)/100, random.randint(0,100)/100
      if rand_num1 <= self.mutation_rate:
        offspring1.mutate()
      if rand_num2 <= self.mutation_rate:
        offspring2.mutate()
      self.population.extend([offspring1, offspring2])

    self.population = self.survivor_selection_function(self.population_size)
  
  # def process_offspring_range(self, start, end, parents, lock):
  #   for k in range(start, end, 2):
  #       offspring1, offspring2 = self.cross_over_function(parents[k], parents[k + 1])
  #       rand_num1, rand_num2 = random.randint(0, 100) / 100, random.randint(0, 100) / 100
  #       if rand_num1 <= self.mutation_rate:
  #           offspring1.mutate()
  #       if rand_num2 <= self.mutation_rate:
  #           offspring2.mutate()
  #       with lock:
  #           self.population.extend([offspring1, offspring2])


  
  @abstractmethod
  def run():
    pass
    