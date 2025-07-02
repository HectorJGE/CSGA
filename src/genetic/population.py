from src.genetic.individual import Individual
import random

class Population:
    individuals = []
    
    def __init__(self, size, item_pool, set_size):
        self.individuals = [Individual(random.sample(item_pool, set_size)) for _ in range(size)]

    def evolve(self):
        # Selección
        # Cruce
        # Mutación
        # Evaluación
        pass

def tournament_selection(population, k):
    selected = random.sample(population, k)
    return max(selected, key=lambda ind: ind.fitness)