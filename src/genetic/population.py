import random
from src.genetic.individual import Individual
from src.genetic.fitness import evaluate

class Population:
    def __init__(self, size, item_pool, items_per_set):
        self.size = size
        self.item_pool = item_pool
        self.items_per_set = items_per_set
        self.individuals = self._initialize_population()

    def _initialize_population(self):
        """Crea una población inicial aleatoria"""
        population = []
        ids_pool = [item["id"] for item in self.item_pool]
        for _ in range(self.size):
            movie_ids = random.sample(ids_pool, self.items_per_set)
            population.append(Individual(movie_ids))
        return population

    def evolve(self):
        """Aplica selección, cruce y mutación para crear una nueva generación"""
        self.individuals.sort(key=lambda ind: ind.fitness or 0, reverse=True)

        # Selección: elitismo + torneo
        next_gen = self.individuals[:5]  # elitismo: conservar los 5 mejores

        while len(next_gen) < self.size:
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            child1, child2 = parent1.crossover(parent2)
            child1.mutate(self.item_pool)
            child2.mutate(self.item_pool)
            next_gen.extend([child1, child2])

        self.individuals = next_gen[:self.size]

        # Evaluar nueva población
        for ind in self.individuals:
            evaluate(ind, self.ratings_df)

    def _tournament_selection(self, k=3):
        competitors = random.sample(self.individuals, k)
        competitors.sort(key=lambda ind: ind.fitness or 0, reverse=True)
        return competitors[0]

    def set_ratings(self, ratings_df):
        self.ratings_df = ratings_df