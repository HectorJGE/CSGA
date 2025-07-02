import random

class Individual:
    def __init__(self, movie_ids):
        self.movie_ids = movie_ids  # Lista de N movieIds únicos
        self.fitness = None

    def mutate(self, item_pool, mutation_rate=0.2):
        """Reemplaza aleatoriamente algunos ítems del individuo"""
        for i in range(len(self.movie_ids)):
            if random.random() < mutation_rate:
                replacement = random.choice(item_pool)["id"]
                while replacement in self.movie_ids:
                    replacement = random.choice(item_pool)["id"]
                self.movie_ids[i] = replacement

    def crossover(self, other):
        """Combina dos individuos para crear dos hijos"""
        cut = len(self.movie_ids) // 2
        child1_ids = self.movie_ids[:cut] + [mid for mid in other.movie_ids if mid not in self.movie_ids[:cut]]
        child2_ids = other.movie_ids[:cut] + [mid for mid in self.movie_ids if mid not in other.movie_ids[:cut]]
        return Individual(child1_ids[:len(self.movie_ids)]), Individual(child2_ids[:len(self.movie_ids)])