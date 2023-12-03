import numpy as np
import matplotlib.pyplot as plt
from typing import Union

SeedType = Union[int, float, str, bytes, bytearray, None]

def plot_particles(particulas, global_best=None):
    x = particulas[:][0]
    y = particulas[:][1]

    plt.scatter(x, y, c='blue', marker='o', label='Partículas')

    if global_best is not None:
        plt.scatter(global_best[0], global_best[1], c='red', marker='*', label='Melhor Global')

    plt.xlabel('Coordenada X')
    plt.ylabel('Coordenada Y')
    plt.title('PSO - Otimização de Partículas')
    plt.legend()
    plt.savefig('particulas.png')

    plt.show()

def rastrigin(particle: list) -> float:
    size = len(particle)
    cos_values = [np.cos(2 * np.pi * x) for x in particle]
    sum_result = sum(np.power(particle, 2) - 10 * np.array(cos_values))
    result = 10 * size + sum_result

    return result

def fitness_fn(particle):
    n = len(particle)

    # Critérios de penalização
    r = 10
    s = 12

    # Cálculo do fitness
    fx = 10 * n + np.sum(particle**2 - 10 * np.cos(2 * np.pi * particle))  # Fitness do indivíduo
    gx = np.sum(r * np.maximum(0, np.sin(2 * np.pi * particle) + 0.5))   # gi(x) desigualdade
    hx = np.sum(s * np.abs(np.cos(2 * np.pi * particle) + 0.5))        # hi(x0) igualdade
    px = fx + gx + hx                                # Fitness com penalizações aplicadas

    return px

def init_matrix(swarm_size: int, dimensions: int, inferior_limit: float, upper_limit: float, seed: SeedType = None):
    np.random.seed(seed)
    return np.random.uniform(low=inferior_limit, high=upper_limit, size=(swarm_size, dimensions))