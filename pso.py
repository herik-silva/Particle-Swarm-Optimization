from utils import rastrigin, fitness_fn, init_matrix, SeedType
import numpy as np
import random as rand
import os
from time import time

def particle_swarm_optimization(dimensions: int, swarm_size: int, iterations: int, inf_limit: float, up_limit: float, seed: SeedType, experiment: int, objective_function = rastrigin):
    particles = init_matrix(swarm_size, dimensions, inf_limit, up_limit, seed)
    velocity = init_matrix(swarm_size, dimension, inf_limit, up_limit, seed)

    initial_position = particles.copy()
    fitness = np.array([objective_function(particle) for particle in particles])

    pbest = particles.copy()

    gbest_index = np.where(fitness == min(fitness))[0]
    index = gbest_index[0]
    global_best = particles[index, :]
    fitness_global_best = objective_function(particles[index][:])
    all_gbest_fitness = []

    # Parâmetros do PSO
    inertia_w = 0.5
    personal_weight = 1.2
    global_weight = 1.6

    for index in range(iterations):
        os.system("clear")
        print(f"Progresso PSO {int((index/iterations)*100)}% (Experimento {experiment})")
        for particle_i in range(swarm_size):
            r1, r2 = rand.random(), rand.random()
            vet_inertia = inertia_w * velocity[particle_i][:]
            vet_local = (personal_weight * r1) * (pbest[particle_i][:] - particles[particle_i][:])
            vet_global = (global_weight * r2) * (global_best - particles[particle_i][:])

            velocity[particle_i][:] = vet_inertia + vet_local + vet_global

            velocity = np.clip(velocity, -0.255, 0.255)

            particles[particle_i][:] += velocity[particle_i][:]

            new_fitness = objective_function(particles[particle_i][:])

            fitness[particle_i] = np.round(new_fitness, 3)

            if new_fitness < objective_function(pbest[particle_i][:]):
                pbest[particle_i][:] = particles[particle_i][:]

        particles = np.clip(particles, inf_limit, up_limit)
        min_fit = np.min(fitness)
        if min_fit <= fitness_global_best:
            gbest_index = np.where(fitness == min(fitness))[0]
            index = gbest_index[0]
            global_best = particles[index][:]
            fitness_global_best = objective_function(particles[index][:])

        all_gbest_fitness.append(min_fit)
        
    # plot_particles(particles, global_best)
    return [np.round(fitness_global_best, 3), np.round(global_best, 3), [particles, global_best, all_gbest_fitness, initial_position]]


dimension = 5
population = 130
iterations = 100
inf_limit = -5.12
up_limit = 5.12
seed = 79

times = []
best_time = 10000
bests = []
particles = []
initial_position = []
global_best = []
best_fit = 10000
all_fitness = []
best_case = 0
MAX = 20

for index in range(MAX):
    print(f"Progresso {(index/MAX) * 100}% ({index}/{MAX})")
    init_time = time()
    result = particle_swarm_optimization(dimension, population, iterations, inf_limit, up_limit, seed, index+1, fitness_fn)
    end_time = time()
    times.append(end_time - init_time)
    all_fitness.append(result[2][2])
    if result[0] < best_fit:
        best_fit = result[0]
        [particles, global_best, bests, initial_position] = result[2]
        best_case = index+1
        best_time = end_time - init_time

mean_time = np.mean(times)
mean_fitness = np.mean(all_fitness)
std_dev_time = np.std(times)
std_dev_fitness = np.std(all_fitness)

print(f"Média de Fitness: {mean_fitness}")
print(f"Desvio Padrão de Fitness: {std_dev_fitness}")
print(f"Média de tempo: {mean_time}")
print(f"Desvio Padrão de tempo: {std_dev_time}")
print(f"Melhor caso: {best_case}")
print(f"Melhor tempo: {best_time}")
print(f"Melhor Fitness: {best_fit}")

print(f"Melhor fit: {result[0]}")
print(global_best)
