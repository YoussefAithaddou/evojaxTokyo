import argparse
import os
import shutil
import jax
from evojax.policy.mlp import MLPPolicy
from evojax.task.slimevolley import SlimeVolley
from evojax import util
import multiprocessing
import pickle
import neat
import jax.numpy as jnp


runs_per_net = 2


os.environ["JAX_TRACEBACK_FILTERING"] = "off"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

max_steps = 3000

train_task = SlimeVolley(test=False, max_steps=max_steps)
key = jax.random.PRNGKey(0)[None, :]



def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        
        task_state = train_task.reset(key)
        fitness = 0.0
        done = False
        while not done:
            observation=task_state.obs[0]

            action = net.activate(observation)
          
            task_state, reward, done =train_task.step(task_state, action)
            fitness += reward

        fitnesses.append(fitness)

    return jnp.mean(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)

def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)


if __name__ == '__main__':
    run()