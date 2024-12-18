
import argparse
import os
import shutil
import jax

from evojax.task.slimevolley import SlimeVolley
from evojax.policy.mlp import MLPPolicy
from evojax.algo import CMA
from evojax import Trainer
from evojax import util
import neat
import multiprocessing
import jax.numpy as jnp



import pickle


# Fitness function for NEAT
def eval_genome(genome, config):
    # Create the neural network from the genome
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    key = jax.random.PRNGKey(0)[None, :]
    max_steps = 3000
    train_task = SlimeVolley(test=False, max_steps=max_steps)
    env = train_task
    state = env.reset(key)
    obs=state.obs
    done = False
    total_reward = 0

    while not done:
        # Use the neural network to compute actions
        inputs = jnp.array(obs[0])
        move = net.activate(inputs)
        #print("*"*10)
        #print(move)

        
        # Create the action array based on the condition that each value is >= 0.5
        action = [int(move[i] >= 0.5) for i in range(3)]

        if action[0]+action[2]==2:
            if move[0]>move[1]:
                action[1]=0
            else:
                move[0]=0

        # Convert to a JAX array and cast to integers
        action = jnp.array([action], dtype=jnp.int32)
        #print(action)
                
        state, reward, done= env.step(state, action)
        obs=state.obs

        total_reward += reward[0]

    return jnp.array(total_reward)


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

    #print(winner)



if __name__ == '__main__':
    run()