
import argparse
import os
import shutil
import jax

from evojax.task.slimevolley import SlimeVolley
from evojax.policy.mlp import MLPPolicy
from evojax.algo import CMA
from evojax import Trainer
from evojax import util


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--pop-size', type=int, default=128, help='ES population size.')
    parser.add_argument(
        '--hidden-size', type=int, default=20, help='Policy hidden size.')
    parser.add_argument(
        '--num-tests', type=int, default=100, help='Number of test rollouts.')
    parser.add_argument(
        '--n-repeats', type=int, default=16, help='Training repetitions.')
    parser.add_argument(
        '--max-iter', type=int, default=5, help='Max training iterations.')
    parser.add_argument(
        '--test-interval', type=int, default=50, help='Test interval.')
    parser.add_argument(
        '--log-interval', type=int, default=10, help='Logging interval.')
    parser.add_argument(
        '--seed', type=int, default=123, help='Random seed for training.')
    parser.add_argument(
        '--init-std', type=float, default=0.5, help='Initial std.')
    parser.add_argument(
        '--gpu-id', type=str, help='GPU(s) to use.')
    parser.add_argument(
        '--debug', action='store_true', help='Debug mode.')
    config, _ = parser.parse_known_args()
    return config


def main(config):
    log_dir = './log/slimevolley'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    logger = util.create_logger(
        name='SlimeVolley', log_dir=log_dir, debug=config.debug)
    logger.info('EvoJAX SlimeVolley')
    logger.info('=' * 30)

    max_steps = 3
    train_task = SlimeVolley(test=False, max_steps=max_steps)
    test_task = SlimeVolley(test=True, max_steps=max_steps)
    policy = MLPPolicy(
        input_dim=train_task.obs_shape[0],
        hidden_dims=[config.hidden_size, ],
        output_dim=train_task.act_shape[0],
        output_act_fn='tanh',
    )
    solver = CMA(
        pop_size=config.pop_size,
        param_size=policy.num_params,
        init_stdev=config.init_std,
        seed=config.seed,
        logger=logger,
    )
    # Train.
    trainer = Trainer(
        policy=policy,
        solver=solver,
        train_task=train_task,
        test_task=test_task,
        max_iter=config.max_iter,
        log_interval=config.log_interval,
        test_interval=config.test_interval,
        n_repeats=config.n_repeats,
        n_evaluations=config.num_tests,
        seed=config.seed,
        log_dir=log_dir,
        logger=logger,
    )
    trainer.run(demo_mode=False)

    # Test the final model.
    src_file = os.path.join(log_dir, 'best.npz')
    tar_file = os.path.join(log_dir, 'model.npz')
    shutil.copy(src_file, tar_file)
    trainer.model_dir = log_dir
    trainer.run(demo_mode=True)

    # Visualize the policy.
    task_reset_fn = jax.jit(test_task.reset)
    policy_reset_fn = jax.jit(policy.reset)
    step_fn = jax.jit(test_task.step)
    action_fn = jax.jit(policy.get_actions)
    best_params = trainer.solver.best_params[None, :]
    key = jax.random.PRNGKey(0)[None, :]

    task_state = task_reset_fn(key)
    policy_state = policy_reset_fn(task_state)
    screens = []
    for _ in range(max_steps):
        action, policy_state = action_fn(task_state, best_params, policy_state)
        #print(f"State is {task_state}")
        task_state, reward, done = step_fn(task_state, action)
        screens.append(SlimeVolley.render(task_state))

    gif_file = os.path.join(log_dir, 'slimevolley.gif')
    screens[0].save(gif_file, save_all=True, append_images=screens[1:],
                    duration=40, loop=0)
    logger.info('GIF saved to {}.'.format(gif_file))


if __name__ == '__main__':
    configs = parse_args()
    if configs.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = configs.gpu_id
    main(configs)