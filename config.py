"""Defines the hyperparameters configuration."""

import os
import argparse
import yaml
import io

from util import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        description="Arguments for RL training and evaluation")

    # Session configuration
    parser.add_argument('--name',
                        type=str,
                        default='default',
                        help="Name of this session / model to be tagged.")
    
    # Training setup
    parser.add_argument('--n_iter',
                        type=int,
                        default=3000000,
                        help='Number of training iterations (epochs).')
    parser.add_argument('--batch_size',
                        type=int,
                        default=256,
                        help="Batch size.")
    # optimization hyperparameters
    parser.add_argument('--lr',
                        type=float,
                        default=3e-4,
                        help="Learning rate.")

    # Data loading and model saving
    parser.add_argument(
        '--purge',
        action='store_true',
        help="Whether to purge existing logs and summaries in directory.")
    parser.add_argument('--train_ckpt_interval',
                        type=int,
                        default=50000,
                        help="Save training checkpoints every n iters.")
    parser.add_argument('--log_interval',
                        type=int,
                        default=1000,
                        help="Log stats every n iters.")
    
    parser.add_argument('--train_dir',
                        type=str,
                        default='train',
                        help="Directory to save training models and viz.")
    parser.add_argument('--eval_dir',
                        type=str,
                        default='eval',
                        help="Directory to save evaluation models and viz.")
    parser.add_argument('--log_dir',
                        type=str,
                        default='logs',
                        help="Directory to save logs.")

    return parser


def parse(root=os.path.dirname(os.path.abspath(__file__)),
          config_file=None,
          save_config=False):
    parser = get_parser()
    if config_file is not None:
        with open(config_file, 'r') as stream:
            config = yaml.safe_load(stream)
            args = parser.parse_args(config)
    else:
        args = parser.parse_args()

    # Configure directories
    args.train_dir = os.path.join(root, args.train_dir, args.name)
    args.eval_dir = os.path.join(root, args.eval_dir, args.name)

    # Create directories if needed
    if not os.path.isdir(args.train_dir):
        os.makedirs(args.train_dir)
    if not os.path.isdir(args.eval_dir):
        os.makedirs(args.eval_dir)
    if not os.path.isdir(args.log_dir):
        os.makedirs(args.log_dir)

    if save_config:
        save_file = 'config_{}.yaml'.format(args.model_name)
        with io.open(save_file, 'w') as outfile:
            yaml.dump(args, outfile)

    return args


def get_optimizer(args):
    # return tf.keras.optimizers.Adam(learning_rate=args.lr)
    pass
