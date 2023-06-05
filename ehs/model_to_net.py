#!/usr/bin/env python3
import argparse
import os
import yaml
import tfprocess

argparser = argparse.ArgumentParser(description='Convert model to net.')
argparser.add_argument('--cfg',
        type=argparse.FileType('r'),
        help='yaml configuration with training parameters')
args = argparser.parse_args()
cfg = yaml.safe_load(args.cfg.read())
print(yaml.dump(cfg, default_flow_style=False))

tfp = tfprocess.TFProcess(cfg)
tfp.init_net()

tfp.restore()

import tensorflow as tf
import numpy as np

zeros = [np.zeros(16)]
ones = [np.ones(16)]
halves = [np.ones(16) * 0.5]

input = tf.constant([[
    zeros,
    zeros,
    zeros,
    zeros,
    zeros,
    zeros,
    zeros,
    zeros]])

# Ah 2h  3c 4c 5c 6c 7c
xinput = tf.constant([[
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    [[0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]],
    [[0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1]],
    [[1, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1]]]])

# ['2h', '3c'], ['Qc', 'Jh', '7d', '3s', '4d']
input = tf.constant([[
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]],
    [[0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1]],
    [[0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1]],
    [[0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1]],
    [[1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1]]]])

res = tfp.model(input)
print(input)
tf.print(res)

root_dir = os.path.join(cfg['training']['path'], cfg['name'])
if not os.path.exists(root_dir):
    os.makedirs(root_dir)
path = os.path.join(tfp.root_dir, tfp.cfg['name'])
steps = tfp.global_step.read_value().numpy()
leela_path = path + '-' + str(steps)
tfp.save_leelaz_weights(leela_path)
