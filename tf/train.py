#!/usr/bin/env python3

import argparse
import yaml
import random
import glob
import sys
import os
from chunkparser import ChunkParser

def get_chunks(data_prefix):
    return glob.glob(data_prefix + '*.gz')


def get_all_chunks(path):
    chunks = []
    for d in glob.glob(path):
        chunks += get_chunks(d)
    print("got", len(chunks), "chunks for", path)
    return chunks

def get_latest_chunks(path, num_chunks):
    chunks = get_all_chunks(path)
    if len(chunks) < num_chunks:
        print("Not enough chunks {}".format(len(chunks)))
        sys.exit(1)

    chunks = chunks[:num_chunks]
    print("{} - {}".format(os.path.basename(chunks[-1]),
        os.path.basename(chunks[0])))

    random.shuffle(chunks)
    return chunks


def main(cmd):
    cfg = yaml.safe_load(cmd.cfg.read())
    print(yaml.dump(cfg, default_flow_style=False))

    num_chunks = cfg['dataset']['num_chunks']
    train_ratio = cfg['dataset']['train_ratio']
    num_train = int(num_chunks * train_ratio)

    chunks = get_latest_chunks(cfg['dataset']['input'], num_chunks)
    train_chunks = chunks[:num_train]
    test_chunks = chunks[num_train:]


    total_batch_size = cfg['training']['batch_size']
    batch_splits = cfg['training'].get('num_batch_splits', 1)
    if total_batch_size % batch_splits != 0:
        raise ValueError('num_batch_splits must divide batch_size evenly')
    split_batch_size = total_batch_size // batch_splits

    train_parser = ChunkParser(train_chunks)

    test_parser = ChunkParser(test_chunks)

    import tensorflow as tf

    from chunkparsefunc import parse_function
    from tfprocess import TFProcess
    tfprocess = TFProcess(cfg)

    train_dataset = tf.data.Dataset.from_generator(
            train_parser.parse,
            output_types=(tf.string, tf.string, tf.string))
    train_dataset = train_dataset.map(parse_function)
    test_dataset = tf.data.Dataset.from_generator(
            test_parser.parse,
            output_types=(tf.string, tf.string, tf.string))
    test_dataset = test_dataset.map(parse_function)

    train_dataset = train_dataset.prefetch(4)
    test_dataset = test_dataset.prefetch(4)

    tfprocess.init(train_dataset, test_dataset)

    tfprocess.restore()

    num_evals = cfg['training'].get('num_test_positions',
            len(test_chunks) * 10)
    num_evals = max(1, num_evals // split_batch_size)
    print("Using {} evaluation batches".format(num_evals))
    tfprocess.total_batch_size = total_batch_size
    tfprocess.process_loop(total_batch_size, num_evals)

    train_parser.shutdown()
    test_parser.shutdown()



if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description=\
            'Tensorflow pipeline for training lheadsup')
    argparser.add_argument('--cfg',
            type = argparse.FileType('r'),
            help='yaml configuration with training parameters')
    main(argparser.parse_args())
