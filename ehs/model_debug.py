#!/usr/bin/env python3
import yaml
import argparse
import struct
import numpy as np
import tensorflow as tf
import tfprocess


import gzip
import itertools
import sys
import os


V6_STRUCT_STRING = '>14sf'
v6_struct = struct.Struct(V6_STRUCT_STRING)


flat_planes = []
for i in range(2):
   flat_planes.append(
		   (np.zeros(8, dtype=np.float32) + i).tobytes())

def convert_v6_to_tuple(content):
    (cards, value) = v6_struct.unpack(content)

    value = struct.pack('f', value)

    planes = np.unpackbits(np.frombuffer(cards, dtype=np.uint8)).astype(np.float32)

    planes = planes[0:4*8].tobytes() + \
            flat_planes[1] + \
            planes[4*8:].tobytes() + \
            flat_planes[1]

    assert len(planes) == ((7 * 2 + 1 + 1) * 8 * 4)

    return (planes, value)

def parse_function(cards, value):
    cards = tf.io.decode_raw(cards, tf.float32)
    value = tf.io.decode_raw(value, tf.float32)

    cards = tf.reshape(cards, (-1, 16, 8, 1))
    value = tf.reshape(value, (-1, 1))

    #tf.print(cards, summarize=16)

    cards = tf.transpose(cards, perm=[0, 2, 3, 1])

    return (cards, value)

def card_sort(cards):
    cards.sort(key=lambda card: 
            -1 * ord(card[1]) * 10000 - ord(card[0]))
    return cards

def split_cards(cards):
    return [cards[i:i+2] for i in range(0, len(cards), 2)]

ranks = "23456789TJQKA"
encode_suit = { 'h': 1, 's': 2, 'd': 4, 'c': 8 }

def encode_card(card):
    return [ranks.index(card[0]) + 1, encode_suit[card[1]]]


def encode_board(hand, board):
    def flatten(l):
        return [item for sublist in l for item in sublist]
    padding = [0 for _ in range(0, 5 - len(board))]
    return flatten([encode_card(card) for card in hand + board]) + padding

def hand_to_tensor(hand, board):

    encoded = encode_board(card_sort(split_cards(hand)), card_sort(split_cards(board)))
    packed = struct.pack(V6_STRUCT_STRING, struct.pack("b"*len(encoded), *encoded), 0.0)

    print(hand, packed)
    cards, value = parse_function(*convert_v6_to_tuple(packed))

    return cards

def packed_to_tensor(packed):
    return parse_function(*convert_v6_to_tuple(packed))

def single_file_gen(filename):
    try:
        record_size = v6_struct.size
        with gzip.open(filename, 'rb') as chunk_file:
            while True:
                chunkdata = chunk_file.read(256 * record_size)
                if len(chunkdata) == 0:
                    break
                for item in sample_record(chunkdata):
                    yield item

    except:
        print("failed to parse {}".format(filename))
        sys.exit(1)


def sample_record(chunkdata):
    record_size = v6_struct.size

    for i in range(0, len(chunkdata), record_size):
        record = chunkdata[i:i+record_size]
        print(record)
        yield record

def tensor_gen(gen):
    for r in gen:
        yield packed_to_tensor(r)

def parse_tensor_gen(filename):
    gen = single_file_gen(filename)
    gen = tensor_gen(gen)
    for t in gen:
        yield t

def main():
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

    def predict(hand, board):
        input = hand_to_tensor(hand, board)
        res = tfp.model(input)
        #print(hand, board)
        #tf.print(input, summarize=16)
        #tf.print(res)
        return res[0][0].numpy()


    def l_test():
        l = [
                ["JdKc", "7s9h5h3d8c", -0.04204002395272255],
                ["7c6c", "3sKd8d5d2d", -0.008527020923793316],
                ["5hKd", "3sQhJh9h4d", -0.14606991410255432],
                ["7s2c", "9sAh8h3dJc", -0.03928561136126518],
                ["7s6c", "QsKs5s9c3c", 0.11524167656898499],
                ["2s3h", "As7h5h9d8c", -0.005144748371094465],
                ["As2h", "Jd6dTc9c4c", 0.06501973420381546],
                ["Jc5c", "As6s3d7c2c", 0.1521560549736023],
                ["Ks2c", "3sJh9h4hAc", -0.16145703196525574],
                ["Ad3d", "9h2hQd7d5c", -0.008095151744782925],

                ]

        res = [abs(predict(x[0], x[1]) - x[2]) for x in l]
        res.sort()
        print(res)

    def file_test():
        gen = parse_tensor_gen('../lheadsup-play/data/data_high_sub/data_ehs_1.gz')
        s = []
        i = 0
        for t in gen:
            output = tfp.model(t[0])
            target = t[1]
            tf.print(t[0], output, target, summarize=16)
            diff = tf.abs(target - output)
            correct = tf.cast(tf.less_equal(diff, 0.09), tf.float32)
            s.append(correct + 0)
            i = i + 1
            if i == 8:
                sys.exit(0)

        print("{:g}".format(tf.reduce_mean(s)))




    #file_test()
    l_test()


main()

