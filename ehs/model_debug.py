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
    padding = [0 for _ in range(0, (5 - len(board)) * 2)]
    return flatten([encode_card(card) for card in hand + board]) + padding

def hand_to_tensor(hand, board):

    encoded = encode_board(card_sort(split_cards(hand)), card_sort(split_cards(board)))
    packed = struct.pack(V6_STRUCT_STRING, struct.pack("b"*len(encoded), *encoded), 0.0)

    print(encoded,packed)

    cards, value = parse_function(*convert_v6_to_tuple(packed))

    print(cards, value)

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
        #print(record)
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
                ["Tc3d", "4s3s6c", 0.62],
                ["Ah5d", "TcJc2c", 0.48],
                ["QdJc", "3c6d8c", 0.48],
                ["9h5d", "6d7sKd", 0.28],
                ["6cTs", "Th2cQc", 0.76],
                ["2s2d", "4c6c3s", 0.56],
                ["Qs3d", "2h8c8d", 0.36],
                ["Qs9s", "5d3cAd", 0.36],
                ["QcQh", "6d2cQd", 0.96],
                ["8h3d", "4c4hJd", 0.2],
                ["3cQs", "7c8dTh", 0.46],
                ["9sAh", "Ts8d3h", 0.56],
                ["ThJh", "9c4s8s", 0.66],
                ["5c8d", "5s4dAs", 0.62],
                ["Ac5d", "3s6h6d", 0.56],
                ["Ts2s", "8sQsJs", 0.92],
                ["2c8s", "TsAs3h", 0.32],
                ["Ks3c", "Qh7sKh", 0.78],
                ["2c2d", "8s4c6s", 0.46],
                ["8cKc", "KsJd5c", 0.88],
                ["9s6c", "Kh5s9d", 0.74],
                ["9h4h", "4s7sKd", 0.5],
                ["2d3s", "As8sTc", 0.3],
                ["Kc3h", "KdTdQs", 0.78],
                ["Ks9s", "Kd6sQd", 0.8],
                ["AhAd", "ThAc8h", 0.96],
                ["AdJh", "Ac4dKh", 0.86],
                ["AsKc", "6s6d2d", 0.64],
                ["Kd4c", "AhKhJd", 0.74],
                ["7sJh", "9d7c5h", 0.66],
                ["AsKc", "4c8cQd", 0.64],
                ["3c3s", "8d4sKs", 0.42],
                ["6d5h", "5d2d2c", 0.64],
                ["8dAs", "2h6dJd", 0.48],
                ["Ah9s", "TdJc5s", 0.46],
                ["Ts3s", "3cKd4s", 0.64],
                ["2hKh", "AcJhAh", 0.62],
                ["2dKc", "QsKd9s", 0.8],
                ["JdQh", "KsQsKd", 0.84],
                ["7s3h", "7c2d5d", 0.68],
                ["KdTh", "9d6dTs", 0.7],
                ["Ac8d", "KsQh6h", 0.52],
                ["QcAh", "Tc4d3d", 0.58],
                ["Qs8c", "TdJcTc", 0.62],
                ["6s2s", "Qc7cJd", 0.18],
                ["7d7c", "AsThJs", 0.46],
                ["4h2c", "Qc3s9h", 0.16],
                ["Tc3c", "6cKs5d", 0.42],
                ["2s3d", "As8cTs", 0.24],
                ["Jc3s", "4hJsKc", 0.78],
        ["Jh3c", "Ts2h6s", 0.16],
        ["2s9h", "6s8hQh", 0.28],
        ["TsTc", "Ks4d8h", 0.72],
        ["6c8c", "TsJcTc", 0.66],
        ["Td5s", "5c8hTs", 0.94],
        ["2hKh", "Th3cAh", 0.72],
                ]

        l = [
                ["Tc3d", "4s3s6c", 0.62],
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

