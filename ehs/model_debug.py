#!/usr/bin/env python3
import yaml
import argparse
import struct
import numpy as np
import tensorflow as tf
import tfprocess


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
    return [encode_suit[card[1]], ranks.index(card[0]) + 1]


def encode_board(hand, board):
    def flatten(l):
        return [item for sublist in l for item in sublist]
    padding = [0 for _ in range(0, 5 - len(board))]
    return flatten([encode_card(card) for card in hand + board]) + padding

def hand_to_tensor(hand, board):

    encoded = encode_board(card_sort(split_cards(hand)), card_sort(split_cards(board)))
    packed = struct.pack(V6_STRUCT_STRING, struct.pack("b"*len(encoded), *encoded), 0.0)

    #print(packed)
    cards, value = parse_function(*convert_v6_to_tuple(packed))

    return cards



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
        #tf.print(input, summarize=16)
        #tf.print(res)
        return res[0][0].numpy()


    l = [
            ["Kh8d", "7s6sJhTdQc", 0.858961820602417],
            ["Th4h", "Jh8h3h6d8c", 0.9844734072685242],
            ["4sJc", "Qs9s3sTh5d", 0.8920246362686157],
            ["TsKh", "As6s6d4d2c", 0.9628521203994751],
            ["6d9c", "Qs5s2s7h3d", 0.662505567073822],
            ["5sQh", "6s4hQd5dKc", 0.8620002865791321],
            ["AsQh", "3sKhAd7dAc", 0.7919456958770752],
            ["Ts4s", "7s4dTc7c5c", 0.7360737919807434],
            ["6h9d", "9s4s2s9c5c", 0.8968966007232666],
            ["4hKc", "Qs8d7d3d2d", 0.4145245850086212],
            ["KsAh", "As5hKc5c2c", 0.9993021488189697],
            ["QsQc", "Ks6s5sTh8h", 0.8547195792198181],
            ["6h7d", "3sTd4dJc9c", 0.7881593704223633],
            ["6dQc", "Qs7sJh6h2d", 0.6692348718643188],
            ["Jh4c", "JsTh4hQdKc", 0.48631131649017334],
            ["9sQh", "8s2s8h7d7c", 0.6400206089019775],
            ["Jh3d", "7sKh7h3hTc", 0.9804498553276062],
            ["6c5c", "8s5s2hKcAc", 0.9995077252388],
            ["Js2c", "5s2sJd9d7d", 0.38180622458457947],
            ["6d5c", "Jh8h6h8d7d", 0.9945066571235657],
            ["6s2d", "8sKd9d5d3c", 0.9323699474334717],
            ["TsKh", "QsAs9sTd8c", 0.9974063634872437],
            ["5h8c", "6h6d5dKc9c", 0.9995768666267395],
            ["Kh5c", "3s2hTcQcKc", 0.9263896942138672],
            ["8d6c", "TsAs9hTd7c", 0.4953218698501587],
            ["Qs2h", "8s5sAh2d6c", 0.9438793063163757],
            ["Jh7h", "9sTh8h6hJc", 0.994195282459259],
            ["2s8c", "9h8h7d6d3c", 0.8029337525367737],
            ["Js4c", "5s6h4hTc5c", 0.6713663935661316],
            ["9d5c", "AsJhKd3d7c", 0.9924954175949097],
            ["8hQc", "Kh4hKdAd8c", 0.5994656085968018],
            ["3hAc", "4sAhQdJc7c", 0.7857797741889954],
            ["KhJc", "As7s9h8h6c", 0.88312828540802],
            ["8d3c", "Ks6h4d7c2c", 0.9639249444007874],
            ["AsAc", "5s4s2s2h4c", 0.9998379349708557],
            ["KdAd", "6sThJdQc7c", 0.9786974787712097],
            ["6c4c", "TdKd5d8c5c", 0.9635355472564697],
            ["2s4c", "Th5h8d6d9c", 0.8401994109153748],
            ["8s5h", "QhAh5d3d4c", 0.8746167421340942],
            ["Td5d", "Ts2s3h8dTc", 0.8495154976844788],
            ["Th9c", "TsJs7sKd7d", 0.5877373814582825],
            ["8h5h", "Js9h7h3hAd", 0.9964980483055115],
            ["ThTd", "As8sAd9d4d", 0.9975208044052124],
            ["4h8c", "Js2hJd6dQc", 0.7906825542449951],
            ["8s7s", "2sTh8d5d8c", 0.8190532326698303],
            ["9c6c", "5s3hJdJc7c", 0.9837952852249146],
            ["6s3s", "Ah7h3dQcJc", 0.7865113019943237],
            ["Th7c", "9s6d2dJc8c", 0.8593088388442993],
            ["KdAc", "5s6hTdJd8c", 0.967566967010498],
            ["KsQc", "As6sAd3c2c", 0.9475566744804382],
    ["Jh2c", "Qs6s2hKd4c", 0.693235456943512],
    ["Ac8c", "Ks2h8d4d2d", 0.9999322295188904],
    ["8h4h", "9s5sAh2hJd", 0.997076153755188],
    ["Js9h", "5h7d6dKc7c", 0.5637603998184204],
    ["7d5d", "KsJs2s9d3d", 0.9926197528839111],
    ["7h7d", "5sThAc9c3c", 0.5047258734703064],
    ["6dKc", "Qs6s3sTd5c", 0.928500235080719],
    ["6h5c", "Ts8s4hKd8c", 0.5772183537483215],
    ["As5h", "TsKs4d8c2c", 0.859008252620697],
    ["8s3d", "7s2sTh8c4c", 0.5662224292755127],
    ["5hQd", "5s9h7h8dQc", 0.8999106884002686],
    ["2s5c", "Js3s6hTc4c", 0.7186793088912964],
    ["JsAh", "9h7hJd8dAc", 0.9420137405395508],
    ["3sTc", "7s2sQc8c2c", 0.8789860606193542],
    ["AsJc", "6s5s7d2dTc", 0.8836846947669983],
    ["5sTh", "6sAh5h5d8c", 0.9030818939208984],
    ["9dAc", "QsKsTdQdJc", 0.9999743103981018],
    ["5hAc", "Js8h6h4hKc", 0.9568840861320496],
    ["5hTd", "9s7sJh9h8d", 0.9650157690048218],
    ["9h4d", "KsJh6h9dTc", 0.6754776239395142],
    ["3s9h", "7hQd6d6c4c", 0.45587238669395447],
    ["2h2c", "KsQd8d7d2d", 0.36472824215888977],
    ["6s5s", "JsJh9c7c2c", 0.46486783027648926],
    ["Ts4s", "7s4hQdJd9c", 0.968371570110321],
    ["4hQc", "As8sQh9c5c", 0.8439956307411194],
    ["Qc4c", "TsKh6h2h9d", 0.8791645169258118],
    ["AhKc", "Js7s4s6h9d", 0.9512664675712585],
    ["Ts5c", "2hTdAd7d4d", 0.5253696441650391],
    ["4sKc", "6s3sQh4hAd", 0.6141452789306641],
    ["Js6d", "4sAhTd9c8c", 0.8631868362426758],
    ["8s4d", "7d3dQc8c3c", 0.9991849064826965],
    ["4d5c", "Js9h4hQc2c", 0.7266932129859924],
    ["4dKc", "Js3s9h6hQc", 0.7938061356544495],
    ["Js8s", "Qs2hQdAd2d", 0.9834260940551758],
    ["Ad9c", "3h9d6dAc5c", 0.9929509162902832],
    ["Qh7c", "Qs5h4hJd4d", 0.9720094203948975],
    ["4s3h", "Qs7s2s9hJc", 0.9501143097877502],
    ["8h2d", "4s2hKd6d6c", 0.9893339276313782],
    ["JsKc", "KsTh2hAdTc", 0.9958246946334839],
    ["4h5d", "JsAh7h7d4d", 0.8694030046463013],
    ["Th9h", "7s5hTd8d2c", 0.6806188225746155],
    ["KsAd", "As3sTdJd2d", 0.9832942485809326],
    ["Kc6c", "9s8sQh8h4h", 0.3384437561035156],
    ["4d8c", "4sKh8h3c2c", 0.8809608221054077],
    ["5dTc", "KsJs9hKd2c", 0.47534945607185364],
    ["6s4c", "4s2h7c6c5c", 0.7670107483863831],
    ["7h6c", "9s6sJd8c2c", 0.9156476855278015],
    ["5s3s", "4sQh7hJd9c", 0.8004223108291626],
    ["9h3c", "JdAd4d3dJc", 0.5889617800712585],
    ["4s2h", "Qs9s6s8h2c", 0.9984365105628967],

]

    
    res = [predict(x[0], x[1]) - x[2] for x in l]
    res.sort()
    print(res)


main()
