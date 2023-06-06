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

    encoded = encode_board(split_cards(hand), split_cards(board))
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


    l = [["AcQh", "Th5c8d8hQc", 0.9052196741104126],
		["3sJh", "3c7h2c6c2h", 0.9281209707260132],
		["4c4s", "Qh8h3c8c6d", 0.960192859172821],
		["9dQh", "JhTsQs7dTc", 0.8394347429275513],
		["8dJd", "5h2sQc6h5s", 0.6399486660957336],
		["Js3c", "3s8h9s8dQs", 0.7730302810668945],
		["9hJh", "5d8d9s7sQh", 0.4931734800338745],
		["9s3h", "8c8h2sQdAs", 0.41168031096458435],
		["7dQc", "6sKd7hAcTh", 0.5314173698425293],
		["2h3h", "Ah4cQh8h5h", 0.9877171516418457]]

    l = [
["2dAc", "9c3h3sKd2c", 0.9654282927513123],
["2cQd", "6h3s4c7s8d", 0.7546229958534241],
["Ks8s", "5c8h3s9h7h", 0.9312437176704407],
["3dAc", "5s2hKh7cJc", 0.7372666001319885],
["JsTh", "6h9h3d2h9s", 0.9845695495605469],
["9dJd", "3sKh9sJh6h", 0.9193941354751587],
["5s2h", "2c7c7s9c9h", 0.80277019739151],
["TdQc", "8dKh8h2d6d", 0.7814067006111145],
["8c4h", "QdTcKcAd7h", 0.7493577599525452],
["6hTs", "6d7sJs7h6s", 0.5927615165710449],
["6sJh", "Td4cAh8hTc", 0.8127343654632568],
["9s3c", "QcAcKc3h7h", 0.6522048115730286],
["Ac3c", "TcQs2sKh9c", 0.7716678380966187],
["5h2d", "Ad9dQhTc4s", 0.5643678307533264],
["6d2s", "Td9h6c8s2h", 0.8653018474578857],
["4c3s", "9cThKsTdKh", 0.7831098437309265],
["4hJc", "5d8d3cKh2h", 0.9762787818908691],
["KhTs", "Th4dKd5c7h", 0.9388298988342285],
["3s2s", "Kd6dKs9h4s", 0.7251231074333191],
["As8s", "AdJdTd8dQd", 0.8615777492523193],
["8sQc", "2hQdKdJcAs", 0.8797057271003723],
["3c8s", "QdJsJd3hQh", 0.8938112258911133],
["6c8h", "JsTcQc9hKd", 0.7588903307914734],
["9h2s", "TsKs9c7h8c", 0.7335706353187561],
["2d5c", "3s6c4c3c7s", 0.7534487843513489],
["6sQc", "AcAd2d5dQh", 0.8501235842704773],
["6c7d", "Ah6h2c2s5h", 0.835423469543457],
["8cTd", "As3h7s7h9c", 0.7841051816940308],
["KsTh", "Jc2s4dQdQs", 0.6086835861206055],
["6d7d", "3s8d9d4sTc", 0.5817752480506897],
["Ac7c", "6d4cJd9dTs", 0.7762334942817688],
["7dQs", "Ks7c3dJcAd", 0.6390220522880554],
["Ah8c", "9dKs6h8hKd", 0.8116360306739807],
["4hAh", "Ac3s7d9hKc", 0.7009153366088867],
["8h3h", "TcTh7h9c8s", 0.6334728002548218],
["7hTh", "8d9s2h7d2s", 0.8102671504020691],
["4sAs", "AdKs5cQs8h", 0.6799364686012268],
["QcQh", "9hKhTc8sTh", 0.8704478740692139],
["Tc8s", "6sJh8hKh4s", 0.5963398218154907],
["Th7d", "8sAd6d8d7c", 0.838892936706543],
["9sAd", "Ac2d5d7d2c", 0.87807297706604],
["5s6h", "3c3d6s8h4d", 0.3366546332836151],
["5d7d", "6d7hJd2dTc", 0.9946475028991699],
["Jd3h", "2s7d4c6sJs", 0.6890265941619873],
["2h3s", "7sKh2c7c5d", 0.5027307868003845],
["7cKd", "6dKcQc2c5d", 0.8083106279373169],
["Ah3h", "3sJh6d2sTc", 0.686552882194519],
["7h6s", "Js3h6c8s9h", 0.9079641103744507],
["Ad8c", "Js7sKd5cQd", 0.7465837001800537],
["KsTs", "Th4d4h5c7d", 0.35416942834854126],
["3sKs", "7d2sAhTc6s", 0.7644678950309753],
["7s3h", "8d5c4h9s7h", 0.8257814049720764],
["7dQs", "TdTs5dTh2c", 0.7828158736228943],
["8s9s", "TsTc6hAs7s", 0.5441810488700867],
["QhQs", "Jh8c8h4s5s", 0.7943642735481262],
["2s8h", "JhJdJc2d7d", 0.5434560179710388],
["2d5h", "4s6sAc8cKc", 0.6682083010673523],
["7cJs", "As2h2dQh4s", 0.4186134934425354],
["Qc8s", "Tc3hAhJhAd", 0.7242556810379028],
["TcAd", "5hTd6d4s2s", 0.9514563679695129],
["7s3c", "8sAhAs5s6d", 0.7370384931564331],
["5hQc", "9cAdTd4hAs", 0.7604270577430725],
["Jc9d", "Td6sKsTs4h", 0.7317849397659302],
["JsQh", "9cJh9h9d4c", 0.8769485950469971],
["4dAd", "4c6s2h9c8c", 0.8700742125511169],
["Ks4d", "5s4h3d4c9h", 0.6995712518692017],
["6c7d", "3h8d8h6d9h", 0.6576694846153259],
["6s8c", "Js4d8s3d2h", 0.6937681436538696],
["5cJs", "2dQh9sAs9d", 0.8899943828582764],
["2sKh", "8c7s9d6s5h", 0.6510369777679443],
["6hJh", "4s9hQhQd4c", 0.6528394818305969],
["Ah8h", "4dJcJh4c9h", 0.6582512855529785],
["7cTh", "5d5sTc2s9h", 0.7943534851074219],
["9sAh", "Qc8h2sQd4d", 0.8561148047447205],
["Qs7d", "4d2sAc2d9s", 0.6381699442863464],
["7hAh", "Kd2sQc9sTs", 0.6786957383155823],
["5s4d", "2h7h5c7cKd", 0.7035903334617615],
["2h8s", "QsQh5cAh9d", 0.7874934077262878],
["KdAc", "2s4s3sQs7s", 0.7664987444877625],
["4c6s", "Ts2d2c7dQc", 0.9568412899971008],
["Jc4d", "4hQh4sTh3h", 0.9341971278190613],
["2c9c", "JsQh6c8d7h", 0.40714946389198303],
["5dQs", "3dTc5cTs8s", 0.8006684184074402],
["TdQs", "Jh3d7c8s4s", 0.7436758875846863],
["9hQd", "2c4cJdAd3s", 0.47867172956466675],
["8c8h", "Qh4d5dKc8d", 0.7861694693565369],
["6dAc", "KhQd5hKd9d", 0.7725724577903748],
["9dAc", "ThJh5d4hTc", 0.9227815866470337],
["Ks8h", "5dJd8sAdKh", 0.7541289329528809],
["7c5s", "4c4d8d2hTh", 0.42777708172798157],
["8d4c", "8h3hKs5hTh", 0.7365147471427917],
["9s5d", "Ah8s8cJs3s", 0.3988688290119171],
["Kh7s", "5d9s6c3s2d", 0.9546423554420471],
["8cAc", "Jd4c9d3d3s", 0.7732993960380554],
["2hAd", "3dKsJs8sTd", 0.7136526107788086],
["8c2h", "8hTh2dQhAd", 0.9414483308792114],
["5h3d", "JhTsAc3cKh", 0.5187946557998657],
["2sQh", "2c4sThTs3c", 0.9412458539009094],
["KcTs", "AdKsAs6d7d", 0.9077715277671814],
["Ac4c", "Qh7c9d8cKd", 0.4212833344936371],

]
    
    res = [predict(x[0], x[1]) - x[2] for x in l]
    res.sort()
    print(res)


main()
