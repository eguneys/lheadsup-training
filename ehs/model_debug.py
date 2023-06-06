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

def card_sort(cards):
    cards.sort(key=lambda card: 
            -1 * ord(card[1]) * 10000 - ord(card[0]))
    return cards

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
        tf.print(input, summarize=16)
        #tf.print(res)
        return res[0][0].numpy()


    l = [
            ["7c3h", "Ad4sTd8cQc", 0.17822183668613434],
            ["Jc2d", "8cQdTcKd3s", 0.31663402915000916],
            ["JhAc", "7c5cKs7s8d", 0.043862905353307724],
            ["2cKs", "Jd8cAd4hTc", 0.2686551511287689],
            ["Th9h", "6dQcAdKs3c", 0.3563416600227356],
            ["7sAd", "9d5cJh2s3d", 0.18815815448760986],
            ["JsTd", "2c8hJhAsKs", 0.2455063760280609],
            ["6h4c", "TdJh5sKd2d", 0.257347971200943],
            ["7d9s", "8dKc3h9dAd", 0.3475920855998993],
            ["Ks3h", "KhTdTc6cAd", 0.19897057116031647],
            ["AsTc", "KhAc7h5d5c", 0.516659677028656],
            ["Jd3c", "7c2d9h4dQh", 0.44877323508262634],
            ["2c8d", "KcJcQc6dQh", 0.3359892964363098],
            ["Ks2c", "Td8cQc6hKd", 0.4764872193336487],
            ["4s7s", "4hKdTh5d2h", 0.31102532148361206],
            ["Kc7c", "7h8d8c9s2h", 0.7237674593925476],
            ["8sKs", "Tc5cTd6cJc", 0.10817597806453705],
            ["Kd9c", "Qc9s2h5dKc", 0.7317783236503601],
            ["7dKs", "KcQc5c6dAd", -0.021044790744781494],
            ["Ks8s", "2s6c7c3c6d", 0.13142845034599304],
            ["5c3h", "6cQhQd3sQs", 0.24109022319316864],
            ["JhAh", "8s5dAcQh3c", 0.003936583176255226],
            ["4c8s", "6hTh6d8d2h", 0.5700782537460327],
            ["Qs2s", "9s3h2d3sKd", 0.5692641139030457],
            ["6cTh", "8cAh2c3h7h", 0.3937116265296936],
            ["Qs7c", "5d2hAh4c2c", 0.47814875841140747],
            ["8c3s", "Jh6c5cAs6h", 0.04153147712349892],
            ["6h5d", "Jh3sJc2d2s", 0.3369848430156708],
            ["Jd9h", "3h9dTdAhJc", 0.13731792569160461],
            ["6d8h", "9hQsQc2cKc", 0.07721675932407379],
            ["4hAs", "TcJc6sKh4d", 0.21099989116191864],
            ["8c7d", "7h8sTc7c4s", 0.08005809783935547],
            ["Td8s", "Js5d3c3s4h", 0.31353622674942017],
            ["6c9d", "AcTs3s7cTc", 0.4819313883781433],
            ["8dKd", "5d2cAd3sJc", 0.29313239455223083],
            ["5sAc", "Tc4s7sQs9s", 0.16605247557163239],
            ["7s6c", "Ah9s6d2cJh", 0.2659044861793518],
            ["Kd3h", "5s6d7s2s3c", 0.2579653263092041],
            ["8d4s", "Jd4d7hAc9d", 0.2067108154296875],
            ["9h2h", "JdKc9s2dTs", 0.2417181134223938],
            ["8dQh", "8h2dTdThAh", -0.032052114605903625],
            ["7c5h", "Jh3sJd5sTh", 0.35289716720581055],
            ["3hTc", "As6dAd7s4d", 0.03382231667637825],
            ["4d2s", "2d3h5s7h6c", 0.22996963560581207],
            ["5h7h", "7cJc5sQd9s", 0.3694147765636444],
            ["2sAh", "5h9d9c6cKh", 0.15920494496822357],
            ["2cJs", "Ah6s5c3s8s", 0.34257352352142334],
            ["2h4d", "Ts5hAc6hQd", 0.4309574365615845],
            ["Ks7c", "8h4sKd6sTh", 0.2720138728618622],
            ["2d2c", "7c4d5s3h7s", 0.5314376354217529],
    ["9h7c", "3dKh6dQc5c", 0.4490853250026703],
    ["Ad3s", "4h9sKd8hKs", 0.01909634657204151],
    ["Td7h", "8d6hQh8c3c", 0.1147388443350792],
    ["8c2s", "2hThJcAsQh", -0.09371336549520493],
    ["Ts3h", "4cQsQc7cAd", 0.1050073504447937],
    ["Qh8c", "3h9cQcAc8s", -0.018249206244945526],
    ["8c6d", "6sKdJsJh3c", 0.2957919239997864],
    ["Kd2d", "9cQdQc5d9h", 0.607348620891571],
    ["9s8h", "QcAh2cQd3d", 0.09782443195581436],
    ["3c9d", "As7h5c6dJs", 0.3484968841075897],
    ["As3h", "5hAcJhAhTs", 0.19165965914726257],
    ["9dAs", "6sAhKh2s6c", 0.380222350358963],
    ["9cJh", "8d5c8cQs7d", 0.29464051127433777],
    ["8d5s", "Kd7hAs9c2s", 0.01093309372663498],
    ["4c2c", "Td9c3cTcAc", 0.3889983892440796],
    ["5d6s", "QdAc4sTdQc", 0.18978668749332428],
    ["Jh4s", "4c2s5d5hTd", 0.23194870352745056],
    ["5s2h", "Qd6d9hAs5h", 0.432643324136734],
    ["8d4h", "7dTs5c6dJs", 0.089678555727005],
    ["Kh4c", "7d5h7s2hQd", 0.5242686867713928],
    ["8d5h", "QhKh4h6d7d", 0.006383649073541164],
    ["7c3c", "Ah4c8d2dAc", 0.5793015956878662],
    ["Tc5s", "TdQh7sQs4d", 0.06128932163119316],
    ["QcTd", "Jh8h5h9h9d", 0.1398816555738449],
    ["8h6h", "Ts5hKsJcQd", 0.499254435300827],
    ["9sJc", "JsQc2d3sJd", 0.3516646921634674],
    ["AdAh", "Td7dJhJcQd", 0.14092294871807098],
    ["6c8h", "9c4d5hJcAc", 0.33840134739875793],
    ["TcQd", "Qs6h4dThQh", 0.24892200529575348],
    ["4c4d", "2c7s5sQs6s", 0.18707279860973358],
    ["TcAd", "Ac9s7s2s3c", 0.6693133115768433],
    ["Qs7s", "AsKh3c5c6d", 0.13554427027702332],
    ["6s6c", "Js8cTd5cQd", 0.253030002117157],
    ["4dJh", "2c8dJcAcQd", 0.11656887084245682],
    ["2h4s", "Jd6s8d7s3d", 0.2587305009365082],
    ["3d7c", "TsTc5c6dJc", 0.3285207450389862],
    ["7s7c", "4s5s2cAh9c", 0.05500577390193939],
    ["8s7d", "5d2h4cKdJc", 0.4940786361694336],
    ["Qc9s", "6c9c5dQhAd", 0.11283495277166367],
    ["6dTd", "5hKh8s8h8d", -0.07884349673986435],
    ["4dKh", "3cQc3hTdTs", 0.44240015745162964],
    ["3dAd", "Ac5hKc2sQc", 0.38562098145484924],
    ["6c7c", "2dKs7s7d8c", 0.41446200013160706],
    ["4c8h", "7s6c3sThTd", 0.29398950934410095],
    ["Kh4c", "Th5d9sAdAh", 0.3441663384437561],
    ["Th3d", "2c6s4c8cKs", 0.2165112942457199],
    ["9s7d", "6cQd6d4dKh", 0.1754741221666336],
    ["QhKh", "QcJh7c8cTc", 0.24982278048992157],
    ["Jd9d", "Kc3c5d8c9h", 0.63258296251297],
    ["Qc2s", "2h9sKs6h2c", 0.16814033687114716],

]

    
    res = [abs(predict(x[0], x[1]) - x[2]) for x in l]
    res.sort()
    print(res)


main()
