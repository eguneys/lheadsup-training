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
["3c9d", "3d5c9s6h4c", 0.9482905864715576],
["Ac4c", "2d5d9sJd8c", 0.9792873859405518],
["9cKh", "8c3cTd7h8h", 0.20731917023658752],
["4d3s", "3hKd5dTc7s", 0.9038909077644348],
["6s2s", "7s5cAh3d6d", 0.6734130382537842],
["5h7d", "TsKdAhAc3d", 0.5361753702163696],
["Kc7c", "6cJh2hThQd", 0.3651759922504425],
["6cAs", "AhKc7s9d8d", 0.8138650059700012],
["7c3s", "4cKc3d8c5d", 0.7769248485565186],
["Ad8h", "TsKhTd7sJs", 0.7614416480064392],
["8s7s", "Qs8h6c4c5s", 0.7884080410003662],
["Jd6s", "5d5s5c7c3d", 0.731752336025238],
["3c4c", "3hTcQh4h7d", 0.922346830368042],
["Qh2d", "6sAs4cQd7c", 0.7738164067268372],
["Js9s", "As9dTcJc9h", 0.8893669247627258],
["8s6d", "8d2dJd8h6s", 0.8278462290763855],
["Qh3h", "Td4d4s7hKc", 0.6780411005020142],
["Ad8c", "6cKs2c3cJh", 0.9832432270050049],
["7d2d", "KhJd7h8dQd", 0.5198773145675659],
["3h7s", "JhQdAs8hKd", 0.6174798011779785],
["Td8d", "7hJs6h5hTs", 0.7703465819358826],
["2s4c", "QdAdAs7h9s", 0.5509493350982666],
["Kc5c", "Kd3d7hAdQh", 0.8154102563858032],
["2s5h", "Kc5sKh9hAh", 0.600292444229126],
["4d8c", "Ks5dTsQh9s", 0.5077160596847534],
["5cTc", "2c5s9cAc5d", 0.9977666139602661],
["Jc2d", "Qc6h8d6s2c", 0.8398181796073914],
["5h3h", "4hAd5cAsQc", 0.4937385320663452],
["9hJd", "Th8s5dJs9s", 0.8780717253684998],
["8d4h", "2s8h2c5c4d", 0.7657917737960815],
["Ah5s", "7s3c9h6h8c", 0.8107879161834717],
["Qd6c", "As4hAc4sKc", 0.774526059627533],
["3c2h", "7h9c8d7sTc", 0.9071835279464722],
["Kh8h", "9c3c7d2s6d", 0.5285794734954834],
["Ah9s", "Kh9dJcAdQs", 0.8893983364105225],
["3d2s", "3h8c5s6s7s", 0.5917932987213135],
["9h4h", "Jc3cQd3h5d", 0.6337248086929321],
["8hQc", "6d4sTs3h4d", 0.8779124617576599],
["QhQs", "3sQc5sJc4c", 0.8485614061355591],
["2cQh", "7dKhQd7hAd", 0.2929251790046692],
["7c6h", "Ks2s4dQd8h", 0.5580477118492126],
["3c3h", "5c6dJh4h6h", 0.9559637904167175],
["7d4c", "Ac2s4d3s7s", 0.841403603553772],
["2h3h", "8cKsThAcQc", 0.5696051120758057],
["Kd7h", "5c6c9h3d7d", 0.8925925493240356],
["7s9s", "2s3h2d3cAc", 0.8856501579284668],
["Jd8c", "5dQh7sKh9s", 0.42617765069007874],
["6s8d", "2dJdAc3s7d", 0.9798057079315186],
["JhTd", "3d7h9dQd3s", 0.7662492990493774],
["5c4h", "AdKcTs6sQs", 0.5434174537658691],
["4hAc", "AsQs9c2d6s", 0.8736192584037781],
["Ah4s", "4c9sJd5c8h", 0.7332639098167419],
["6dJd", "2c6h2sQd7h", 0.8089471459388733],
["4s7h", "3s9cTh5d4d", 0.7410290241241455],
["8h8d", "6cQc2s2cTs", 0.9777868986129761],
["5sKs", "7hAd2c9d6s", 0.9360712766647339],
["7h3d", "Js2sKs8h8d", 0.6523972153663635],
["3hQc", "As5c2c7c8c", 0.7286980748176575],
["Qc5c", "ThAd9c3c4s", 0.9847815632820129],
["Ts5d", "4s9h4dKs2h", 0.6821996569633484],
["5h4h", "7cJcAhTd8d", 0.5614148378372192],
["JsKd", "9d2dAdTs2s", 0.7806717157363892],
["8dQh", "7d5d5h3dAc", 0.7660271525382996],
["7c2d", "Kc5s6d4c7s", 0.8210400938987732],
["6cJc", "AdKdTc4c4h", 0.8492482304573059],
["3dTh", "8h9c3c5s7d", 0.7934250235557556],
["3s8d", "7c3h2cAc6c", 0.22594094276428223],
["6sQc", "5d6dJh4c4h", 0.7670207023620605],
["9h5s", "Kd6d8c9s3s", 0.9459723830223083],
["3h5d", "9s5cQh4cAd", 0.7429019212722778],
["9dJs", "KsTh4cAcKd", 0.8456594944000244],
["Qh8c", "Qc9dJs8s2s", 0.6920142769813538],
["4h8h", "Js3c4sAh2c", 0.8146792650222778],
["3h5c", "AdQdKsJs9h", 0.7499369382858276],
["AcKh", "4dAh8d9sQs", 0.7425184845924377],
["7h9s", "Th3d2d7c8h", 0.6703929901123047],
["Jc5d", "7dJd9cQdJs", 0.7712424397468567],
["Jd7h", "5c2cAd9d4s", 0.7108184695243835],
["8c2c", "3s9c2d6s7s", 0.9616775512695312],
["7dQc", "Qs8dJs4dJd", 0.9351701736450195],
["3hAd", "5c8cKs2hJd", 0.6713632345199585],
["6h7d", "JcTd6c3hAs", 0.7525256872177124],
["Qs2s", "Th5hJcQc6h", 0.570355236530304],
["3c5h", "AhQh3h6cKs", 0.9859346747398376],
["9d5h", "3h9s8c9hTd", 0.6993809342384338],
["Kd3h", "AdJdQh6dTd", 0.5082124471664429],
["4cQc", "7h6dTh2h3h", 0.8144569993019104],
["ThJs", "Kd9s7cQhAc", 0.8582838177680969],
["Js3d", "Jc9dKd3h2s", 0.9075754880905151],
["7h3s", "Qc4s6c2h7c", 0.702400803565979],
["KdAd", "5dQdJd7dAs", 0.8645042777061462],
["4h9c", "TcJhQcJc3d", 0.810913622379303],
["Kh8c", "9hAh4cKc6d", 0.874021053314209],
["9hKd", "9sTc6d5c2c", 0.7416812181472778],
["TsTd", "Kh2h3dQc3c", 0.5572909116744995],
["TsTd", "5dJsAdThKd", 0.8981838822364807],
["3h5h", "8s5sJh2hTc", 0.8951568603515625],
["Qd4d", "9h8h9c2s7d", 0.9367459416389465],
["Ts2s", "8c2dKs7hAd", 0.7249531149864197],
["5d8c", "4d2cQcKsAh", 0.7299617528915405],

]

    
    res = [predict(x[0], x[1]) - x[2] for x in l]
    res.sort()
    print(res)


main()
