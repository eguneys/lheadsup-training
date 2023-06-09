import gzip
import os
import numpy as np
import json
import base64

def nested_getattr(obj, attr):
    attributes = attr.split(".")
    for a in attributes:
        obj = obj.get(a)
    return obj

def conv_layer_js():
    return {
            "weights": {},
            "bn_gammas": {},
            "bn_betas": {},
            "bn_means": {},
            "bn_stddivs": {}
            }

class JSNet:
    def __init__(self):
        self.js = {
                "weights": {
                    "input": conv_layer_js(),
                    "residual": [],
                    "hs": conv_layer_js(),
                    "ppot": conv_layer_js(),
                    "npot": conv_layer_js(),
                    "ip1_hs_w": {},
                    "ip1_hs_b": {},
                    "ip2_hs_w": {},
                    "ip2_hs_b": {},

                    "ip1_ppot_w": {},
                    "ip1_ppot_b": {},
                    "ip2_ppot_w": {},
                    "ip2_ppot_b": {},

                    "ip1_npot_w": {},
                    "ip1_npot_b": {},
                    "ip2_npot_w": {},
                    "ip2_npot_b": {},
                    }
                }
        self.weights = []

    def fill_layer_v2(self, layer, params):
        params = params.flatten().astype(np.float32)
        layer["min_val"] = 0 if len(params) == 1 else float(np.min(params))
        layer["max_val"] = 1 if len(params) == 1 and np.max(params) == 0 else float(np.max(params))

        if layer.get("max_val") == layer.get("min_val"):
            params = (params - layer.get("min_val"))
        else:
            params = (params - layer.get("min_val")) / (layer.get("max_val") - layer.get("min_val"))

        params *= 0xffff
        params = np.round(params)
        #print(params.astype(np.uint16)[0:50])
        #print(len(base64.b64encode(params.astype(np.uint16).tobytes()).decode('utf-8')))
        #print(base64.b64encode(params.astype(np.uint16).tobytes()).decode('utf-8'))
        layer["params"] = base64.b64encode(params.astype(np.uint16).tobytes()).decode('utf-8')

    def save_proto(self, filename):
        if len(filename.split('.')) == 1:
                filename += '.json.gz'

        with gzip.open(filename, 'wb') as f:
             data = json.dumps(self.js).encode('utf-8')
             f.write(data)
        size = os.path.getsize(filename) / 1024**2
        print("Weights saved as '{}' {}M".format(filename, round(size, 2)))


    def tf_name_to_pb_name(self, name):
        def convblock_to_bp(w):
            w = w.split(':')[0]
            d = {
                'kernel': 'weights',
                'gamma': 'bn_gammas',
                'beta': 'bn_betas',
                'moving_mean': 'bn_means',
                'moving_variance': 'bn_stddivs',
                'bias': 'biases'
            }
            return d[w]

        def se_to_bp(l, w):
            if l == 'dense1':
                n = 1
            elif l == 'dense2':
                n = 2
            else:
                raise ValueError('Unable to decode SE-weight {}/{}'.format(
                    l, w))
            w = w.split(':')[0]
            d = {'kernel': 'w', 'bias': 'b'}

            return d[w] + str(n)

        def value_to_bp(val, l, w):
            if l == 'dense1':
                n = 1
            elif l == 'dense2':
                n = 2
            else:
               raise ValueError('Unable to decode value weight {}/{}'.format(
                   l, w))
            w = w.split(':')[0]
            d = {'kernel': 'ip{}_{}_w', 'bias': 'ip{}_{}_b'}

            return d[w].format(n, val)

        layers = name.split('/')
        base_layer = layers[0]
        weights_name = layers[-1]
        pb_name = None
        block = None

        if base_layer == 'input':
            pb_name = 'input.' + convblock_to_bp(weights_name)
        elif base_layer == 'hs':
            if 'dense' in layers[1]:
                pb_name = value_to_bp('hs', layers[1], weights_name)
            else:
                pb_name = 'hs.' + convblock_to_bp(weights_name)
        elif base_layer == 'ppot':
            if 'dense' in layers[1]:
                pb_name = value_to_bp('ppot', layers[1], weights_name)
            else:
                pb_name = 'ppot.' + convblock_to_bp(weights_name)
        elif base_layer == 'npot':
            if 'dense' in layers[1]:
                pb_name = value_to_bp('npot', layers[1], weights_name)
            else:
                pb_name = 'npot.' + convblock_to_bp(weights_name)
        elif base_layer.startswith('residual'):
            block = int(base_layer.split('_')[1]) - 1
            if layers[1] == '1':
                 pb_name = 'conv1.' + convblock_to_bp(weights_name)
            elif layers[1] == '2':
                 pb_name = 'conv2.' + convblock_to_bp(weights_name)
            elif layers[1] == 'se':
                pb_name = 'se.' + se_to_bp(layers[-2], weights_name)
        return (pb_name, block)


    def fill_net_v2(self, all_weights):
        weight_names = [w[0] for w in all_weights]

        for name, weights in all_weights:
            layers = name.split('/')
            weights_name = layers[-1]

            if weights.ndim == 4:
                #[filter_height, filter_width, in_channels, out_channels]
                #[output, input, filter_size, filter_size]
                #print(name, weights.shape)
                weights = np.transpose(weights, axes=[3, 2, 0, 1])

            pb_name, block = self.tf_name_to_pb_name(name)

            if pb_name is None:
                raise ValueError(
                        "Dont know where to store weight in json: {}".format(
                            name))

            if block is None:
                pb_weights = self.js.get("weights")
            else:
                assert block >= 0
                while block >= len(self.js.get("weights").get("residual")):
                        self.js.get("weights").get("residual").append({
                            "conv1": conv_layer_js(),
                            "conv2": conv_layer_js(),
                            "se": { 
                                "w1": {},
                                "b1": {},
                                "w2": {},
                                "b2": {}
                                }
                            })
                pb_weights = self.js.get("weights").get("residual")[block]

            #if pb_name == 'input.weights': print(weights.shape)
            # print(pb_name, nested_getattr(pb_weights, pb_name))
            self.fill_layer_v2(nested_getattr(pb_weights, pb_name), weights)

            if pb_name.endswith('bn_betas'):
                gamma_name = name.replace('beta', 'gamma')
                if gamma_name in weight_names:
                    continue
                gamma = np.ones(weights.shape)
                pb_gamma = pb_name.replace('bn_betas', 'bn_gammas')
                self.fill_layer_v2(nested_getattr(pb_weights, pb_gamma), gamma)

