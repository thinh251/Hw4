import os


def load_layers_definition(network_description):
    """Return a dictionary of layers, with the key is the index of the layer
        value is an array [feature_size, num_feature]
    """
    if os.path.isfile(network_description):
        layers = dict()
        with open(network_description) as nd_file:
            layer_index = 1
            for line in nd_file:
                fn = map(int, line.rstrip('\n').split(" "))
                layers[layer_index] = fn
                layer_index += 1
        return layers
    else:
        IOError('Network description file does not exist')


test_layer = load_layers_definition('network_description')
print test_layer
