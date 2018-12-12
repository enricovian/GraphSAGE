from collections import namedtuple

import graphsage.layers as layers

# ClassifierInfo is a namedtuple that specifies the architecture of a classifier layer
ClassifierInfo = namedtuple("ClassifierInfo",
    ['dim', # layer dimension
     'droput', # layer droput rate
     'act' # activation function
    ])

class EmbeddingsClassifier:
    """
    Classifier for embeddings.
    """

    def __init__(self, input_dim, num_classes, layers_info, **kwargs):
        '''
        Args:
            - input_dim: dimension of the input embeddings
            - num_classes: number of classes (dimension of the output).
            - layer_infos: List of ClassifierInfo namedtuples that describe the
                parameters of all the classifier layers.
        '''

        self.layers = []

        # Create hidden layers
        last_layer_dim = input_dim
        for dim, dropout, act in layers_info:
            layer = layers.Dense(last_layer_dim, dim,
                dropout=dropout,
                act=act)
            last_layer_dim = dim
            self.layers.append(layer)
        output_layer = layers.Dense(last_layer_dim, num_classes,
            act=lambda x : x)
        self.layers.append(output_layer)

    def predict(self, inputs):
        arg = inputs
        for layer in self.layers:
            arg = layer(arg)
        return arg

    def var_values(self):
        vars = []
        for layer in self.layers:
            vars.extend([var for var in layer.vars.values()])
        return vars

    def __call__(self, inputs):
        return self.predict(inputs)
