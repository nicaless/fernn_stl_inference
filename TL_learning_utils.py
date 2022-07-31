import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K


class TLPass(tf.keras.layers.Layer):
    """
    Simple Pass Through Layer
    Helpful for structuring larger networks
    """
    def __init__(self, units, mult=1, **kwargs):
        self.units = units
        self.state_size = units
        self.mult = mult
        super(TLPass, self).__init__(**kwargs)

    def build(self, input_shape):
        self.build = True

    def call(self, inputs, states):
        outputs = inputs * self.mult
        return outputs, [outputs]


class TLWeightTransform(tf.keras.layers.Layer):
    """
    For Weighting inputs and/or transforming inputs to other TL layers
    """
    def __init__(self, units, w=None, transform=None, quantize=False, **kwargs):
        """
        :param units: Usually 1, no need to save memory for this
        :param w: if you want to set an initial weight.
            If not setting a weight, typically you should set quantize=True
        :param transform: if not none, weights reduced to a single value
        (ideally after quantization) to pass to a layer that will only take in single input
        :param quantize: used for setting which items are 'chosen' for the next layer
        :param kwargs:
        """
        self.units = units
        self.state_size = units
        self.given_w = w
        self.transform = transform
        self.quantize = quantize
        super(TLWeightTransform, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.given_w is not None:
            self.w = tf.convert_to_tensor(
                np.ones((input_shape[-1])) * self.given_w,
                dtype=tf.float32)
        else:
            self.w = self.add_weight(shape=(input_shape[-1],),
                                     initializer='random_normal',
                                     name='input_weights')

        self.full_weights = K.get_value(self.w).copy()

        if self.quantize:
            if self.quantize == 'one-hot':
                max_weight = tf.reduce_max(self.full_weights, 0, keepdims=True)
                arg_max_weight = tf.argmax(self.full_weights)
                weight = tf.one_hot(arg_max_weight,
                                    depth=self.full_weights.shape[0])
                alpha = max_weight
                # alpha = 1
            else:
                weight = tf.math.sign(self.full_weights)
                alpha = tf.norm(self.full_weights,
                                ord=1) / self.full_weights.shape[0]

            quant_weights = weight * alpha
            self.last_quant_weights = tf.identity(quant_weights)

            if self.given_w is None:
                K.set_value(self.w, quant_weights)
            else:
                self.w = quant_weights

        self.built = True

    def get_config(self):
        config = super(TLWeightTransform, self).get_config()
        config.update({'units': self.units,
                       'w': self.w.numpy(), 'quantize': self.quantize,
                       'transform': self.transform})
        return config

    def call(self, inputs):
        weight = self.w
        # if not self.quantize:
        #     weight = self.w
        #     alpha = 1
        # else:
        #     if self.quantize == 'one-hot':
        #         max_weight = tf.reduce_max(self.w, 0, keepdims=True)
        #         arg_max_weight = tf.argmax(self.w)
        #         weight = tf.one_hot(arg_max_weight, depth=self.w.shape[0])
        #
        #         alpha = max_weight
        #
        #         # weight = tf.reduce_max(self.w, 0, keepdims=True)
        #         # alpha = tf.norm(self.w, ord=1)
        #     else:
        #         # do rastegari method
        #         weight = tf.math.sign(self.w)
        #         alpha = tf.norm(self.w, ord=1) / self.w.shape[0]
        #
        # weighted_inputs = tf.math.multiply(inputs, weight) * alpha
        weighted_inputs = tf.math.multiply(inputs, weight)

        if self.transform is not None:
            weighted_inputs = tf.reduce_sum(weighted_inputs, 2, keepdims=True)

        return weighted_inputs

    def on_batch_end(self):
        if self.quantize:
            new_weights = K.get_value(self.w)
            weights_update = new_weights - self.last_quant_weights
            self.full_weights += weights_update

            if self.quantize == 'one-hot':
                max_weight = tf.reduce_max(self.full_weights, 0, keepdims=True)
                arg_max_weight = tf.argmax(self.full_weights)
                weight = tf.one_hot(arg_max_weight,
                                    depth=self.full_weights.shape[0])
                alpha = max_weight
                # alpha = 1
            else:
                weight = tf.math.sign(self.full_weights)
                alpha = tf.norm(self.full_weights,
                                ord=1) / self.full_weights.shape[0]

            self.last_quant_weights = weight * alpha

            if self.given_w is None:
                K.set_value(self.w, tf.identity(self.last_quant_weights))
            else:
                self.w = tf.identity(self.last_quant_weights)


def pos_robustness(y_true, y_pred):
    sign_diff = tf.subtract(tf.cast(y_pred, dtype='float32'),
                            tf.cast(y_true, dtype='float32'))
    loss = tf.reduce_mean(tf.math.abs(sign_diff))
    return loss


def extract_weights(layer_names, layer_weights, choice_config,
                    quant_method='one-hot', normalizer=None):
    result = {}
    for name, weights in zip(layer_names, layer_weights):
        if 'choice' in name:
            layer_name = name.split('/')[0]
            conf = choice_config[layer_name]
            if quant_method == 'one-hot':
                true_weights = [np.round(np.float64(w), 3) for w in weights]
                result[layer_name] = {'choice': conf[int(np.argmax(weights))],
                                      'weight': np.float64(np.max(weights)),
                                      'true_weights': true_weights}
            else:
                # assumes quant using rastegari
                true_weights = [np.round(np.float64(w), 3) for w in weights]
                pos_weights = [w for w in weights if w > 0]
                pos_weights_index = [i for i in range(len(weights))
                                     if weights[i] > 0]
                result[layer_name] = {'choices': [conf[i]
                                            for i in pos_weights_index],
                                      'weights': np.float64(pos_weights),
                                      'true_weights': true_weights}
        elif 'dense' in name:
            continue
        else:
            if 'atom' in name:
                layer_name = name.split('/')[0]
                if layer_name not in result.keys():
                    result[layer_name] = {'weight': 1, 'bias': 0}

                if 'weight' in name:
                    result[layer_name]['weight'] = np.float64(weights[0][0])
                else:
                    result[layer_name]['bias'] = np.float64(weights[0])

                if normalizer is not None:
                    if layer_name in normalizer.keys():
                        c = -1 * result[layer_name]['bias'] / \
                            result[layer_name]['weight']
                        inq = 'geq' if \
                            np.sign(result[layer_name]['weight']) >= 0 \
                            else 'leq'
                        result[layer_name]['c'] = c * normalizer[layer_name]
                        result[layer_name]['inq'] = inq
            elif 'interval_weights' in name:
                layer_name = name.split('/')[0] + '_interval'
                result[layer_name] = [np.round(np.float64(w), 3)
                                      for w in weights]
            else:
                result[name] = weights
    return result





