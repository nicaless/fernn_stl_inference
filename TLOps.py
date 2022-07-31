import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import RNN
from tensorflow.keras.models import Model, Sequential


class TLAtom(tf.keras.layers.Layer):
    def __init__(self, units, w=None, b=None, **kwargs):
        self.units = units
        self.state_size = units
        self.given_w = w
        self.given_b = b
        super(TLAtom, self).__init__(**kwargs)

    def build(self, input_shape):
        # TODO: fix to make multi-cell cases possible
        if self.given_w is not None:
            if isinstance(self.given_w, list):
                w = np.reshape(self.given_w, (input_shape[-1], self.units))
                self.w = tf.convert_to_tensor(w, dtype=tf.float32)
            else:
                self.w = tf.convert_to_tensor(
                    np.ones((input_shape[-1], self.units)) * self.given_w,
                    dtype=tf.float32)
        else:
            self.w = self.add_weight(shape=(input_shape[-1], 1),
                                     initializer='random_normal',
                                     name='weight')
        if self.given_b is not None:
            self.b = self.given_b
        else:
            self.b = self.add_weight(shape=(self.units,),
                                     initializer='random_normal',
                                     name='bias')
        self.built = True

    def call(self, inputs, states):
        h = K.dot(inputs, self.w) + self.b
        output = h
        return output, [output]

    def get_config(self):
        config = super(TLAtom, self).get_config()
        config.update({'units': self.units, 'w': self.w.numpy(),
                       'b': self.b if isinstance(self.b, int) else self.b.numpy()})
        return config


class TLNot(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        self.output_size = 1
        super(TLNot, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        output = -1 * inputs
        return output, [output]

    def get_config(self):
        config = super(TLNot, self).get_config()
        config.update({'units': self.units})
        return config


class TLAnd(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        self.output_size = 1
        super(TLAnd, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        output = tf.math.reduce_min(inputs, 1, keepdims=True)
        return output, [output]

    def get_config(self):
        config = super(TLAnd, self).get_config()
        config.update({'units': self.units})
        return config


class TLOR(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        self.output_size = 1
        super(TLOR, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        output = tf.math.reduce_max(inputs, 1, keepdims=True)
        return output, [output]

    def get_config(self):
        config = super(TLOR, self).get_config()
        config.update({'units': self.units})
        return config


class TLNext(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        self.output_size = 1
        super(TLNext, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        start = self.state_size - 1
        end = self.state_size

        prev = states[0][:, start:end]
        prev_state = states[0][:, 1:]

        output = prev
        new_state = tf.concat([prev_state, inputs], 1)

        return output, new_state

    def get_config(self):
        config = super(TLNext, self).get_config()
        config.update({'units': self.units})
        return config


class TLAlwLearnBounds(tf.keras.layers.Layer):
    def __init__(self, units, w, **kwargs):
        self.units = units
        self.w = w
        super(TLAlwLearnBounds, self).__init__(**kwargs)

    def build(self, input_shape):
        self.betas = self.add_weight(shape=(self.w, 1),
                                     initializer='random_normal',
                                     name='interval_weights')
        self.full_weights = K.get_value(self.betas).copy()

        weight = tf.math.sign(self.full_weights)
        sub = tf.ones_like(weight) * 9999
        weight_mod = tf.where(tf.equal(weight, -1), sub, weight)
        self.last_quant_weights = tf.identity(weight_mod)
        K.set_value(self.betas, weight_mod)

    def get_config(self):
        config = super(TLAlwLearnBounds, self).get_config()
        config.update({'units': self.units})

    def call(self, inputs):
        output = tf.reduce_min(tf.math.multiply(inputs, self.betas),
                               2, keepdims=True)
        return output

    def on_batch_end(self):
        weight = tf.math.sign(self.full_weights)
        sub = tf.ones_like(weight) * 9999
        weight_mod = tf.where(tf.equal(weight, -1), sub, weight)
        self.last_quant_weights = tf.identity(weight_mod)
        K.set_value(self.betas, weight_mod)


class TLAlw(tf.keras.layers.Layer):
    def __init__(self, units, start=None, end=None, **kwargs):
        '''
        :param units:
        :param start:
        :param end: if you want the timestep x to be included, set end = x + 2
        :param kwargs:
        '''
        self.units = units
        self.state_size = units
        self.output_size = 1
        self.bounded = (start is not None) and (end is not None)
        if self.bounded:
            # self.start = self.state_size - end
            # self.end = self.state_size - start
            self.start = start
            self.end = end
        else:
            self.start = self.state_size - 1
            self.end = self.state_size
        super(TLAlw, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        start = self.start
        end = self.end

        prev = states[0][:, start:end]
        prev_min = tf.math.reduce_min(prev, 1, keepdims=True)
        prev_state = states[0][:, 1:]

        if self.bounded:
            # output = prev_min
            # output = tf.minimum(inputs, prev_min)
            new_state = tf.concat([prev_state, inputs], 1)
            int = new_state[:, start:end]
            output = tf.math.reduce_min(int, 1, keepdims=True)
        else:
            output = tf.minimum(inputs, prev_min)
            new_state = tf.concat([prev_state, output], 1)

        return output, new_state

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.ones((batch_size, self.state_size)) * 9999

    def get_config(self):
        config = super(TLAlw, self).get_config()
        config.update({'units': self.units,
                       'start': self.start, 'end': self.end})
        return config


class TLEvLearnBounds(tf.keras.layers.Layer):
    def __init__(self, units, w, **kwargs):
        self.units = units
        self.w = w
        super(TLEvLearnBounds, self).__init__(**kwargs)

    def build(self, input_shape):
        self.betas = self.add_weight(shape=(self.w, 1),
                                     initializer='ones',
                                     name='interval_weights')
        self.full_weights = K.get_value(self.betas).copy()

        weight = tf.math.sign(self.full_weights)
        sub = tf.ones_like(weight) * -9999
        weight_mod = tf.where(tf.equal(weight, -1), sub, weight)
        self.last_quant_weights = tf.identity(weight_mod)
        K.set_value(self.betas, weight_mod)

    def get_config(self):
        config = super(TLEvLearnBounds, self).get_config()
        config.update({'units': self.units})

    def call(self, inputs):
        output = tf.reduce_max(tf.math.multiply(inputs, self.betas),
                               2, keepdims=True)
        return output

    def on_batch_end(self):
        new_weights = K.get_value(self.betas)
        weights_update = new_weights - self.last_quant_weights
        self.full_weights += weights_update

        weight = tf.math.sign(self.full_weights)
        sub = tf.ones_like(weight) * -9999
        weight_mod = tf.where(tf.equal(weight, -1), sub, weight)
        self.last_quant_weights = tf.identity(weight_mod)
        K.set_value(self.betas, weight_mod)


class TLEv(tf.keras.layers.Layer):
    def __init__(self, units, start=None, end=None, **kwargs):
        '''
        :param units:
        :param start:
        :param end: if you want the timestep x to be included, set end = x + 2
        :param kwargs:
        '''
        self.units = units
        self.state_size = units
        self.output_size = 1
        self.bounded = (start is not None) and (end is not None)
        if self.bounded:
            self.start = self.state_size - end
            self.end = self.state_size - start
        else:
            self.start = self.state_size - 1
            self.end = self.state_size
        super(TLEv, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    def call(self, inputs, states):
        start = self.start
        end = self.end

        prev = states[0][:, start:end]
        prev_max = tf.math.reduce_max(prev, 1, keepdims=True)
        prev_state = states[0][:, 1:]

        if self.bounded:
            # output = prev_max
            output = tf.maximum(inputs, prev_max)
            new_state = tf.concat([prev_state, inputs])
        else:
            output = tf.maximum(inputs, prev_max)
            new_state = tf.concat([prev_state, output], 1)

        return output, new_state

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        return tf.ones((batch_size, self.state_size)) * -9999

    def get_config(self):
        config = super(TLEv, self).get_config()
        config.update({'units': self.units,
                       'start': self.start, 'end': self.end})
        return config


class TLUntil(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        self.output_size = 1
        super(TLUntil, self).__init__(**kwargs)

    def build(self, input_shape):
        self.built = True

    # # MY IMPLEMENTATION
    # def call(self, inputs, states):
    #     start = self.units - 1
    #     end = self.units
    #
    #     phi_input = tf.reshape(inputs[:, 0], shape=(tf.shape(inputs)[0], 1))
    #     psi_input = tf.reshape(inputs[:, 1], shape=(tf.shape(inputs)[0], 1))
    #
    #     # INNER INF
    #     prev_inf = states[0][:, 0][:, start:end]
    #     new_inf = tf.minimum(prev_inf, phi_input)
    #     new_inf_state = tf.concat([
    #         states[0][:, 0][:, 1:],
    #         new_inf], 1)
    #
    #     # MIN PSI AND PHI
    #     min_psi_phi = tf.minimum(phi_input, psi_input)
    #
    #     # PREVIOUS SUP INPUT
    #     prev_b = states[0][:, 1]
    #
    #     # UPDATE B
    #     update_b = tf.minimum(prev_b, phi_input)
    #
    #     # max of update_b
    #     max_update_b = tf.reduce_max(update_b, 1, keepdims=True)
    #
    #     # sup
    #     sup = tf.maximum(max_update_b, min_psi_phi)
    #     out = sup
    #
    #     # save new input for sup
    #     new_b = tf.concat([
    #         update_b[:, 1:],
    #         min_psi_phi
    #     ], 1)
    #
    #     # UPDATE STATE
    #     new_state = tf.concat(
    #         [tf.expand_dims(new_inf_state, axis=1),
    #          tf.expand_dims(new_b, axis=1)], 1)
    #
    #     return out, new_state

    # HOUSSAM's IMPLEMENTATION
    def call(self, inputs, states):
        start = self.units - 1
        end = self.units

        phi_input = tf.reshape(inputs[:, 0], shape=(tf.shape(inputs)[0], 1))
        psi_input = tf.reshape(inputs[:, 1], shape=(tf.shape(inputs)[0], 1))

        # get prev out
        prev_r = states[0][:, start:end]

        # MAX prev_r and psi
        max_prev_r_psi = tf.maximum(prev_r, psi_input)

        # MIN phi and max_prev_r_psi
        out = tf.minimum(phi_input, max_prev_r_psi)

        # UPDATE OUT
        new_state = tf.concat([states[0][:, 1:], out], 1)

        return out, new_state

    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        # inner_inf = tf.ones((batch_size, 1, self.units)) * 9999
        # b_out = tf.ones((batch_size, 1, self.units)) * -9999
        # init_state = tf.concat([inner_inf, b_out], 1)

        return tf.ones((batch_size, self.state_size)) * -9999

    def get_config(self):
        config = super(TLEv, self).get_config()
        config.update({'units': self.units,
                       'start': self.start, 'end': self.end})
        return config
    