import tensorflow as tf
from tensorflow.keras.layers import concatenate, RNN
from TLOps import TLEv, TLAlw, TLNext, TLAtom, TLAnd, TLOR, TLUntil, \
    TLAlwLearnBounds, TLEvLearnBounds
from TL_learning_utils import TLWeightTransform, TLPass


# @tf.function
def build_atom_choice(atom_names, template_name, inputs, normalizer,
                      choice_config, quantize='one-hot'):
    atoms = []
    new_atom_names = []
    num_inputs = len(inputs)
    for i in range(num_inputs):
        ini = inputs[i]
        new_atom_name = template_name + '_' + atom_names[i]
        normalizer[new_atom_name] = normalizer[atom_names[i]]
        atom = RNN(TLAtom(1), return_sequences=True, name=new_atom_name)(ini)
        atoms.append(atom)
        new_atom_names.append(new_atom_name)

    atom_choice_input = concatenate(atoms)
    name = template_name + '_' + 'atom_choice'
    atom_choice = TLWeightTransform(1, transform='sum', quantize=quantize,
                                    name=name)(atom_choice_input)
    choice_config[name] = new_atom_names
    return atom_choice, normalizer, choice_config


# @tf.function
def build_op_choice(template_name, T, op_input, choice_config,
                    quantize='one-hot', with_next=False, learn_bounds=True):

    if learn_bounds:
        # sig_opt_input = tf.keras.layers.Activation(tf.keras.activations.sigmoid)(op_input)

        alw_name = template_name + '_' + 'alw'
        alw = TLAlwLearnBounds(1, w=T, name=alw_name)(op_input)
        ev_name = template_name + '_' + 'ev'
        ev = TLEvLearnBounds(1, w=T, name=ev_name)(op_input)
    else:
        alw_name = template_name + '_' + 'alw'
        alw = RNN(TLAlw(T), return_sequences=True, name=alw_name)(op_input)
        ev_name = template_name + '_' +  'ev'
        ev = RNN(TLEv(T), return_sequences=True, name=ev_name)(op_input)

    if with_next:
        next_name = template_name + '_' + 'X'
        next = RNN(TLNext(T), return_sequences=True, name=next_name)(op_input)
        op_choice_names = [alw_name, ev_name, next_name]
        op_choice_input = concatenate([alw, ev, next])
    else:
        op_choice_names = [alw_name, ev_name]
        op_choice_input = concatenate([alw, ev])

    name = template_name + '_' + 'op_choice'
    op_choice = TLWeightTransform(1, transform='sum', quantize=quantize,
                                  name=name)(op_choice_input)
    choice_config[name] = op_choice_names
    return op_choice, choice_config


# @tf.function
def build_andor_choice(template_name, input1, input2, choice_config,
                       quantize='one-hot'):
    inputs = concatenate([input1, input2])

    and_input = TLWeightTransform(1, w=1)(inputs)
    and_name = template_name + '_' + 'and'
    and_output = RNN(TLAnd(1), return_sequences=True, name=and_name)(and_input)

    or_input = TLWeightTransform(1, w=1)(inputs)
    or_name = template_name + '_' + 'or'
    or_output = RNN(TLOR(1), return_sequences=True, name=or_name)(or_input)

    andor_choice_names = [and_name, or_name]
    choice_input = concatenate([and_output, or_output])

    name = template_name + '_' + 'andor_choice'
    choice_output = TLWeightTransform(1, transform='sum', quantize=quantize,
                                      name=name)(choice_input)
    choice_config[name] = andor_choice_names
    return choice_output, choice_config


# @tf.function
def build_binary_choice(template_name, T, input1, input2, choice_config,
                        quantize='one-hot', with_since=False):
    inputs = concatenate([input1, input2])

    and_input = TLWeightTransform(1, w=1)(inputs)
    and_name = template_name + '_and'
    and_output = RNN(TLAnd(1), return_sequences=True, name=and_name)(and_input)

    or_input = TLWeightTransform(1, w=1)(inputs)
    or_name = template_name + '_or'
    or_output = RNN(TLOR(1), return_sequences=True, name=or_name)(or_input)

    if with_since:
        until_output = RNN(TLUntil(T), return_sequences=True,
                           name=template_name + '_' + 'since')(inputs)
        bin_choice_names = [and_name, or_name, template_name + '_' + 'since']
        choice_input = concatenate([and_output, or_output, until_output])
    else:
        bin_choice_names = [and_name, or_name]
        choice_input = concatenate([and_output, or_output])

    name = template_name + '_' + 'bin_choice'
    choice_output = TLWeightTransform(1, transform='sum', quantize=quantize,
                                      name=name)(choice_input)
    choice_config[name] = bin_choice_names
    return choice_output, choice_config


# @tf.function
def build_template_choice(template_outputs, name='template_choice', quantize='one-hot'):
    template_choice_input = concatenate(template_outputs)
    template_choice = TLWeightTransform(
        1, transform='sum', quantize=quantize,
        name=name)(template_choice_input)
    return template_choice


# @tf.function
def build_since(template_name, T, since_inputs):
    until_input = concatenate(since_inputs)
    until_output = RNN(TLUntil(T), return_sequences=True,
                       name=template_name + '_' + 'since')(until_input)
    return until_output
