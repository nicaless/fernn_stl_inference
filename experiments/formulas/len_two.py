import sys
sys.path.append('/home/nicaless/repos/gan_tl')

from formulas.formula_utils import build_atom_choice, build_op_choice, \
    build_andor_choice, build_template_choice, build_since

from tensorflow.keras import Input
from tensorflow.keras.layers import RNN
from TL_learning_utils import TLPass


def len_two(atom_names, input_shape, normalizer, quantize='one-hot'):
    CHOICE_CONFIG = {}

    Q = quantize
    num_inputs = len(atom_names)
    T = input_shape[0]
    inputs = []
    for i in range(num_inputs):
        ini = Input(input_shape)
        inputs.append(ini)

    atom_choice, normalizer, CHOICE_CONFIG = \
        build_atom_choice(atom_names, 'tp1', inputs, normalizer,
                          CHOICE_CONFIG, quantize=Q)

    op_choice, CHOICE_CONFIG = \
        build_op_choice('tp1', T, atom_choice, CHOICE_CONFIG, quantize=Q)

    formula = RNN(TLPass(1), name='formula')(op_choice)

    return formula, inputs, CHOICE_CONFIG, normalizer
