import sys
sys.path.append('/home/nicaless/repos/gan_tl')

from formulas.formula_utils import build_atom_choice, build_op_choice, \
    build_andor_choice, build_template_choice, build_since

from tensorflow.keras import Input
from tensorflow.keras.layers import RNN
from TL_learning_utils import TLPass


def len_five(atom_names, input_shape, normalizer, quantize='one-hot', with_since=False):
    CHOICE_CONFIG = {'template_choice': ['CavCb', 'CC(avb)', 'C(acCb)']}

    Q = quantize
    num_inputs = len(atom_names)
    T = input_shape[0]
    inputs = []
    for i in range(num_inputs):
        ini = Input(input_shape)
        inputs.append(ini)

    atom1_choice, normalizer, CHOICE_CONFIG = \
        build_atom_choice(atom_names, 'tp1_1', inputs, normalizer,
                          CHOICE_CONFIG, quantize=Q)
    atom2_choice, normalizer, CHOICE_CONFIG = \
        build_atom_choice(atom_names, 'tp1_2', inputs, normalizer,
                          CHOICE_CONFIG, quantize=Q)

    # 'CavCb'
    op11_choice, CHOICE_CONFIG = \
        build_op_choice('tp1_op1', T, atom1_choice, CHOICE_CONFIG, quantize=Q)
    op12_choice, CHOICE_CONFIG = \
        build_op_choice('tp1_op2', T, atom2_choice, CHOICE_CONFIG, quantize=Q)
    tp1_andor, CHOICE_CONFIG = \
        build_andor_choice('tp1', op11_choice, op12_choice, CHOICE_CONFIG, quantize=Q)

    # 'CCacb'
    tp2_andor, CHOICE_CONFIG = \
        build_andor_choice('tp2', op11_choice, op12_choice, CHOICE_CONFIG, quantize=Q)

    op_choice, CHOICE_CONFIG = \
        build_op_choice('tp1_op1', T, tp1_andor, CHOICE_CONFIG, quantize=Q)

    if with_since:
        CHOICE_CONFIG['template_choice'] = ['Cavb', 'C(aUb)', 'CaUb', 'aUCb']

        # C(aUb)
        until1_output = build_since('tp2', T, [atom1_choice, atom3_choice])

        # CaUb
        op3_choice, CHOICE_CONFIG = \
            build_op_choice('tp3_op1', T, atom1_choice, CHOICE_CONFIG, quantize=Q)
        until2_output = build_since('tp3', T, [op3_choice, atom2_choice])

        # aUCb
        op4_choice, CHOICE_CONFIG = \
            build_op_choice('tp4_op1', T, atom2_choice, CHOICE_CONFIG, quantize=Q)
        until3_output = build_since('tp4', T, [atom1_choice, op4_choice])

        template_choice = build_template_choice([op_choice, until1_output, until2_output, until3_output])
        formula = RNN(TLPass(1), name='formula')(template_choice)
    else:
        formula = RNN(TLPass(1), name='formula')(op_choice)

    return formula, inputs, CHOICE_CONFIG, normalizer
