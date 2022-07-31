import sys
sys.path.append('/home/nicaless/repos/gan_tl')

from formulas.formula_utils import build_atom_choice, build_op_choice, \
    build_andor_choice, build_template_choice, build_since

from tensorflow.keras import Input
from tensorflow.keras.layers import RNN
from TL_learning_utils import TLPass


def three_deep(atom_names, input_shape, normalizer,
               quantize='one-hot', with_since=False):
    CHOICE_CONFIG = {'template_choice': ['CCp', 'CpcCp', 'C(pcp)']}

    Q = quantize
    num_inputs = len(atom_names)
    T = input_shape[0]
    inputs = []
    for i in range(num_inputs):
        ini = Input(input_shape)
        inputs.append(ini)

    # TEMPLATE 1 CCp
    tp1_atom, normalizer, CHOICE_CONFIG = \
        build_atom_choice(atom_names, 'tp1', inputs, normalizer,
                          CHOICE_CONFIG, quantize=Q)

    tp1_op1, CHOICE_CONFIG = \
        build_op_choice('tp1_op1', T, tp1_atom, CHOICE_CONFIG, quantize=Q,
                        with_next=True)

    tp1_op2, CHOICE_CONFIG = \
        build_op_choice('tp1_op2', T, tp1_op1, CHOICE_CONFIG, quantize=Q)

    # TEMPLATE 2 CpcCp
    tp2_atom1, normalizer, CHOICE_CONFIG = \
        build_atom_choice(atom_names, 'tp2_1', inputs, normalizer,
                          CHOICE_CONFIG, quantize=Q)

    tp2_op1, CHOICE_CONFIG = \
        build_op_choice('tp2_op1', T, tp2_atom1, CHOICE_CONFIG, quantize=Q)

    tp2_atom2, normalizer, CHOICE_CONFIG = \
        build_atom_choice(atom_names, 'tp2_2', inputs, normalizer,
                          CHOICE_CONFIG, quantize=Q)

    tp2_op2, CHOICE_CONFIG = \
        build_op_choice('tp2_op2', T, tp2_atom2, CHOICE_CONFIG, quantize=Q)

    tp2_andor, CHOICE_CONFIG = \
        build_andor_choice('tp2', tp2_op1, tp2_op2, CHOICE_CONFIG, quantize=Q)

    # TEMPLATE 3 C(pcp)
    tp3_atom1, normalizer, CHOICE_CONFIG = \
        build_atom_choice(atom_names, 'tp3_1', inputs, normalizer,
                          CHOICE_CONFIG, quantize=Q)

    tp3_atom2, normalizer, CHOICE_CONFIG = \
        build_atom_choice(atom_names, 'tp3_2', inputs, normalizer,
                          CHOICE_CONFIG, quantize=Q)

    tp3_andor, CHOICE_CONFIG = \
        build_andor_choice('tp3', tp3_atom1, tp3_atom2,
                           CHOICE_CONFIG, quantize=Q)

    tp3_op, CHOICE_CONFIG = \
        build_op_choice('tp3', T, tp3_andor, CHOICE_CONFIG, quantize=Q)

    if with_since:
        CHOICE_CONFIG['template_choice'].append(['(pcp)Sp'])
        CHOICE_CONFIG['template_choice'].append(['pS(pcp)'])
        CHOICE_CONFIG['template_choice'].append(['(pcp)S(pcp)'])

        # TEMPLATE 4 (pcp)Sp
        tp4_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp4_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp4_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp4_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp4_andor, CHOICE_CONFIG = \
            build_andor_choice('tp4', tp4_atom1, tp4_atom2,
                               CHOICE_CONFIG, quantize=Q)

        tp4_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp4_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp4_until = build_since('tp4', T, [tp4_andor, tp4_atom3])

        # TEMPLATE 5 pS(pcp)
        tp5_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp5_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp5_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp5_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp5_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp5_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp5_andor, CHOICE_CONFIG = \
            build_andor_choice('tp5', tp5_atom2, tp5_atom3,
                               CHOICE_CONFIG, quantize=Q)

        tp5_until = build_since('tp5', T, [tp5_atom1, tp5_andor])

        # TEMPLATE 6 (pcp)S(pcp)
        tp6_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp6_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp6_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp6_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp6_andor1, CHOICE_CONFIG = \
            build_andor_choice('tp6_andor1', tp6_atom1, tp6_atom2,
                               CHOICE_CONFIG, quantize=Q)

        tp6_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp6_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp6_atom4, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp6_4', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp6_andor2, CHOICE_CONFIG = \
            build_andor_choice('tp6_andor2', tp6_atom3, tp6_atom4,
                               CHOICE_CONFIG, quantize=Q)

        tp6_until = build_since('tp6', T, [tp6_andor1, tp6_andor2])

        template_choice = build_template_choice([tp1_op2, tp2_andor, tp3_op,
                                                 tp4_until, tp5_until,
                                                 tp6_until])
    else:
        template_choice = build_template_choice([tp1_op2, tp2_andor, tp3_op])

    formula = RNN(TLPass(1), name='formula')(template_choice)
    return formula, inputs, CHOICE_CONFIG, normalizer
