import sys
sys.path.append('/home/nicaless/repos/gan_tl')

from formulas.formula_utils import build_atom_choice, build_op_choice, \
    build_andor_choice, build_template_choice, build_since

from tensorflow.keras import Input
from tensorflow.keras.layers import RNN
from TL_learning_utils import TLPass


# TEMPLATES = ['C(pcp)cCp', 'C(pcp)cC(pcp)',
#              'CC(pcp)', 'C((Cp)cp)',
#              '(CCp)c(Cp)', '(CCp)c(CCp)']
TEMPLATES = ['C(pcp)cCp', 'CC(pcp)', '(CCp)c(Cp)']


def four_deep(atom_names, input_shape, normalizer,
              quantize='one-hot', templates=TEMPLATES):
    CHOICE_CONFIG = {'template_choice': TEMPLATES}

    Q = quantize
    num_inputs = len(atom_names)
    T = input_shape[0]
    inputs = []
    for i in range(num_inputs):
        ini = Input(input_shape)
        inputs.append(ini)

    template_choice = []
    # TEMPLATE 1 'C(pcp)cCp'
    if 'C(pcp)cCp' in templates:
        tp1_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp1_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp1_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp1_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp1_andor1, CHOICE_CONFIG = \
            build_andor_choice('tp1_andor1', tp1_atom1, tp1_atom2,
                               CHOICE_CONFIG, quantize=Q)

        tp1_op1, CHOICE_CONFIG = \
            build_op_choice('tp1_op1', T, tp1_andor1, CHOICE_CONFIG, quantize=Q)

        tp1_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp1_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp1_op2, CHOICE_CONFIG = \
            build_op_choice('tp1_op2', T, tp1_atom3, CHOICE_CONFIG, quantize=Q)

        tp1_andor2, CHOICE_CONFIG = \
            build_andor_choice('tp1_andor2', tp1_op1, tp1_op2,
                               CHOICE_CONFIG, quantize=Q)
        template_choice.append(tp1_andor2)

    # TEMPLATE 2 'C(pcp)cC(pcp)'
    if 'C(pcp)cC(pcp)' in templates:
        tp2_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp2_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp2_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp2_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp2_andor1, CHOICE_CONFIG = \
            build_andor_choice('tp2_andor1', tp2_atom1, tp2_atom2,
                               CHOICE_CONFIG, quantize=Q)

        tp2_op1, CHOICE_CONFIG = \
            build_op_choice('tp2_op1', T, tp2_andor1, CHOICE_CONFIG, quantize=Q)

        tp2_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp2_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp2_atom4, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp2_4', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp2_andor2, CHOICE_CONFIG = \
            build_andor_choice('tp2_andor2', tp2_atom3, tp2_atom4,
                               CHOICE_CONFIG, quantize=Q)

        tp2_op2, CHOICE_CONFIG = \
            build_op_choice('tp2_op2', T, tp2_andor2, CHOICE_CONFIG, quantize=Q)

        tp2_andor3, CHOICE_CONFIG = \
            build_andor_choice('tp2_andor3', tp2_op1, tp2_op2,
                               CHOICE_CONFIG, quantize=Q)
        template_choice.append(tp2_andor3)

    # TEMPLATE 3 'CC(pcp)'
    if 'CC(pcp)' in templates:
        tp3_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp3_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp3_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp3_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp3_andor, CHOICE_CONFIG = \
            build_andor_choice('tp3', tp3_atom1, tp3_atom2,
                               CHOICE_CONFIG, quantize=Q)

        tp3_op1, CHOICE_CONFIG = \
            build_op_choice('tp3_op1', T, tp3_andor, CHOICE_CONFIG,
                            quantize=Q, with_next=True)

        tp3_op2, CHOICE_CONFIG = \
            build_op_choice('tp3_op2', T, tp3_op1, CHOICE_CONFIG,
                            quantize=Q)
        template_choice.append(tp3_op2)

    # TEMPLATE 4 'C((Cp)cp)'
    if 'C((Cp)cp)' in templates:
        tp4_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp4_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp4_op1, CHOICE_CONFIG = \
            build_op_choice('tp4_op1', T, tp4_atom1, CHOICE_CONFIG,
                            quantize=Q, with_next=True)

        tp4_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp4_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        tp4_andor, CHOICE_CONFIG = \
            build_andor_choice('tp4', tp4_op1, tp4_atom2,
                               CHOICE_CONFIG, quantize=Q)

        tp4_op2, CHOICE_CONFIG = \
            build_op_choice('tp4_op2', T, tp4_andor, CHOICE_CONFIG, quantize=Q)
        template_choice.append(tp4_op2)

    # TEMPLATE 5 '(CCp)c(Cp)'
    if '(CCp)c(Cp)' in templates:
        tp5_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp5_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp5_op1, CHOICE_CONFIG = \
            build_op_choice('tp5_op1', T, tp5_atom1, CHOICE_CONFIG,
                            quantize=Q, with_next=True)
        tp5_op2, CHOICE_CONFIG = \
            build_op_choice('tp5_op2', T, tp5_op1, CHOICE_CONFIG, quantize=Q)

        tp5_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp5_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp5_op3, CHOICE_CONFIG = \
            build_op_choice('tp5_op3', T, tp5_atom2, CHOICE_CONFIG, quantize=Q)

        tp5_andor, CHOICE_CONFIG = \
            build_andor_choice('tp5', tp5_op2, tp5_op3,
                               CHOICE_CONFIG, quantize=Q)

        template_choice.append(tp5_andor)

    # TEMPLATE 6 '(CCp)c(CCp)':
    if '(CCp)c(CCp)' in templates:
        tp6_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp6_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp6_op1, CHOICE_CONFIG = \
            build_op_choice('tp6_op1', T, tp6_atom1, CHOICE_CONFIG,
                            quantize=Q, with_next=True)
        tp6_op2, CHOICE_CONFIG = \
            build_op_choice('tp6_op2', T, tp6_op1, CHOICE_CONFIG, quantize=Q)

        tp6_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp6_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp6_op3, CHOICE_CONFIG = \
            build_op_choice('tp6_op3', T, tp6_atom2, CHOICE_CONFIG,
                            quantize=Q, with_next=True)
        tp6_op4, CHOICE_CONFIG = \
            build_op_choice('tp6_op4', T, tp6_op3, CHOICE_CONFIG, quantize=Q)

        tp6_andor, CHOICE_CONFIG = \
            build_andor_choice('tp6', tp6_op2, tp6_op4,
                               CHOICE_CONFIG, quantize=Q)

        template_choice.append(tp6_andor)

    template_choice = build_template_choice(template_choice)

    formula = RNN(TLPass(1), name='formula')(template_choice)
    return formula, inputs, CHOICE_CONFIG, normalizer
