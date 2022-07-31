import sys
sys.path.append('/home/nicaless/repos/gan_tl')

from formulas.formula_utils import build_atom_choice, build_op_choice, \
    build_andor_choice, build_template_choice, build_since

from tensorflow.keras import Input
from tensorflow.keras.layers import RNN
from TL_learning_utils import TLPass


# TEMPLATES = ['C(C(pcp)cp)', 'C(C(pcp)cCp)', 'C(C(pcp)cC(pcp))',
#              'CC(pcp)cCp', 'CC(pcp)cCCp', 'CC(pcp)cC(pcp)', 'CC(pcp)cCC(pcp)']
TEMPLATES = ['C(C(pcp)cp)', 'CC(pcp)cCp', 'CC(pcp)cC(pcp)']


def five_deep(atom_names, input_shape, normalizer,
              quantize='one-hot', templates=TEMPLATES):
    CHOICE_CONFIG = {'template_choice': templates}

    Q = quantize
    num_inputs = len(atom_names)
    T = input_shape[0]
    inputs = []
    for i in range(num_inputs):
        ini = Input(input_shape)
        inputs.append(ini)

    template_choice = []
    # TEMPLATE 1 'C(C(pcp)cp)'
    if 'C(C(pcp)cp)' in templates:
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
            build_op_choice('tp1_op1', T, tp1_andor1,
                            CHOICE_CONFIG, quantize=Q, with_next=True)

        tp1_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp1_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp1_andor2, CHOICE_CONFIG = \
            build_andor_choice('tp1_andor2', tp1_op1, tp1_atom3,
                               CHOICE_CONFIG, quantize=Q)

        tp1_op2, CHOICE_CONFIG = \
            build_op_choice('tp1_op2', T, tp1_andor2,
                            CHOICE_CONFIG, quantize=Q)

        template_choice.append(tp1_op2)

    # TEMPLATE 2 'C(C(pcp)cCp)'
    if 'C(C(pcp)cCp)' in templates:
        tp2_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp2_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp2_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp2_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp2_andor1, CHOICE_CONFIG = \
            build_andor_choice('tp2_andor1', tp1_atom2, tp2_atom2,
                               CHOICE_CONFIG, quantize=Q)
        tp2_op1, CHOICE_CONFIG = \
            build_op_choice('tp2_op1', T, tp2_andor1,
                            CHOICE_CONFIG, quantize=Q, with_next=True)

        tp2_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp2_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp2_op2, CHOICE_CONFIG = \
            build_op_choice('tp2_op2', T, tp2_atom3,
                            CHOICE_CONFIG, quantize=Q, with_next=True)

        tp2_andor2, CHOICE_CONFIG = \
            build_andor_choice('tp2_andor2', tp2_op1, tp2_op2,
                               CHOICE_CONFIG, quantize=Q)

        tp2_op3, CHOICE_CONFIG = \
            build_op_choice('tp2_op3', T, tp2_andor2,
                            CHOICE_CONFIG, quantize=Q)

        template_choice.append(tp2_op3)

    # TEMPLATE 3 'C(C(pcp)cC(pcp))'
    if 'C(C(pcp)cC(pcp))' in templates:
        tp3_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp3_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp3_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp3_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp3_andor1, CHOICE_CONFIG = \
            build_andor_choice('tp3_andor1', tp3_atom1, tp3_atom2,
                               CHOICE_CONFIG, quantize=Q)
        tp3_op1, CHOICE_CONFIG = \
            build_op_choice('tp3_op1', T, tp3_andor1,
                            CHOICE_CONFIG, quantize=Q, with_next=True)

        tp3_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp3_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp3_atom4, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp3_4', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp3_andor2, CHOICE_CONFIG = \
            build_andor_choice('tp3_andor2', tp3_atom3, tp3_atom4,
                               CHOICE_CONFIG, quantize=Q)
        tp3_op2, CHOICE_CONFIG = \
            build_op_choice('tp3_op2', T, tp3_andor2,
                            CHOICE_CONFIG, quantize=Q, with_next=True)

        tp3_andor3, CHOICE_CONFIG = \
            build_andor_choice('tp3_andor3', tp3_op1, tp3_op2,
                               CHOICE_CONFIG, quantize=Q)

        tp3_op3, CHOICE_CONFIG = \
            build_op_choice('tp3_op3', T, tp3_andor3,
                            CHOICE_CONFIG, quantize=Q)

        template_choice.append(tp3_op3)

    # TEMPLATE 4 'CC(pcp)cCp'
    if 'CC(pcp)cCp' in templates:
        tp4_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp4_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp4_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp4_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp4_andor1, CHOICE_CONFIG = \
            build_andor_choice('tp4_andor1', tp4_atom1, tp4_atom2,
                               CHOICE_CONFIG, quantize=Q)
        tp4_op1, CHOICE_CONFIG = \
            build_op_choice('tp4_op1', T, tp4_andor1,
                            CHOICE_CONFIG, quantize=Q, with_next=True)
        tp4_op2, CHOICE_CONFIG = \
            build_op_choice('tp4_op2', T, tp4_op1, CHOICE_CONFIG, quantize=Q)

        tp4_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp4_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp4_op3, CHOICE_CONFIG = \
            build_op_choice('tp4_op3', T, tp4_atom3,
                            CHOICE_CONFIG, quantize=Q)
        tp4_andor2, CHOICE_CONFIG = \
            build_andor_choice('tp4_andor2', tp4_op2, tp4_op3,
                               CHOICE_CONFIG, quantize=Q)
        template_choice.append(tp4_andor2)

    # TEMPLATE 5 'CC(pcp)cCCp'
    if 'CC(pcp)cCCp' in templates:
        tp5_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp5_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp5_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp5_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp5_andor1, CHOICE_CONFIG = \
            build_andor_choice('tp5_andor1', tp5_atom1, tp5_atom2,
                               CHOICE_CONFIG, quantize=Q)
        tp5_op1, CHOICE_CONFIG = \
            build_op_choice('tp5_op1', T, tp5_andor1,
                            CHOICE_CONFIG, quantize=Q, with_next=True)
        tp5_op2, CHOICE_CONFIG = \
            build_op_choice('tp5_op2', T, tp5_op1, CHOICE_CONFIG, quantize=Q)

        tp5_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp5_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp5_op3, CHOICE_CONFIG = \
            build_op_choice('tp5_op3', T, tp5_atom3,
                            CHOICE_CONFIG, with_next=True, quantize=Q)
        tp5_op4, CHOICE_CONFIG = \
            build_op_choice('tp5_op4', T, tp5_op3, CHOICE_CONFIG, quantize=Q)
        tp5_andor2, CHOICE_CONFIG = \
            build_andor_choice('tp5_andor2', tp5_op2, tp5_op4,
                               CHOICE_CONFIG, quantize=Q)
        template_choice.append(tp5_andor2)

    # TEMPLATE 6 'CC(pcp)cC(pcp)':
    if 'CC(pcp)cC(pcp)' in templates:
        tp6_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp6_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp6_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp6_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp6_andor1, CHOICE_CONFIG = \
            build_andor_choice('tp6_andor1', tp6_atom1, tp6_atom2,
                               CHOICE_CONFIG, quantize=Q)
        tp6_op1, CHOICE_CONFIG = \
            build_op_choice('tp6_op1', T, tp6_andor1,
                            CHOICE_CONFIG, quantize=Q, with_next=True)
        tp6_op2, CHOICE_CONFIG = \
            build_op_choice('tp6_op2', T, tp6_op1, CHOICE_CONFIG, quantize=Q)

        tp6_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp6_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp6_atom4, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp6_4', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp6_andor2, CHOICE_CONFIG = \
            build_andor_choice('tp6_andor2', tp6_atom1, tp6_atom2,
                               CHOICE_CONFIG, quantize=Q)
        tp6_op3, CHOICE_CONFIG = \
            build_op_choice('tp6_op3', T, tp6_andor2,
                            CHOICE_CONFIG, quantize=Q)
        tp6_andor3, CHOICE_CONFIG = \
            build_andor_choice('tp6_andor3', tp6_op2, tp6_op3,
                               CHOICE_CONFIG, quantize=Q)
        template_choice.append(tp6_andor3)

    # TEMPLATE 7 'CC(pcp)cCC(pcp)'
    if 'CC(pcp)cCC(pcp)' in templates:
        tp7_atom1, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp7_1', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp7_atom2, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp7_2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp7_andor1, CHOICE_CONFIG = \
            build_andor_choice('tp7_andor1', tp7_atom1, tp7_atom2,
                               CHOICE_CONFIG, quantize=Q)
        tp7_op1, CHOICE_CONFIG = \
            build_op_choice('tp7_op1', T, tp7_andor1,
                            CHOICE_CONFIG, quantize=Q, with_next=True)
        tp7_op2, CHOICE_CONFIG = \
            build_op_choice('tp7_op2', T, tp7_op1, CHOICE_CONFIG, quantize=Q)

        tp7_atom3, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp7_3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp7_atom4, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'tp7_4', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        tp7_andor2, CHOICE_CONFIG = \
            build_andor_choice('tp7_andor2', tp7_atom2, tp7_atom2,
                               CHOICE_CONFIG, quantize=Q)
        tp7_op3, CHOICE_CONFIG = \
            build_op_choice('tp7_op3', T, tp7_andor2,
                            CHOICE_CONFIG, with_next=True, quantize=Q)
        tp7_op4, CHOICE_CONFIG = \
            build_op_choice('tp7_op4', T, tp7_op3,
                            CHOICE_CONFIG, quantize=Q)
        tp7_andor3, CHOICE_CONFIG = \
            build_andor_choice('tp7_andor3', tp7_op2, tp7_op4,
                               CHOICE_CONFIG, quantize=Q)
        template_choice.append(tp7_andor3)

    template_choice = build_template_choice(template_choice)

    formula = RNN(TLPass(1), name='formula')(template_choice)
    return formula, inputs, CHOICE_CONFIG, normalizer
