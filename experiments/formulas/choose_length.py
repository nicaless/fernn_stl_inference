import sys
sys.path.append('/home/nicaless/repos/gan_tl')

from formulas.formula_utils import build_atom_choice, build_op_choice, \
    build_andor_choice, build_template_choice, build_since, build_binary_choice

from tensorflow.keras import Input
from tensorflow.keras.layers import RNN
from TL_learning_utils import TLPass, TLWeightTransform


def choose_length(atom_names, input_shape, normalizer,
                  L=2, quantize='one-hot', with_since=False, excl=False):
    CHOICE_CONFIG = {}

    Q = quantize
    num_inputs = len(atom_names)
    T = input_shape[0]
    inputs = []
    for i in range(num_inputs):
        ini = Input(input_shape)
        inputs.append(ini)

    atom1_choice, normalizer, CHOICE_CONFIG = \
        build_atom_choice(atom_names, 'base1', inputs, normalizer,
                          CHOICE_CONFIG, quantize=Q)

    op1_choice_output, CHOICE_CONFIG = \
        build_op_choice('op1', T, atom1_choice, CHOICE_CONFIG, quantize=Q)

    if L > 2:
        CHOICE_CONFIG['template_choice'] = []
        output_choices = []
        atom2_choice, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'base2', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)
        if not excl:
            CHOICE_CONFIG['template_choice'].append('Cp')
            output_choices.append(op1_choice_output)

    else:
        formula = RNN(TLPass(1), name='formula')(op1_choice_output)
        return formula, inputs, CHOICE_CONFIG, normalizer

    if L >= 3:
        len_3_choices = []
        len_3_choices_names = []
        op2_choice_output, CHOICE_CONFIG = \
            build_op_choice('CCp', T, op1_choice_output, CHOICE_CONFIG, quantize=Q)
        len_3_choices.append(op2_choice_output)
        len_3_choices_names.append('CCp')

        if with_since:
            until_output = build_since('pUp', T, [atom1_choice, atom2_choice])
            until_out = TLWeightTransform(1, w=1, transform='sum',
                                          quantize=quantize)(until_output)
            len_3_choices.append(until_out)
            len_3_choices_names.append('pUp')

        len_3_output = build_template_choice(len_3_choices,
                                             name='len_3_choice')
        if (not excl) or (excl and (L == 3)):
            CHOICE_CONFIG['len_3_choice'] = len_3_choices_names
            CHOICE_CONFIG['template_choice'].append('len_3')
            output_choices.append(len_3_output)

    if L >= 4:
        len_4_choices = []
        len_4_choices_names = []

        bin1_choice, CHOICE_CONFIG = \
            build_binary_choice('C_pBp', T, atom1_choice, atom2_choice,
                                CHOICE_CONFIG, quantize=Q, with_since=with_since)
        op3_choice_output, CHOICE_CONFIG = \
            build_op_choice('op3', T, bin1_choice, CHOICE_CONFIG, quantize=Q)
        len_4_choices.append(op3_choice_output)
        len_4_choices_names.append('C_pBp')

        if with_since:
            until1_output = build_since('CpUp', T, [op1_choice_output,
                                                    atom2_choice])
            until1_out = TLWeightTransform(1, w=1, transform='sum',
                                           quantize=quantize)(until1_output)
            len_4_choices.append(until1_out)
            len_4_choices_names.append('CpUp')

            until2_output = build_since('pUCp', T, [atom2_choice,
                                                    op1_choice_output])
            until2_out = TLWeightTransform(1, w=1, transform='sum',
                                           quantize=quantize)(until2_output)
            len_4_choices.append(until2_out)
            len_4_choices_names.append('pUCp')

        len_4_output = build_template_choice(len_4_choices,
                                             name='len_4_choice')
        if (not excl) or (excl and (L == 4)):
            CHOICE_CONFIG['len_4_choice'] = len_4_choices_names
            CHOICE_CONFIG['template_choice'].append('len_4')
            output_choices.append(len_4_output)

    if L >= 5:
        len_5_choices = []
        len_5_choices_names = []

        op4_choice_output, CHOICE_CONFIG = \
            build_op_choice('CC_pBp', T, op3_choice_output, CHOICE_CONFIG,
                            quantize=Q)
        len_5_choices.append(op4_choice_output)
        len_5_choices_names.append('CC_pBp')

        atom3_choice, normalizer, CHOICE_CONFIG = \
            build_atom_choice(atom_names, 'base3', inputs, normalizer,
                              CHOICE_CONFIG, quantize=Q)

        op5_choice_output, CHOICE_CONFIG = \
            build_op_choice('op5', T, atom3_choice, CHOICE_CONFIG,
                            quantize=Q)

        bin2_choice, CHOICE_CONFIG = \
            build_binary_choice('C_pBCp', T,
                                atom1_choice, op5_choice_output,
                                CHOICE_CONFIG, quantize=Q, with_since=with_since)
        op6_choice, CHOICE_CONFIG = \
            build_op_choice('C_pBCp', T, bin2_choice, CHOICE_CONFIG,
                            quantize=Q)
        len_5_choices.append(op6_choice)
        len_5_choices_names.append('C_pBCp')

        bin3_choice, CHOICE_CONFIG = \
            build_binary_choice('CpBCp', T,
                                op1_choice_output, op5_choice_output,
                                CHOICE_CONFIG, quantize=Q, with_since=with_since)
        len_5_choices.append(bin3_choice)
        len_5_choices_names.append('CpBCp')

        len_5_output = build_template_choice(len_5_choices,
                                             name='len_5_choice')
        if (not excl) or (excl and (L == 5)):
            CHOICE_CONFIG['len_5_choice'] = len_5_choices_names
            CHOICE_CONFIG['template_choice'].append('len_5')
            output_choices.append(len_5_output)

    if L >= 6:
        len_6_choices = []
        len_6_choices_names = []

        # op6_choice_output, CHOICE_CONFIG = \
        #     build_op_choice('op6', T, op5_choice_output, CHOICE_CONFIG,
        #                     quantize=Q)
        bin4_choice, CHOICE_CONFIG =\
            build_binary_choice('CCpBCp', T,
                                len_3_output, op5_choice_output,  #op6_choice_output,
                                CHOICE_CONFIG, quantize=Q, with_since=with_since)
        len_6_choices.append(bin4_choice)
        len_6_choices_names.append('CCpBCp')

        bin5_choice, CHOICE_CONFIG =\
            build_binary_choice('CpBCCp', T,
                                # op6_choice_output,
                                op5_choice_output, len_3_output,
                                CHOICE_CONFIG, quantize=Q, with_since=with_since)
        len_6_choices.append(bin5_choice)
        len_6_choices_names.append('CpBCCp')
        if 'len_3_choice' not in CHOICE_CONFIG:
            CHOICE_CONFIG['len_3_choice'] = len_3_choices_names

        len_6_output = build_template_choice(len_6_choices,
                                             name='len_6_choice')
        if (not excl) or (excl and (L == 6)):
            CHOICE_CONFIG['len_6_choice'] = len_6_choices_names
            CHOICE_CONFIG['template_choice'].append('len_6')
            output_choices.append(len_6_output)

    template_choice = build_template_choice(output_choices)
    formula = RNN(TLPass(1), name='formula')(template_choice)

    return formula, inputs, CHOICE_CONFIG, normalizer
