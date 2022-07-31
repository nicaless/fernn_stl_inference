#!/bin/bash

#echo 'Matlab Experiments'
#/Applications/MATLAB_R2021b.app/bin/matlab -nodesktop -nosplash -r "run_all_experiments_matlab;exit;"

echo 'Len 2'
python3 experiments/experiment_template.py 'hapt' 2 experiments/test_results/ --quantize --excl
#python3 experiments/experiment_template.py 'cruise' 2 experiments/test_results/ --quantize --excl
python3 experiments/experiment_template.py 'ecg' 2 experiments/test_results/ --quantize --excl
#python3 experiments/experiment_template.py 'ecg_cont' 2 experiments/test_results/ --quantize --excl
#python3 experiments/experiment_template.py 'lyft' 2 experiments/test_results/ --quantize --excl
#
#echo 'Len 3'
#python3 experiments/experiment_template.py 'hapt' 3 experiments/test_results/ --quantize --excl
#python3 experiments/experiment_template.py 'ecg' 3 experiments/test_results/ --quantize --excl
#python3 experiments/experiment_template.py 'ecg_cont' 3 experiments/test_results/ --quantize --excl
#python3 experiments/experiment_template.py 'lyft' 3 experiments/test_results/ --quantize --excl
#

