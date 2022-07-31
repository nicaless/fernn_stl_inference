#!/bin/bash

echo 'Starting HAPT Test'

echo 'Alw (x)'
python3 experiments/hapt_data/main.py 1 'test_1_quantize' --quantize
#python3 experiments/hapt_data/main.py 1 'test_1_rast' --rast
#python3 experiments/hapt_data/main.py 1 'test_1'

echo 'choose (Alw, Ev) (x)'
python3 experiments/hapt_data/main.py 2 'test_2_quantize' --quantize
#python3 experiments/hapt_data/main.py 2 'test_2_rast' --rast
python3 experiments/hapt_data/main.py 2 'test_2'
