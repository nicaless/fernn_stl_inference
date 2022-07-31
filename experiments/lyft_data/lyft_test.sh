#!/bin/bash

echo 'Starting Lyft Test'

echo 'choose (Alw, Ev) (a1.x)'
python3 experiments/lyft_data/main.py 1 'test_1_quantize' --quantize
#python3 experiments/lyft_data/main.py 1 'test_1_rast' --rast
python3 experiments/lyft_data/main.py 1 'test_1'

echo 'choose (Alw, Ev) (choose (a1.x, a1.y)'
python3 experiments/lyft_data/main.py 2 'test_2_quantize' --quantize
#python3 experiments/lyft_data/main.py 2 'test_2_rast' --rast
python3 experiments/lyft_data/main.py 2 'test_2'

echo 'choose (Alw, Ev) (choose (a1.x AND a1.y,  a2.x OR a2.y) )'
python3 experiments/lyft_data/main.py 3 'test_3_quantize' --quantize
#python3 experiments/lyft_data/main.py 3 'test_3_rast' --rast
python3 experiments/lyft_data/main.py 3 'test_3'

echo 'Last Test'
python3 experiments/lyft_data/main.py 4 'test_4_quantize' --quantize
#python3 experiments/lyft_data/main.py 4 'test_4_rast' --rast
python3 experiments/lyft_data/main.py 4 'test_4'
