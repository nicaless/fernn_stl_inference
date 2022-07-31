#!/bin/bash

echo 'Starting ECG Test'

echo 'choose (Alw, Ev) (hr)'
python3 experiments/ecg/main.py 1 'test_1_quantize' --quantize
#python3 experiments/ecg/main.py 1 'test_1_rast' --rast
python3 experiments/ecg/main.py 1 'test_1'

echo 'choose (Alw, Ev) (hr_dx)'
python3 experiments/ecg/main.py 2 'test_2_quantize' --quantize
#python3 experiments/ecg/main.py 2 'test_2_rast' --rast
python3 experiments/ecg/main.py 2 'test_2'

echo 'choose (Alw, Ev) (choose hr, hr_dx)'
python3 experiments/ecg/main.py 3 'test_3_quantize' --quantize
#python3 experiments/ecg/main.py 3 'test_3_rast' --rast
python3 experiments/ecg/main.py 3 'test_3'

echo 'choose (Alw, Ev) (choose (hr AND hr_dx,  hr, OR hr_dx) )'
python3 experiments/ecg/main.py 4 'test_4_quantize' --quantize
#python3 experiments/ecg/main.py 4 'test_4_rast' --rast
python3 experiments/ecg/main.py 4 'test_4'

echo 'choose (Alw, Ev) (choose hr, hr_dx)  AND choose (Alw, Ev) (choose hr, hr_dx)'
python3 experiments/ecg/main.py 5 'test_5_quantize' --quantize
#python3 experiments/ecg/main.py 5 'test_5_rast' --rast
python3 experiments/ecg/main.py 5 'test_5'

echo 'choose (Alw, Ev) (choose hr, hr_dx)  AND choose (Alw, Ev) (choose hr, hr_dx)'
python3 experiments/ecg/main.py 6 'test_6_quantize' --quantize
#python3 experiments/ecg/main.py 6 'test_6_rast' --rast
python3 experiments/ecg/main.py 6 'test_6'

echo 'choose (Alw, Ev) (choose hr, hr_dx)  AND choose (Alw, Ev) (choose hr, hr_dx)'
python3 experiments/ecg/main.py 7 'test_7_quantize' --quantize
#python3 experiments/ecg/main.py 7 'test_7_rast' --rast
python3 experiments/ecg/main.py 7 'test_7'
