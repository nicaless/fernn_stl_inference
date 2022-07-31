#!/bin/bash

echo 'Starting Cruise Control Test'

echo 'Alw (x)'
python3 experiments/cruise_control_train/main.py 1 'test_1_quantize' --quantize

echo 'choose (Alw, Ev) (x)'
python3 experiments/cruise_control_train/main.py 2 'test_2_quantize' --quantize
python3 experiments/cruise_control_train/main.py 2 'test_2'
