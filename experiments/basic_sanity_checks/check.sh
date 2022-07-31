#!/bin/bash

filename='experiments/basic_sanity_checks/training_results.txt'
rm $filename

echo 'Starting Sanity Checks'

echo 'Generating Test Data'
python3 experiments/basic_sanity_checks/generate_random_data.py

echo 'Testing TL Until' && echo 'Testing TL Until' >> $filename
python3 experiments/basic_sanity_checks/tl_until.py

echo 'Testing TL Ev' && echo 'Testing TL Ev' >> $filename
python3 experiments/basic_sanity_checks/tl_ev.py

echo 'Testing TL Alw' && echo 'Testing TL Alw' >> $filename
python3 experiments/basic_sanity_checks/tl_alw.py

echo 'Testing TL Next' && echo 'Testing TL Next' >> $filename
python3 experiments/basic_sanity_checks/tl_next.py

echo 'Testing TL And' && echo 'Testing TL And' >> $filename
python3 experiments/basic_sanity_checks/tl_and.py

echo 'Testing TL Or' && echo 'Testing TL Or' >> $filename
python3 experiments/basic_sanity_checks/tl_or.py

echo 'Testing Ev(Alw(x))' && echo 'Testing Ev(Alw(x))' >> $filename
python3 experiments/basic_sanity_checks/tl_ev_alw.py

echo 'Testing Alw(Ev(x))' && echo 'Testing Alw(Ev(x))' >> $filename
python3 experiments/basic_sanity_checks/tl_alw_ev.py

echo 'Testing Selection between Atoms for TL Ev'
echo 'Testing Selection between Atoms for TL Ev' >> $filename
python3 experiments/basic_sanity_checks/tl_selection_atom.py

echo 'Testing Selection between TL And and TL Or'
echo 'Testing Selection between TL And and TL Or' >> $filename
python3 experiments/basic_sanity_checks/tl_selection_and_or.py

echo 'Testing Selection between TL Ev and TL Alw'
echo 'Testing Selection between TL Ev and TL Alw' >> $filename
python3 experiments/basic_sanity_checks/tl_selection_ev_alw.py

echo 'Testing Selection between TL Next and TL Until'
echo 'Testing Selection between TL Next and TL Until' >> $filename
python3 experiments/basic_sanity_checks/tl_selection_unt_nxt.py

echo 'Checking RNN Predictions Against Breach'

