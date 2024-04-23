#!/bin/bash

function kill_py ()
{ 
    ps aux | grep ywatana+ | egrep 'python' | awk '{ print "kill", $2 }' | sh;
    clear;
    # watch -d -n 1 free -h;
    # sleep 2;
    # clear
}

function run_experiment() {
    rm ./scripts/main -rf
    ./scripts/generate_param_spaces.py
    ./scripts/record_processers.py -i 0.33 -r &
    ./scripts/main.py
    ./scripts/summrize.py
    kill_py
    }

run_experiment

# EOF
