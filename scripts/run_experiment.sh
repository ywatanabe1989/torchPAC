#!/bin/bash

function kill_py ()
{ 
    ps aux | grep ywatana+ | egrep 'python' | awk '{ print "kill", $2 }' | sh;
    # clear;
    # watch -d -n 1 free -h;
    # sleep 2;
    # clear
}

function run_experiment() {
    # Preparation
    ps aux | grep ywatana+ | egrep 'python' | awk '{ print "kill", $2 }' | sh &&
    rm ./scripts/main/2024YY* -rf &&
    rm ./scripts/main/RUNNING/ -rf &&    

    # # Define parameter spaces
    # ./scripts/generate_param_spaces.py

    # Start logging CPU / GPU usages
    ./scripts/record_processers.py -i 0.33 -r &

    # PAC calculation with stats recording
    ./scripts/main.py &&

    # Summarize the metrics
    ./scripts/summarize.py &&

    # Close
    kill_py    
    }

run_experiment

# EOF
