#!/bin/bash
# Time-stamp: "2024-11-04 14:56:10 (ywatanabe)"
# File: ./torchPAC/scripts/run_experiment.sh


LOG_FILE="${0%.sh}.log"

usage() {
    echo "Usage: $0 [-h|--help]"
    echo "Options:"
    echo " -h, --help Display this help message"
    echo
    echo "Example:"
    echo " $0"
    exit 1
}

kill_py() {
    # Terminates all Python processes for the current user
    ps aux | grep "$(whoami)" | grep 'python' | awk '{ print "kill", $2 }' | sh
}

initialize() {
    sudo nvidia-smi -pm 1
    # Preparation
    kill_py
    rm -rf \
       ./scripts/main/FINISHED* \
       ./scripts/main/RUNNING/ \
       /tmp/mngs/processer_usages.csv 
}

run_experiment() {
    # Executes the experiment workflow
    #
    # Usage:
    # run_experiment

    initialize

    # Start logging CPU / GPU usages
    ./scripts/log_processer_usages.py -i 0.33 -r &

    # PAC calculation with stats recording
    ./scripts/main.py &&

    # Plot the metrics
    ./scripts/plot_parallel.py &&

    # Close
    kill_py
}

main() {
    while [[ "$#" -gt 0 ]]; do
        case $1 in
            -h|--help) usage ;;
            *) echo "Unknown parameter passed: $1"; usage ;;
        esac
        shift
    done

    run_experiment
}

{ main "$@"; } 2>&1 | tee "$LOG_FILE"

# EOF
