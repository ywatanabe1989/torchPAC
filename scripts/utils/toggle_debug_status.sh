#!/bin/bash

toggle_debug_status() {
    # Path to the IS_DEBUG.yaml file
    DEBUG_FILE="./config/IS_DEBUG.yaml"

    # Check if the file exists
    if [ ! -f "$DEBUG_FILE" ]; then
        echo "Error: $DEBUG_FILE does not exist."
        return 1
    fi

    # Read the current debug status
    CURRENT_STATUS=$(grep 'IS_DEBUG:' $DEBUG_FILE | awk '{print $2}')

    # Toggle the debug status
    if [ "$CURRENT_STATUS" = "true" ]; then
        NEW_STATUS="false"
    else
        NEW_STATUS="true"
    fi

    # Update the file with the new status
    sed -i "s/IS_DEBUG: $CURRENT_STATUS/IS_DEBUG: $NEW_STATUS/" $DEBUG_FILE

    # Output the new status
    echo "Debug status toggled to $NEW_STATUS in $DEBUG_FILE"
}

toggle_debug_status

# EOF
