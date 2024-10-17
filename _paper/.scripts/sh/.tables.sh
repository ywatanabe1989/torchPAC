#!/bin/bash

function gather_tables() {
    rm ./src/tables/.tex/.All_Tables.tex -f > /dev/null 2>&1
    for table_tex in ./src/tables/.tex/Table_*.tex; do
        fname="${table_tex%.tex}"
        echo "\input{${fname}}" >> ./src/tables/.tex/.All_Tables.tex
    done
}

function csv2tex() {
    yes | rm ./src/tables/.tex/*.tex > /dev/null 2>&1

    csv_dir="./src/tables/"
    legend_dir="./src/tables/"
    
    output_dir="./src/tables/.tex/"

    mkdir -p $csv_dir $output_dir > /dev/null

    ii=0
    for csv_file in "$csv_dir"*.csv; do
        base_name=$(basename "$csv_file" .csv)
        table_id=$(basename "$csv_file" .csv | grep -oP '(?<=Table_ID_)[^\.]+' | tr '[:upper:]' '[:lower:]')

        caption_file=${legend_dir}${base_name}.tex
        width=$(grep -oP '(?<=width=)[0-9.]+\\textwidth' "$caption_file")  # [REVISED]

        tgt_file="$output_dir${base_name}.tex"

        # Determine the number of columns in the CSV file
        num_columns=$(head -n 1 "$csv_file" | awk -F, '{print NF}')

        fontsize="\\small"
        # Create the LaTeX document
        {
            echo "\\pdfbookmark[2]{ID ${table_id}}{id_${table_id}}"
            echo "\\begin{table*}[htbp]"
            echo "\\centering"
            echo "$fontsize"
            echo "\\begin{tabular}{*{$num_columns}{c}}"
            echo "\\toprule"
            # Header
            head -n 1 "$csv_file" | {
                IFS=',' read -ra headers
                for header in "${headers[@]}"; do
                    echo -n "\\textbf{\\thead{${header}}} &" # [REVISED]                    
                done
                echo "\\\\"
            }
            echo "\\midrule"

            # Table contents
            awk 'BEGIN {FPAT = "([^,]+)|(\"[^\"]+\")"; OFS=" & "; row_count=0} 
                NR>1 {
                    if (row_count % 2 == 1) {print "\\rowcolor{lightgray}"}
                    $1=$1; print $0"\\\\"
                    row_count++
                }' "$csv_file"            
            echo "\\bottomrule"
            echo "\\end{tabular}"
            echo "\\captionsetup{width=\textwidth}"
            echo "\\input{${legend_dir}Table_ID_${table_id}}"
            echo ""
            echo "\\label{tab:${table_id}}"            
            echo "\\end{table*}"
            echo "\\restoregeometry"

        } > $tgt_file
    done
}

csv2tex
gather_tables

## EOF
