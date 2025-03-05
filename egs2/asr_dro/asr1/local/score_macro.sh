#!/bin/bash

if [ $USER_SCTK_INSTALL_DIR]; then
    INSTALL_DIR=$USER_SCTK_INSTALL_DIR
else
    INSTALL_DIR="../../../espnet/tools/installers/"
fi
# Usage: ./score_macro.sh --exp_dir [exp_dir]

# Function to check if sclite is available in the custom installation directory
check_sclite() {
    if [ ! -f "$INSTALL_DIR/bin/sclite" ]; then
        echo "sclite could not be found in $INSTALL_DIR. Installing SCTK..."
        install_sctk
    else
        echo "sclite is already installed in $INSTALL_DIR."
        export PATH="$INSTALL_DIR/bin:$PATH"
    fi
}

# Function to install SCTK
install_sctk() {
    if [ ! -d "$INSTALL_DIR" ]; then
        mkdir -p "$INSTALL_DIR"
        cd "$INSTALL_DIR" || exit
        
        git clone https://github.com/usnistgov/SCTK.git .
        
        make config
        make all
        make check
        make install
        make doc
        
        export PATH="$INSTALL_DIR/bin:$PATH"
        echo "SCTK installed successfully in $INSTALL_DIR."
        
    else
        echo "SCTK is already installed in $INSTALL_DIR."
    fi
}

check_sclite

# Put default exp_dir here
exp_dir="/scr/exp001"

# Parse command line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --exp_dir) exp_dir="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Convert files
convert_file() {
    local file_path=$1
    local mapping_file="local/jpn_orgjpn.txt"
    local output_file="${file_path%.*}_mod.trn"
    
    # Replace contents as per requirements
    sed -e 's/mexico-el/mexico/g' \
        -e 's/googlei18n-tts/googlei18n/g' \
        -e 's/googlei18n-asr/googlei18n/g' \
        -e 's/m-ailabs/m/g' \
        -e 's/M-AILABS/m/g' "$file_path" > "$output_file"
    
    while IFS= read -r line; do
        IFS='/' read -ra ADDR <<< "$line"
        old="${ADDR[0]}"
        new="${ADDR[1]}"
        
        sed -i "s|${old}|${new}|g" "$output_file"
    done < "$mapping_file"
    
    echo "$output_file"
}

echo "Converting hyp.trn and ref.trn files..."
hyp_mod=$(convert_file "${exp_dir}/hyp.trn")
ref_mod=$(convert_file "${exp_dir}/ref.trn")

echo "Converted files saved as ${hyp_mod} and ${ref_mod}"

# Run sclite
echo "Running sclite..."
result_file="${exp_dir}/result_mod.txt"
stat_file="${exp_dir}/stats.txt"
sclite -r "${ref_mod}" trn -h "${hyp_mod}" trn -i rm -o all stdout > "${result_file}"
echo "Sclite results saved to ${result_file}"

# Extract and print summary statistics before cleaning
echo "Summary Statistics:"
echo ${result_file}
grep -e Avg -e SPKR -m 2 "${result_file}"

echo "Cleaning results_mod.txt file..."
RESULT_MOD_FILE="${exp_dir}/result_mod.txt"
sed -i 's/[*+]/ /g' "$RESULT_MOD_FILE"
echo "Cleaned ${RESULT_MOD_FILE}."

# Extract and print summary statistics before cleaning
echo "Summary Statistics After Cleaning:"
grep -e Avg -e SPKR -m 2 "${result_file}"

echo "Computing SD across languages..."
python local/multi_superb_score_sd.py --result_file "${result_file}"
