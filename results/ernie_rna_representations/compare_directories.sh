#!/usr/bin/env bash

# This script compares the md5 checksums of files in 'test_seqs' and 'test_seqs_back'.
# It assumes that both directories exist in the current working directory and that 
# they contain files with matching names.

# Exit if either directory doesn't exist
if [ ! -d "test_seqs" ] || [ ! -d "test_seqs_back" ]; then
    echo "Error: Directories test_seqs and/or test_seqs_back do not exist."
    exit 1
fi

# Loop over files in test_seqs
for file in test_seqs/*; do
    # Extract filename without path
    filename=$(basename "$file")

    # Check if corresponding file exists in test_seqs_back
    if [ ! -f "test_seqs_back/$filename" ]; then
        echo "Warning: test_seqs_back/$filename does not exist."
        continue
    fi

    # Compute MD5 checksums for the two files
    md5_test=$(md5sum "test_seqs/$filename" | cut -d ' ' -f 1)
    md5_back=$(md5sum "test_seqs_back/$filename" | cut -d ' ' -f 1)

    # Compare the checksums
    if [ "$md5_test" = "$md5_back" ]; then
        echo "$filename: OK"
    else
        echo "$filename: DIFFER"
    fi
done

