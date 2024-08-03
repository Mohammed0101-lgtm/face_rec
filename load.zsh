#!/bin/zsh

# Define the source base directory and the destination directory
SOURCE_BASE_DIR="/Users/mac/Downloads/imdb_crop"
DEST_DIR="/Users/mac/python/training_dataset"

# Loop through the numbers 00 to 99
for i in {00..99}; do
    SOURCE_DIR="${SOURCE_BASE_DIR}/${i}"
    
    # Check if the source directory exists
    if [ -d "$SOURCE_DIR" ]; then
        echo "Copying directory ${SOURCE_DIR} to ${DEST_DIR}"
        ./load "$SOURCE_DIR" "$DEST_DIR"
    else
        echo "Source directory ${SOURCE_DIR} does not exist."
    fi
done

echo "All directories copied."
