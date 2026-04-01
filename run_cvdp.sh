#!/bin/bash

for item in /glade/work/richling/CVDP-python-dev/CVDP-python/test_config_yamls/*; do
    #echo "yaml file $item"
    # Get just the filename
    filename=$(basename "$item")

    # Strip prefix and suffix
    content=${filename#example_config_}   # removes 'example_config_' from the front
    content=${content%.yaml}              # removes '.yaml' from the end

    #echo "Extracted content: $content"
    echo "$item" "$content"
    #time python /glade/work/richling/CVDP-python-dev/CVDP-python/cvdp/cli.py "$content"
    time python cli.py -c "$item" "$content"
done