#!/bin/bash

ds="N2D-SIM"
seq="04"

seqs="01-02-03-04-05-06"

echo "Segmentation and Tracking for ${ds}-${seq}"

# For each dataset, we trained a classifier for pixelwise segmentation. The 
# parameters of each classifier and the features used for classification
# are stored in SW/dependencies/DATASET/*_classifier.h5 and
# .../*_features.txt, respectively. The parameters needed for 
# tracking are stored in the same folder as *_config.txt


# On some machines, this executable crashes with
#   locale::facet::_S_create_c_locale name not valid
# In this case, run 
#   export LC_ALL="C"
# on the command line before executing this script.

./isbi_pipeline "../${ds}" "${seq}" "dependencies/${ds}/${ds}_${seqs}_config.txt" "dependencies/${ds}/${ds}_${seqs}_classifier.h5" "dependencies/${ds}/${ds}_${seqs}_features.txt"
