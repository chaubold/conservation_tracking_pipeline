#!/bin/bash

ds="N2DL-HeLa"
seq="01"

seqs="01-02"

echo "Segmentation and Tracking for ${ds}-${seq}"

# For each dataset, we trained a classifier for pixelwise segmentation. The 
# parameters of each classifier and the features used for classification
# are stored in SW/dependencies/DATASET/*_classifier.h5 and
# .../*_features.txt, respectively. The parameters needed for 
# tracking are stored in the same folder as *_config.txt


./isbi_pipeline "../${ds}" "${seq}" "dependencies/${ds}/${ds}_${seqs}_config.txt" "dependencies/${ds}/${ds}_${seqs}_classifier.h5" "dependencies/${ds}/${ds}_${seqs}_features.txt"
