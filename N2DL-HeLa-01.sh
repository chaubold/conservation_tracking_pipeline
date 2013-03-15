echo "Segmentation and Tracking for N2DL-HeLa-01"
config_dir="dependencies"
sequence="01"
dataset="N2DLHeLa"

./isbi_pipeline "../$dataset" "$sequence" "$config_dir/$dataset-$sequence.txt" "$config_dir/$dataset-$sequence.ilp"



