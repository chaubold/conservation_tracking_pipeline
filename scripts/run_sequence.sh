#!/bin/bash

# Arguments:
# -d <Dataset Name>
# -s <Sequence Name>
# [-t] Only tracking
# [-g] Debug with gdb
# [-3] use 3D tracking

tflag=0
dim3flag=0
gflag=0

while getopts "d:s:t3g" flag; do
  case "${flag}" in
    d) dataset="${OPTARG}";;
    s) sequence="${OPTARG}";;
    t) tflag=1;;
    g) gflag=1;;
    3) dim3flag=1;;
  esac
done

ISBI_DATA_DIR="/mnt/data1/chaubold/cell_tracking_challenge_15"
ISBI_DATA_DIR="${ISBI_DATA_DIR}/${dataset}/${sequence}"

ISBI_CFG_DIR="/mnt/data1/chaubold/cell_tracking_challenge_15"
ISBI_CFG_DIR="${ISBI_CFG_DIR}/${dataset}/${sequence}_CFG"

ISBI_EXECUTABLE="/mnt/data1/chaubold/cell_tracking_challenge_15"
ISBI_EXECUTABLE="${ISBI_EXECUTABLE}/isbi_tracking_challenge_2013_pipeline-build"

if ((${tflag})); then
  ISBI_EXECUTABLE="${ISBI_EXECUTABLE}/tracking"
else
  ISBI_EXECUTABLE="${ISBI_EXECUTABLE}/isbi_pipeline"
fi

if ((${dim3flag})); then
  ISBI_EXECUTABLE="${ISBI_EXECUTABLE}3d"
fi

if (($gflag)); then
  ISBI_EXECUTABLE="gdb --args ${ISBI_EXECUTABLE}"
fi

if (($tflag)); then
  ${ISBI_EXECUTABLE} "${ISBI_DATA_DIR}" "${ISBI_DATA_DIR}_SEG" "${ISBI_DATA_DIR}_RES" "${ISBI_CFG_DIR}/tracking_config.txt" "${ISBI_CFG_DIR}/classifier.h5" "${ISBI_CFG_DIR}/object_count_features.txt" "${ISBI_CFG_DIR}/division_features.txt"
else
  ${ISBI_EXECUTABLE} "${ISBI_DATA_DIR}" "${ISBI_DATA_DIR}_SEG" "${ISBI_DATA_DIR}_RES" "${ISBI_CFG_DIR}/tracking_config.txt" "${ISBI_CFG_DIR}/classifier.h5" "${ISBI_CFG_DIR}/features.txt" "${ISBI_CFG_DIR}/object_count_features.txt" "${ISBI_CFG_DIR}/division_features.txt"
fi
