import re
import os
import glob
import subprocess
import argparse

def getSearchParameterDicts(config):
    vary_parameters = {
        "transParameter"    : [0.2, 5.0],
        "detWeight"         : [0.2, 5.0],
        "appearanceCost"    : [0.2, 5.0],
        "disappearanceCost" : [0.2, 5.0],
        "divWeight"         : [0.2, 5.0],
        "transWeight"       : [0.2, 5.0]}
    return_configs = []
    return_configs.append(config.copy())
    for key in vary_parameters:
        for value in vary_parameters[key]:
            return_configs.append(config.copy())
            old_value = return_configs[-1][key]
            return_configs[-1][key] = value * float(old_value)
    return return_configs

def readCSVToDict(path):
    csv_file = open(path, "r")
    if csv_file.closed:
        raise RuntimeError("Could not read file: {}".format(path))
    tuple_pattern = re.compile(r'^(.+),(.+)$')
    ret_dict = {}
    for line in csv_file:
        match = tuple_pattern.match(line)
        if match is None:
            raise RuntimeError("Invalid line in config: {}".format(line))
        else:
            ret_dict[match.group(1)] = match.group(2)
    csv_file.close()
    return ret_dict

def saveDictToCSV(path, config_dict):
    csv_file = open(path, "w")
    if csv_file.closed:
        raise RuntimeError("Could not write file: {}".format(path))
    for key in config_dict:
        csv_file.write("{},{}\n".format(key, config_dict[key]))
    csv_file.close()

def isSequenceDir(directory):
    matches = (re.search(r'/\d\d$', directory) is not None)
    return (matches and os.path.isdir(directory))

def getSequences(directory):
    sub_dirs = filter(isSequenceDir, glob.glob("{}/*".format(directory)))
    if len(sub_dirs) == 0:
        raise RuntimeError("no sequences found in directory {}".format(directory))
    return sorted(sub_dirs)

def trackSequenceWithConfig(directory, config_path, tracking_executable):
    directory.rstrip('/')
    seg_directory = "{}_SEG".format(directory)
    res_directory = "{}_RES".format(directory)
    classifier_path = "{}_CFG/classifier.h5".format(directory)
    div_feat_path = "{}_CFG/division_features.txt".format(directory)
    obj_cnt_feat_path = "{}_CFG/object_count_features.txt".format(directory)
    print("call track for directory {}".format(directory))
    return subprocess.call([
        tracking_executable,
        directory, seg_directory, res_directory,
        config_path,
        classifier_path,
        obj_cnt_feat_path,
        div_feat_path])

def evalTracking(directory, eval_exec):
    full_directory = os.path.abspath(directory)
    matches = re.search(r'^(.+)/(\d\d)/*$', full_directory)
    if matches is None:
        raise RuntimeError("cannot get sequence based on path: {}".format(full_directory))
    print("Evaluate dataset {} with sequence {}".format(matches.group(1), matches.group(2)))
    output = subprocess.check_output([eval_exec, matches.group(1), matches.group(2)])
    for line in output.splitlines():
        matches = re.search(r'^TRA measure:\s*(.+)$', line)
        if matches is None:
            raise RuntimeError("cannot parse output of TRAMeasure: {}".format(line))
        return float(matches.group(1))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do a grid search to find the best parameter set for tracking")
    parser.add_argument("-d", dest="dir_base", required=True, type=str, help="Directory of the sequence")
    parser.add_argument("--tra_exec", dest="tra_exec", required=True, type=str, help="Tracking executable")
    parser.add_argument("--eval_exec", dest="eval_exec", required=True, type=str, help="TRAMeasure executable")
    args = parser.parse_args()
    dir_base = args.dir_base.rstrip('/')
    # get the config
    base_config = readCSVToDict("{}_CFG/tracking_config.txt".format(dir_base))
    configs = getSearchParameterDicts(base_config)
    # do the tracking
    results_filename = "{}/tra_measures.txt".format(dir_base)
    results = open(results_filename, "w")
    results.close()
    for index, config in enumerate(configs):
        config_path = "{}/tracking_config_{}.txt".format(dir_base, index)
        saveDictToCSV(config_path, config)
        trackSequenceWithConfig(dir_base, config_path, args.tra_exec)
        tra_measure = evalTracking(dir_base, args.eval_exec)
        results = open(results_filename, "a")
        results.write("{}: {}\n".format(index, tra_measure))
        results.close()

