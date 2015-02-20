// stl
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <map>
#include <sstream>
#include <utility>

// boost
#include <boost/filesystem.hpp>
#include <boost/bind.hpp>
#include <boost/function.hpp>

// vigra
#include <vigra/impex.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/random_forest.hxx>

// pgmlink
#include <pgmlink/traxels.h>
#include <pgmlink/tracking.h>

// own
#include "common.h"
#include "pipeline_helpers.hxx"
#include "segmentation.hxx"
#include "traxel_extractor.hxx"
#include "lineage.hxx"
#include "division_feature_extractor.hxx"


// Aliases for convenience
namespace fs = boost::filesystem;
namespace isbi = isbi_pipeline;

// Vigra MultiArray types
typedef vigra::MultiArray<2, isbi::DataType> DataMatrixType;
typedef vigra::MultiArray<2, isbi::LabelType> LabelMatrixType;

int main(int argc, char** argv) {
  // arg 1: dataset folder
  try {
    // check for correct number of arguments
    if (argc < 8) {
      throw isbi::ArgumentError();
    }
    // TODO what about presmoothing?
    // bool presmoothing = true;

    // dataset variables
    isbi::rstrip(argv[1], '/');
    isbi::rstrip(argv[2], '/');
    isbi::rstrip(argv[3], '/');
    std::string raw_dir_str(argv[1]);
    std::string seg_dir_str(argv[2]);
    std::string res_dir_str(argv[3]);
    std::string config_file_path(argv[4]);
    std::string classifier_file_path(argv[5]);
    std::string region_feature_file_path(argv[6]);
    std::string division_feature_file_path(argv[7]);

    isbi::PathType raw_dir = fs::system_complete(raw_dir_str);
    isbi::PathType seg_dir = fs::system_complete(seg_dir_str);
    isbi::PathType res_dir = fs::system_complete(seg_dir_str);

    // check validity of dataset variables
    isbi::check_directory(raw_dir, false);
    isbi::check_directory(seg_dir, false);
    isbi::check_directory(res_dir, true);

    // get config
    isbi::TrackingOptions options(config_file_path);
    if(!options.is_legal()) {
      throw std::runtime_error("Bad options for tracking");
    }

    // get the tracker name
    const std::string& tracker_name = options.get_option<std::string>("tracker");
    // check if we have conservation tracking - get the random forests and
    // region features if applicable
    std::vector<std::string> region_feature_list;
    isbi::RandomForestVectorType region_feature_rfs;
    unsigned int max_object_num = 0;
    std::vector<std::string> division_feature_list;
    isbi::RandomForestVectorType division_feature_rfs;
    size_t template_size = 0;
    if (!tracker_name.compare("ConsTracking")) {
      // get the region features used in project
      bool read_status = isbi::read_region_features_from_file(
        region_feature_file_path,
        region_feature_list);
      if (read_status) {
        throw std::runtime_error(
          "Could not open file " + region_feature_file_path);
      }
      // get the rf
      read_status = isbi::get_rfs_from_file(
        region_feature_rfs,
        classifier_file_path,
        "CountClassification/ClassifierForests/Forest",
        1,
        4);
      // get the max_object_num
      max_object_num = options.get_option<int>("maxObj");
      if (!read_status) {
        throw std::runtime_error(
          "Set to ConsTracking but no ClassifierForest found");
      }
      // get the division features used in project
      read_status = isbi::read_region_features_from_file(
        division_feature_file_path,
        division_feature_list);
      if (read_status == 1) {
        throw std::runtime_error(
          "Could not open file " + division_feature_file_path);
      }
      // get the rf
      read_status = isbi::get_rfs_from_file(
        division_feature_rfs,
        classifier_file_path,
        "DivisionDetection/ClassifierForests/Forest",
        1,
        4);
      if (!read_status) {
        throw std::runtime_error(
          "Set to ConsTracking but no DivisionClassifierForest found");
      }
      // get the template size
      template_size = options.get_option<int>("templateSize");
    }
    // get an empty coordinate map
    isbi::CoordinateMapPtrType coordinate_map_ptr(
      new isbi::CoordinateMapType);

    //=========================================================================
    // Segmentation
    //=========================================================================
    // run segmentation for all tif files in the given folder
    // write results to <dataset_folder>/<dataset_sequence_segmentation>

    int timestep = 0;
    isbi::TraxelStoreType ts;
    // sort filenames
    std::vector<isbi::PathType> raw_fn_vec =
      isbi::get_files(raw_dir, ".tif", true);
    std::vector<isbi::PathType> seg_fn_vec =
      isbi::get_files(seg_dir, ".tif", true);
    // check if they are of the same length
    if (raw_fn_vec.size() != seg_fn_vec.size()) {
      throw std::runtime_error(
        "The raw images do not match the segmentation images");
    }
    // create the traxel extractor
    isbi::TraxelExtractor<2> traxel_extractor(
      max_object_num,
      region_feature_list,
      region_feature_rfs,
      options.get_option<int>("borderWidth"),
      options.get_option<int>("size_range_0"),
      options.get_option<int>("size_range_1"));

    // create division feature extractor
    isbi::DivisionFeatureExtractor<2, isbi::LabelType> div_feature_extractor(template_size);
    // storage for all traxels of a frame
    isbi::TraxelVectorType traxels_per_frame[2];
    size_t current_frame = 0;

    // iterate over the filenames TODO timestep counting not correct
    std::vector<isbi::PathType>::iterator raw_it, seg_it;
    for (
      raw_it = raw_fn_vec.begin(), seg_it = seg_fn_vec.begin();
      raw_it != raw_fn_vec.end();
      raw_it++, seg_it++, timestep++)
    {
      // for easier readability, create references to traxels for last and
      // current frame:
      isbi::TraxelVectorType& traxels_current_frame = traxels_per_frame[current_frame];
      isbi::TraxelVectorType& traxels_last_frame = traxels_per_frame[1 - current_frame];

      std::string raw_filename(raw_it->string());
      std::string seg_filename(seg_it->string());
      std::cout << "processing image " + raw_filename + " ...\n";
      std::cout << "with segmentation " + seg_filename + " ...\n";
      // load the raw image
      DataMatrixType image;
      vigra::importImage(raw_filename, image);
      // load the label image
      isbi::Segmentation<2> segmentation;
      vigra::importImage(seg_filename, segmentation.label_image_);
      // read label count
      isbi::LabelType min, max;
      segmentation.label_image_.minmax(&min, &max);
      segmentation.label_count_ = max;
      // create traxels and add them to the traxelstore
      traxel_extractor.extract(
        segmentation,
        image,
        timestep,
        traxels_current_frame);
      // get the coordinate map
      if (!tracker_name.compare("ConsTracking")) {
        isbi::fill_coordinate_map(
          traxels_current_frame,
          segmentation.label_image_,
          coordinate_map_ptr);
      }
      // extract division features if this was not the first frame
      if(raw_it != raw_fn_vec.begin()) {
        if(!tracker_name.compare("ConsTracking")) {
          div_feature_extractor.extract(
            traxels_last_frame,
            traxels_current_frame,
            segmentation.label_image_);
          div_feature_extractor.compute_div_prob(
            traxels_last_frame,
            division_feature_list,
            division_feature_rfs);
        }
        // add all traxels of last frame to traxelstore
        for(pgmlink::Traxel& t : traxels_last_frame) {
          pgmlink::add(ts, t);
        }
      }
      current_frame = 1 - current_frame;
    }
    // add remaining traxels (after switching in "last frame") to traxelstore
    for(pgmlink::Traxel& t : traxels_per_frame[1 - current_frame]) {
      t.features["divProb"] = {0.0f};
      pgmlink::add(ts, t);
    }

    // isbi::print_traxelstore(std::cout, ts);

    //=========================================================================
    // track!
    //=========================================================================
    isbi::EventVectorVectorType events = isbi::track(
      ts,
      options,
      coordinate_map_ptr);

    //=========================================================================
    // handle results
    //=========================================================================
    // create lineage object
    isbi::Lineage lineage(events);
    std::cout << lineage;
    // relabel the image
    timestep = 0;
    for (
      std::vector<isbi::PathType>::iterator seg_it = seg_fn_vec.begin();
      seg_it != seg_fn_vec.end();
      seg_it++, timestep++)
    {
      std::string filename(seg_it->string());
      std::cout << "relabel " << filename << std::endl;
      // load the segmentation once again
      isbi::Segmentation<2> segmentation;
      vigra::importImage(filename, segmentation.label_image_);
      // relabel the label image
      lineage.relabel<2>(segmentation.label_image_, timestep, coordinate_map_ptr);
      // save the relabeled image
      std::stringstream res_image_path;
      res_image_path << res_dir_str << "/" << "mask"
        << isbi::zero_padding(timestep, 3) << ".tif";
      std::cout << "Save results to " << res_image_path.str() << std::endl;
      vigra::exportImage(segmentation.label_image_, res_image_path.str());
    }

    return 0;

  } catch (isbi::ArgumentError& e) {
    std::cout << e.what();
    std::cout << "Usage: ";
    std::cout << argv[0] << " <image folder> <segmentation folder>"
      << " <result_folder> <config file> <classifier file>"
      << " <region feature file> <division feature file>" << std::endl;
    return 0;
  } catch (std::runtime_error& e) {
    std::cout << "Program crashed:\n";
    std::cout << e.what();
    std::cout << std::endl;
    return 0;
  }
}
