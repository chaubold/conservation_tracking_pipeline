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
#include "pipeline_helpers.hxx"
#include "segmentation.hxx"
#include "traxel_extractor.hxx"
#include "lineage.hxx"


// Aliases for convenience
namespace fs = boost::filesystem;
namespace isbi = isbi_pipeline;

// Scalar types
typedef isbi::FeatureType DataType;
// Container types of the standard library
typedef std::vector<std::pair<std::string, double> > StringDoublePairVectorType;
typedef std::map<std::string, double> StringDoubleMapType;
// Vigra MultiArray types
typedef vigra::MultiArray<2, DataType> DataMatrixType;
typedef vigra::MultiArray<2, unsigned short> UShortMatrixType;
// Further vigra types
typedef vigra::RandomForest<unsigned> RandomForestType;

int main(int argc, char** argv) {
  // arg 1: dataset folder
  try {
    // check for correct number of arguments
    if (argc < 7) {
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

    fs::path raw_dir = fs::system_complete(raw_dir_str);
    fs::path seg_dir = fs::system_complete(seg_dir_str);
    fs::path res_dir = fs::system_complete(seg_dir_str);

    // check validity of dataset variables
    if (!fs::exists(raw_dir)) {
      throw std::runtime_error(raw_dir_str + " does not exist");
    } else if (!fs::is_directory(raw_dir)) {
      throw std::runtime_error(raw_dir_str + " is not a directory");
    }
    if (!fs::exists(seg_dir)) {
      throw std::runtime_error(seg_dir_str + " does not exist");
    } else if (!fs::is_directory(seg_dir)) {
      throw std::runtime_error(seg_dir_str + " is not a directory");
    }
    if (!fs::exists(res_dir)) {
      if (!fs::create_directory(res_dir)) {
        throw std::runtime_error("Could not create directory: " + res_dir_str);
      }
    } else if (!fs::is_directory(res_dir)) {
      throw std::runtime_error(res_dir_str + " is not a directory");
    }

    // make unary for copying only filepaths that contain *.tif
    boost::function<bool (fs::directory_entry&)> tif_chooser = bind(
      isbi::contains_substring_boost_path, _1, ".tif");

    // get config
    isbi::TrackingOptions options(config_file_path);
    if(!options.is_legal()) {
      throw std::runtime_error("Bad options for tracking");
    }

    // check if we have conservation tracking - get the random forests and
    // region features if applicable
    std::vector<std::string> region_feature_list;
    std::vector<RandomForestType> region_feature_rfs;
    unsigned int max_object_num = 0;
    if (!options.get_option<std::string>("tracker").compare("ConsTracking")) {
      // get the region features used in project
      bool read_status = isbi::read_region_features_from_file(
        region_feature_file_path,
        region_feature_list);
      if (!read_status) {
        throw std::runtime_error(
          "Could not open file " + region_feature_file_path);
      }
      // get the rf
      read_status = isbi::get_rfs_from_file(
        region_feature_rfs,
        classifier_file_path,
        "CountClassification/ClassifierForests/Forest",
        4,
        1);
      // get the max_object_num
      max_object_num = options.get_option<int>("maxObj");
      if (!read_status) {
        throw std::runtime_error(
          "Set to ConsTracking but no ClassifierForest found");
      }
    }

    //=========================================================================
    // Segmentation
    //=========================================================================
    // run segmentation for all tif files in the given folder
    // write results to <dataset_folder>/<dataset_sequence_segmentation>

    int timestep = 0;
    pgmlink::TraxelStore ts;
    // sort filenames
    std::vector<fs::path> raw_fn_vec;
    isbi::copy_if_own(
      fs::directory_iterator(raw_dir),
      fs::directory_iterator(),
      std::back_inserter(raw_fn_vec),
      tif_chooser);
    std::sort(raw_fn_vec.begin(), raw_fn_vec.end());
    std::vector<fs::path> seg_fn_vec;
    isbi::copy_if_own(
      fs::directory_iterator(seg_dir),
      fs::directory_iterator(),
      std::back_inserter(seg_fn_vec),
      tif_chooser);
    std::sort(seg_fn_vec.begin(), seg_fn_vec.end());
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
      options.get_option<int>("border"),
      options.get_option<int>("size_from"),
      options.get_option<int>("size_to"));

    // iterate over the filenames TODO timestep counting not correct
    std::vector<fs::path>::iterator raw_it, seg_it;
    for (
      raw_it = raw_fn_vec.begin(), seg_it = seg_fn_vec.begin();
      raw_it != raw_fn_vec.end();
      raw_it++, seg_it++, timestep++)
    {
      std::string raw_filename(raw_it->string());
      std::string seg_filename(seg_it->string());
      std::cout << "processing image " + raw_filename + " ...\n";
      std::cout << "with segmentation " + seg_filename + " ...\n";
      // load the raw image
      vigra::MultiArray<2, unsigned> image;
      isbi::read_tif_image(raw_filename, image);
      // load the label image
      isbi::Segmentation<2> segmentation;
      isbi::read_tif_image(seg_filename, segmentation.label_image_);
      // read label count
      unsigned int min, max;
      segmentation.label_image_.minmax(&min, &max);
      segmentation.label_count_ = max;
      // create traxels and add them to the traxelstore
      traxel_extractor.extract(segmentation, image, timestep, ts);
    }

    //=========================================================================
    // track!
    //=========================================================================
    isbi::EventVectorVectorType events = isbi::track(ts, options);

    //=========================================================================
    // handle results
    //=========================================================================
    // create lineage object
    isbi::Lineage lineage(events);
    std::cout << lineage;
    // relabel the image
    timestep = 0;
    for (
      std::vector<fs::path>::iterator seg_it = seg_fn_vec.begin();
      seg_it != seg_fn_vec.end();
      seg_it++, timestep++)
    {
      std::string filename(seg_it->string());
      std::cout << "relabel " << filename << std::endl;
      // load the segmentation once again
      isbi::Segmentation<2> segmentation;
      isbi::read_tif_image(filename, segmentation.label_image_);
      // relabel the label image
      lineage.relabel<2>(segmentation.label_image_, timestep);
      // save the relabeled image
      std::stringstream res_image_path;
      res_image_path << res_dir_str << "/" << "mask"
        << isbi::zero_padding(timestep, 3) << ".tif";
      std::cout << "Save results to " << res_image_path.str() << std::endl;
      isbi::save_tif_image(res_image_path.str(), segmentation.label_image_);
    }

    return 0;

  } catch (isbi::ArgumentError& e) {
    std::cout << e.what();
    std::cout << "Usage: ";
    std::cout << argv[0] << " <image folder> <segmentation folder>"
      << " <result_folder> <config file> <classifier file>"
      << " <region feature file>" << std::endl;
    return 0;
  } catch (std::runtime_error& e) {
    std::cout << "Program crashed:\n";
    std::cout << e.what();
    std::cout << std::endl;
    return 0;
  }
}
