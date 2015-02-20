// TODO rename read_region_features_from_file
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
#include <boost/shared_ptr.hpp>

// vigra
#include <vigra/impex.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/random_forest.hxx>

// pgmlink
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
  try {
    // check for correct number of arguments
    if (argc < 9) {
      throw isbi::ArgumentError();
    }

    // dataset variables
    isbi::rstrip(argv[1], '/');
    isbi::rstrip(argv[2], '/');
    isbi::rstrip(argv[3], '/');
    std::string raw_dir_str(argv[1]);
    std::string seg_dir_str(argv[2]);
    std::string res_dir_str(argv[3]);
    std::string config_file_path(argv[4]);
    std::string classifier_file_path(argv[5]);
    std::string feature_file_path(argv[6]);
    std::string region_feature_file_path(argv[7]);
    std::string division_feature_file_path(argv[8]);

    isbi::PathType raw_dir = fs::system_complete(raw_dir_str);
    isbi::PathType seg_dir = fs::system_complete(seg_dir_str);
    isbi::PathType res_dir = fs::system_complete(res_dir_str);

    // check validity of dataset variables
    isbi::check_directory(raw_dir, false);
    isbi::check_directory(seg_dir, true);
    isbi::check_directory(res_dir, true);

    // load segmentation classifier
    isbi::RandomForestVectorType rfs;
    if (!(isbi::get_rfs_from_file(rfs, classifier_file_path))) {
      throw std::runtime_error("Could not load Random Forest classifier!");
    }

    // get features used in project
    isbi::StringDataPairVectorType feature_list;
    int read_status = isbi::read_features_from_file(
      feature_file_path,
      feature_list);
    if (read_status == 1) {
      throw std::runtime_error("Could not open file " + feature_file_path);
    }

    // get config
    isbi::TrackingOptions options(config_file_path);
    if (!options.is_legal()) {
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
      read_status = isbi::read_region_features_from_file(
        region_feature_file_path,
        region_feature_list);
      if (read_status == 1) {
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
          "Set to ConsTracking but no CountClassifierForest found");
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
    pgmlink::TraxelStore ts;
    // get the sorted *.tif filenames
    std::vector<isbi::PathType> fn_vec = isbi::get_files(raw_dir, ".tif", true);
    // initialize vector of label images
    std::vector<std::string> labelimage_fn_vec;
    // create the feature and segmentation calculator
    boost::shared_ptr<isbi::FeatureCalculator<2> > feature_calc_ptr(
      new isbi::FeatureCalculator<2>(feature_list));
    isbi::SegmentationCalculator<2> seg_calc(feature_calc_ptr, rfs);
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
    for (
      std::vector<isbi::PathType>::iterator dir_itr = fn_vec.begin();
      dir_itr != fn_vec.end();
      ++dir_itr, ++timestep)
    {
      // for easier readability, create references to traxels for last and
      // current frame:
      isbi::TraxelVectorType& traxels_current_frame = traxels_per_frame[current_frame];
      isbi::TraxelVectorType& traxels_last_frame = traxels_per_frame[1 - current_frame];

      std::string filename(dir_itr->string());
      std::cout << "processing " + filename + " ...\n";
      // TODO is the following assertion deprecated?
      if (dir_itr->extension().string().compare(".tif")) {
        continue;
      }
      if (!vigra::isImage(filename.c_str())) {
        continue;
      }
      // read the image
      DataMatrixType image;
      vigra::importImage(filename, image);
      // calculate the features
      std::cout << "Calculate features" << std::endl;
      isbi::Segmentation<2> segmentation;
      seg_calc.calculate(image, segmentation);
      // save the segmentation results
      std::stringstream segmentation_result_path;
      segmentation_result_path <<  seg_dir_str << "/"
        << "segmentation" << isbi::zero_padding(timestep, 3) << ".h5";
      std::cout << "Save results to " << segmentation_result_path.str()
        << std::endl;
      segmentation.export_hdf5(segmentation_result_path.str());
      std::stringstream labelimage_path;
      labelimage_path << seg_dir_str << "/seg"
        << isbi::zero_padding(timestep, 3) << ".tif";
      std::cout << "Save results to " << labelimage_path.str() << std::endl;
      vigra::exportImage(segmentation.label_image_, labelimage_path.str());
      labelimage_fn_vec.push_back(labelimage_path.str());
      // create traxels
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
      if(dir_itr != fn_vec.begin()) {
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
          t.features["divProb"] = {0.0f};
          pgmlink::add(ts, t);
        }
      }

      current_frame = 1 - current_frame;
    }
    // add remaining traxels (after switching in "last frame") to traxelstore
    for(pgmlink::Traxel& t : traxels_per_frame[1 - current_frame])
      pgmlink::add(ts, t);
    // end of iteration over all filenames/timesteps

    //isbi::print_traxelstore(std::cout, ts);

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
    // create the lineage trees from the events and print them to the stdout
    isbi::Lineage lineage(events);
    std::cout << lineage;
    timestep = 0;
    for (
      std::vector<std::string>::iterator fn_it = labelimage_fn_vec.begin();
      fn_it != labelimage_fn_vec.end();
      fn_it++, timestep++ )
    {
      LabelMatrixType labelimage;
      // read the label image
      vigra::importImage(*fn_it, labelimage);
      // relabel the label image
      lineage.relabel<2>(labelimage, timestep, coordinate_map_ptr);
      // save results
      std::stringstream labelimage_path;
      labelimage_path << res_dir_str << "/mask"
        << isbi::zero_padding(timestep, 3) << ".tif";
      std::cout << "Save results to " << labelimage_path.str() << std::endl;
      vigra::exportImage(labelimage, labelimage_path.str());
    }

    return 0;
  } catch (isbi::ArgumentError& e) {
    std::cout << e.what();
    std::cout << argv[0] << " <image folder> <segmentation folder>"
      << " <result_folder> <config file> <classifier file>"
      << " <feature file> <region feature file> <division feature file>"
      << std::endl;
    return 0;
  } catch (std::runtime_error& e) {
    std::cout << "Program crashed:\n";
    std::cout << e.what();
    std::cout << std::endl;
    return 0;
  }
}
