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
    if (argc < 5) {
      throw isbi::ArgumentError();
    }
    // TODO what about presmoothing?
    // bool presmoothing = true;

    // dataset variables
    isbi::rstrip(argv[1], '/');
    isbi::rstrip(argv[2], '/');
    isbi::rstrip(argv[3], '/');
    isbi::rstrip(argv[4], '/');
    std::string dataset_folder(argv[1]);
    std::string dataset_sequence(argv[2]);
    std::string h5_dir_str(dataset_folder + "/" + dataset_sequence + "_RES");
    std::string config_file_path(argv[3]);
    std::string region_feature_file_path(argv[4]);

    fs::path h5_dir = fs::system_complete(h5_dir_str);

    // check validity of dataset variables
    if (!fs::exists(h5_dir)) {
      throw std::runtime_error(h5_dir_str + " does not exist");
    }
    if (!fs::is_directory(h5_dir)) {
      throw std::runtime_error(h5_dir_str + " is not a directory");
    }

    // make unary for copying only filepaths that contain *.h5
    boost::function<bool (fs::directory_entry&)> h5_chooser = bind(
      isbi::contains_substring_boost_path, _1, ".h5");

    // get the region features that should be calculated
    std::vector<std::string> region_feature_list;
    int read_status = isbi::read_region_features_from_file(
      region_feature_file_path,
      region_feature_list);
    if (read_status == 1) {
      throw std::runtime_error(
        "Could not open file " + region_feature_file_path);
    }

    // get config
    isbi::TrackingOptions options(config_file_path);
    if(!options.is_legal()) {
      throw std::runtime_error("Bad options for tracking");
    }

    //=========================================================================
    // Segmentation
    //=========================================================================
    // run segmentation for all tif files in the given folder
    // write results to <dataset_folder>/<dataset_sequence_segmentation>

    int timestep = 0;
    pgmlink::TraxelStore ts;
    // sort filenames
    std::vector<fs::path> fn_vec;
    isbi::copy_if_own(
      fs::directory_iterator(h5_dir),
      fs::directory_iterator(),
      std::back_inserter(fn_vec),
      h5_chooser);
    std::sort(fn_vec.begin(), fn_vec.end());
    // create the traxel extractor
    isbi::TraxelExtractor<2> traxel_extractor(0, 100, 0);

    // iterate over the filenames TODO timestep counting not correct
    for (
      std::vector<fs::path>::iterator dir_itr = fn_vec.begin();
      dir_itr != fn_vec.end();
      ++dir_itr, ++timestep)
    {
      std::string filename(dir_itr->string());
      std::cout << "processing " + filename + " ...\n";
      // load the labelimage
      isbi::Segmentation<2> segmentation;
      segmentation.read_hdf5(filename, true);
      // create an empty image
      DataMatrixType empty_image(segmentation.label_image_.shape());
      // create traxels and add them to the traxelstore
      traxel_extractor.extract(
        segmentation,
        empty_image,
        timestep,
        region_feature_list,
        ts);
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
    timestep = 0;
    for (
      std::vector<fs::path>::iterator dir_itr = fn_vec.begin();
      dir_itr != fn_vec.end();
      ++dir_itr, ++timestep)
    {
      std::string filename(dir_itr->string());
      std::cout << "relabel " << filename << std::endl;
      // load the segmentation once again
      isbi::Segmentation<2> segmentation;
      segmentation.read_hdf5(filename);
      // relabel the label image
      lineage.relabel<2>(segmentation.label_image_, timestep);
      // save the relabeled image
      std::stringstream mask_image_path;
      mask_image_path << h5_dir_str << "/" << "mask"
        << isbi::zero_padding(timestep, 3) << ".tif";
      std::cout << "Save results to " << mask_image_path.str() << std::endl;
      vigra::exportImage(
        vigra::srcImageRange(UShortMatrixType(segmentation.label_image_)),
        vigra::ImageExportInfo(mask_image_path.str().c_str()));
    }

    return 0;

  } catch (isbi::ArgumentError& e) {
    std::cout << e.what();
    std::cout << "Usage: ";
    std::cout << argv[0] << " folder sequence config_path" << std::endl;
    return 0;
  } catch (std::runtime_error& e) {
    std::cout << "Program crashed:\n";
    std::cout << e.what();
    std::cout << std::endl;
    return 0;
  }
}
