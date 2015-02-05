/* TODO
 * do label image calculations with unsigned short?
 */

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
#include <vigra/multi_array.hxx>
#include <vigra/random_forest.hxx>

// pgmlink
#include <pgmlink/traxels.h>
#include <pgmlink/tracking.h>

// own
#include "pipeline_helpers.hxx"
#include "segmentation.hxx"
#include "lineage.hxx"


// Aliases for convenience
namespace fs = boost::filesystem;
namespace isbi = isbi_pipeline;

// Typedefs for convenience
// Scalar types
typedef isbi::FeatureType DataType;
// Container types of the standard library
typedef std::vector<std::pair<std::string, double> > StringDoublePairVectorType;
typedef std::map<std::string, double> StringDoubleMapType;
// Vigra MultiArray types
typedef vigra::MultiArray<2, DataType> DataMatrixType;
typedef vigra::MultiArray<2, unsigned> UIntMatrixType;
typedef vigra::MultiArray<2, unsigned short> UShortMatrixType;
// Further vigra types
typedef vigra::RandomForest<unsigned> RandomForestType;

int main(int argc, char** argv) {
  try {
    // check for correct number of arguments
    if (argc < 6) {
      throw isbi::ArgumentError();
    }

    // dataset variables
    isbi::rstrip(argv[1], '/');
    isbi::rstrip(argv[2], '/');
    isbi::rstrip(argv[3], '/');
    isbi::rstrip(argv[4], '/');
    isbi::rstrip(argv[5], '/');
    std::string dataset_folder(argv[1]);
    std::string dataset_sequence(argv[2]);
    std::string tif_dir_str(dataset_folder + "/" + dataset_sequence);
    std::string res_dir_str(dataset_folder + "/" + dataset_sequence+ "_RES");
    std::string config_file_path(argv[3]);
    std::string rf_file_path(argv[4]);
    std::string feature_file_path(argv[5]);

    fs::path tif_dir = fs::system_complete(tif_dir_str);
    fs::path res_dir = fs::system_complete(res_dir_str);

    // check validity of dataset variables
    if (!fs::exists(tif_dir)) {
      throw std::runtime_error(tif_dir_str + " does not exist");
    }
    if (!fs::is_directory(tif_dir)) {
      throw std::runtime_error(tif_dir_str + " is not a directory");
    }
    if (!fs::exists(res_dir)) {
      if (!fs::create_directory(res_dir)) {
        throw std::runtime_error(
          "Could not create directory: " + tif_dir_str + "_RES");
      }
    }

    // make unary for copying only filepaths that contain *.tif
    boost::function<bool (fs::directory_entry&)> tif_chooser = bind(
      isbi::contains_substring_boost_path, _1, ".tif");

    // load segmentation classifier
    std::vector<RandomForestType> rfs;
    if (!(isbi::get_rfs_from_file(rfs, rf_file_path))) {
      throw std::runtime_error("Could not load Random Forest classifier!");
    }

    // get features used in project
    std::string feature_list_path = dataset_folder + "/features.txt";
    StringDoublePairVectorType feature_list;
    int read_status = isbi::read_features_from_file(
      feature_file_path,
      feature_list);
    if (read_status == 1) {
      throw std::runtime_error("Could not open file " + feature_list_path);
    }

    // get config
    isbi::TrackingOptions options(config_file_path);
    if (!options.is_legal()) {
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
      fs::directory_iterator(tif_dir),
      fs::directory_iterator(),
      std::back_inserter(fn_vec),
      tif_chooser);
    std::sort(fn_vec.begin(), fn_vec.end());
    // initialize vector of label images
    std::vector<std::string> labelimage_fn_vec;
    // create the feature and segmentation calculator
    boost::shared_ptr<isbi::FeatureCalculator<2> > feature_calc_ptr(
      new isbi::FeatureCalculator<2>(feature_list));
    isbi::SegmentationCalculator<2> seg_calc(feature_calc_ptr, rfs);
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
      // TODO is the following assertion deprecated?
      if (dir_itr->extension().string().compare(".tif")) {
        continue;
      }
      if (!vigra::isImage(filename.c_str())) {
        continue;
      }
      // read the image
      vigra::ImageImportInfo info(filename.c_str());
      vigra::Shape2 shape(info.width(), info.height());
      // initialize some multi arrays
      DataMatrixType src_unsmoothed(shape);
      // read the image pixel data
      vigra::importImage(info, vigra::destImage(src_unsmoothed));
      std::cout << "Calculate features" << std::endl;
      isbi::Segmentation<2> segmentation;
      seg_calc.calculate(src_unsmoothed, segmentation);
      // save the segmentation results
      std::stringstream segmentation_result_path;
      segmentation_result_path <<  tif_dir_str << "_RES/"
        << "segmentation" << isbi::zero_padding(timestep, 3) << ".h5";
      std::cout << "Save results to " << segmentation_result_path.str()
        << std::endl;
      segmentation.export_hdf5(segmentation_result_path.str());
      std::stringstream labelimage_path;
      labelimage_path << tif_dir_str << "_RES/mask"
        << isbi::zero_padding(timestep, 3) << ".tif";
      std::cout << "Save results to " << labelimage_path.str() << std::endl;
      vigra::exportImage(
        vigra::srcImageRange(UShortMatrixType(segmentation.label_image_)),
        vigra::ImageExportInfo(labelimage_path.str().c_str()));
      labelimage_fn_vec.push_back(labelimage_path.str());
      // create traxels and add them to the traxelstore
      traxel_extractor.extract(segmentation, timestep, ts);
    }
    // end of iteration over all filenames/timesteps

    //=========================================================================
    // track!
    //=========================================================================
    isbi::EventVectorVectorType events = isbi::track(ts, options);

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
      // read the image
      vigra::ImageImportInfo import_info(fn_it->c_str());
      vigra::Shape2 shape(import_info.width(), import_info.height());
      // initialize the multi array
      UIntMatrixType labelimage(shape);
      // read the image pixel data
      vigra::importImage(import_info, vigra::destImage(labelimage));
      // relabel the label image
      lineage.relabel<2>(labelimage, timestep);
      // save results
      std::cout << "Save results to " << *fn_it << std::endl;
      vigra::exportImage(
        vigra::srcImageRange(UShortMatrixType(labelimage)),
        vigra::ImageExportInfo(fn_it->c_str()));
    }

    return 0;
  } catch (isbi::ArgumentError& e) {
    std::cout << e.what();
    std::cout << "Usage: " << argv[0]
      << " folder sequence config_path rf_path feature_path" << std::endl;
    return 0;
  } catch (std::runtime_error& e) {
    std::cout << "Program crashed:\n";
    std::cout << e.what();
    std::cout << std::endl;
    return 0;
  }
}
