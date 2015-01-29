// stl
#include <exception>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <fstream>
#include <map>
#include <algorithm>
#include <cmath>
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
#include <vigra/multi_convolution.hxx>
#include <vigra/labelimage.hxx>
#include <vigra/accessor.hxx>

// pgmlink
#include <pgmlink/traxels.h>
#include <pgmlink/tracking.h>

// own
#include "pipeline_helpers.hxx"
#include "segmentation.hxx"


// Aliases for convenience
namespace fs = boost::filesystem;

// Typedefs for convenience
// TODO Move some to pipeline_helpers eg. EventVector
//
// Scalar types
typedef FeatureType DataType;
// Container types of the standard library
typedef std::vector<std::pair<std::string, double> > StringDoublePairVectorType;
typedef std::map<std::string, double> StringDoubleMapType;
// Vigra MultiArray types
typedef vigra::MultiArray<2, DataType> DataMatrixType;
typedef vigra::MultiArray<2, FeatureType> FeatureMatrixType;
typedef vigra::MultiArray<2, unsigned> UIntMatrixType;
typedef vigra::MultiArray<2, unsigned short> UShortMatrixType;
// Further vigra types
typedef vigra::RandomForest<unsigned> RandomForestType;
// Container types of the standard library with vigra types
// TODO Refactor feature calculation without awkward FeatureStorageType
typedef std::vector<std::vector<FeatureMatrixType> > FeatureStorageType;
// Container types of the standard library with pgmlink types
typedef std::vector<pgmlink::Event> EventVectorType;
typedef std::vector<EventVectorType> EventVectorVectorType;

int main(int argc, char** argv) {
  // arg 1: dataset folder
  try {
    // check for correct number of arguments
    if (argc < 6) {
      throw ArgumentError();
    }
    // TODO what about presmoothing?
    // bool presmoothing = true;

    // dataset variables
    rstrip(argv[1], '/');
    rstrip(argv[2], '/');
    rstrip(argv[3], '/');
    rstrip(argv[4], '/');
    rstrip(argv[5], '/');
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
      contains_substring_boost_path, _1, ".tif");

    // load segmentation classifier
    std::vector<RandomForestType> rfs;
    if (!get_rfs_from_file(rfs, rf_file_path)) {
      throw std::runtime_error("Could not load Random Forest classifier!");
    }

    // get features used in project
    std::string feature_list_path = dataset_folder + "/features.txt";
    StringDoublePairVectorType feature_list;
    int read_status = read_features_from_file(feature_file_path, feature_list);
    if (read_status == 1) {
      throw std::runtime_error("Could not open file " + feature_list_path);
    }

    // get config
    StringDoubleMapType options;
    int config_status = read_config_from_file(config_file_path, options);
    if (config_status == 1) {
      throw std::runtime_error("Could not open file " + config_file_path);
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
    copy_if_own(
      fs::directory_iterator(tif_dir),
      fs::directory_iterator(),
      back_inserter(fn_vec),
      tif_chooser);
    std::sort(fn_vec.begin(), fn_vec.end());
    // create the feature and segmentation calculator
    boost::shared_ptr<isbi_pipeline::FeatureCalculator<2> > feature_calc_ptr(
      new isbi_pipeline::FeatureCalculator<2>(feature_list));
    isbi_pipeline::SegmentationCalculator<2> seg_calc(feature_calc_ptr, rfs);
    // create the traxel extractor
    isbi_pipeline::TraxelExtractor<2> traxel_extractor(0, 100, 0);

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
      isbi_pipeline::Segmentation<2> segmentation;
      seg_calc.calculate(src_unsmoothed, segmentation);
      // save the segmentation results
      std::stringstream segmentation_result_path;
      segmentation_result_path <<  tif_dir_str << "_RES/"
        << "segmentation" << zero_padding(timestep, 2);
      std::cout << "Save results to " << segmentation_result_path.str() + ".h5"
        << std::endl;
      segmentation.export_hdf5(segmentation_result_path.str() + ".h5");
      std::cout << "Save results to " << segmentation_result_path.str() + ".tif"
        << std::endl;
      vigra::exportImage(
        vigra::srcImageRange(UShortMatrixType(segmentation.label_image_)),
        vigra::ImageExportInfo((segmentation_result_path.str()+".tif").c_str()));
      // create traxels and add them to the traxelstore
      traxel_extractor.extract(segmentation, timestep, ts);
    }
    // end of iteration over all filenames/timesteps

    //=========================================================================
    // track!
    //=========================================================================
    EventVectorVectorType events = track(ts, options);

    //=========================================================================
    // handle results
    //=========================================================================
    // Print the results to std::cout
    for(
      EventVectorVectorType::const_iterator ev_it = events.begin();
      ev_it != events.end();
      ++ev_it)
    {
      for (
        EventVectorType::const_iterator ev = ev_it->begin();
        ev != ev_it->end();
        ++ev)
      {
        std::cout << *ev << "\n";
      }
      std::cout << "\n";
    }

    // create the lineage
    std::vector<Lineage> lineage_vec;

    std::vector<fs::path> res_fn_vec;
    // write all filenames that end with *.tif into the res_fn_vec
    copy_if_own(
      fs::directory_iterator(res_dir),
      fs::directory_iterator(),
      back_inserter(res_fn_vec),
      tif_chooser);
    std::sort(res_fn_vec.begin(), res_fn_vec.end());
    std::cout << res_fn_vec[0].string() << "\n";

    // write the lineages to the lineage file
    vigra::ImageImportInfo info_general(res_fn_vec[0].string().c_str());
    vigra::MultiArrayShape<2>::type shape_general(info_general.shape());
    UIntMatrixType base_img(shape_general);
    vigra::importImage(info_general, vigra::destImage(base_img));
    int max_l_id = initialize_lineages<2>(lineage_vec, base_img);

    // create lineages from the events
    transform_events<2>(
      events,
      ++res_fn_vec.begin(),
      res_fn_vec.end(),
      lineage_vec,
      shape_general,
      max_l_id,
      1);
    // save the lineages
    write_lineages(lineage_vec, res_dir_str + "/res_track.txt");

    // base_img = MultiArray<2, unsigned>();

    /* close_open_lineages(lineage_vec, 500);
    for (std::vector<Lineage>::iterator lm_it = lineage_vec.begin(); lm_it != lineage_vec.end(); ++lm_it) {
      cout << *lm_it << "\n";

      }*/

    /* vigra::MultiArray<2, unsigned> relabelll(shape_general);
    relabelll *= 0;
    cout << find_lineage_by_o_id(lineage_vec, 2) << "\n";
    relabel_image<2>(base_img, relabelll, 2, 50);
    exportImage(srcImageRange(relabelll), ImageExportInfo("relabel.tif")); */

    return 0;
  } catch (ArgumentError& e) {
    std::cout << e.what();
    std::cout << "Usage: " << argv[0]
      << " folder sequence config_path rf_path feature_path" << std::endl;
    return 0;
  } catch (std::runtime_error& e) {
    std::cout << "Program crashed:\n";
    std::cout << e.what();
    return 0;
  }
}
