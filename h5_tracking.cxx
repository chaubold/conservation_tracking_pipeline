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
    if (argc < 4) {
      throw ArgumentError();
    }
    // TODO what about presmoothing?
    // bool presmoothing = true;

    // dataset variables
    rstrip(argv[1], '/');
    rstrip(argv[2], '/');
    rstrip(argv[3], '/');
    std::string dataset_folder(argv[1]);
    std::string dataset_sequence(argv[2]);
    std::string h5_dir_str(dataset_folder + "/" + dataset_sequence + "_RES");
    std::string config_file_path(argv[3]);

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
      contains_substring_boost_path, _1, ".h5");

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
      fs::directory_iterator(h5_dir),
      fs::directory_iterator(),
      back_inserter(fn_vec),
      h5_chooser);
    std::sort(fn_vec.begin(), fn_vec.end());
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
      // load the segmentation
      isbi_pipeline::Segmentation<2> segmentation;
      segmentation.read_hdf5(filename);
      // create traxels and add them to the traxelstore
      traxel_extractor.extract(segmentation, timestep, ts);
    }

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
    boost::function<bool (fs::directory_entry&)> tif_chooser = bind(
      contains_substring_boost_path, _1, ".tif");
    copy_if_own(
      fs::directory_iterator(h5_dir),
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
    write_lineages(lineage_vec, h5_dir_str + "/res_track.txt");

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
      << " folder sequence config_path" << std::endl;
    return 0;
  } catch (std::runtime_error& e) {
    std::cout << "Program crashed:\n";
    std::cout << e.what();
    return 0;
  }
}
