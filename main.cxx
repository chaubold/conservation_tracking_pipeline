// stl
#include <exception>
#include <stdexcept>
#include <iostream>
#include <string>

// boost
#include <boost/filesystem.hpp>

// vigra
#include <vigra/impex.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/random_forest.hxx>

// own
#include "pipeline_helpers.hxx"


// namespaces to be used;
using namespace std;
using namespace vigra;
namespace fs = boost::filesystem;

typedef unsigned DATATYPE;

int main(int argc, char** argv) {
  // arg 1: dataset folder
  try {
    // check for correct number of arguments
    if (argc < 3) {
      throw ArgumentError();
    }

    // dataset variables
    rstrip(argv[1], '/');
    rstrip(argv[2], '/');
    string dataset_folder(argv[1]);
    string dataset_sequence(argv[2]);
    string tif_dir_str(dataset_folder + "/" + dataset_sequence);
    
    fs::path tif_dir = fs::system_complete(tif_dir_str);

    // check validity of dataset variables
    if (!fs::exists(tif_dir)) {
      throw runtime_error(tif_dir_str + " does not exist");
    }
    if (!fs::is_directory(tif_dir)) {
      throw runtime_error(tif_dir_str + " is not a directory");
    }

    // load segmentation classifier
    string segmentation_ilp_path(dataset_folder + "/segment.ilp");
    string rf_path("PixelClassification/ClassifierForests/Forest0000");
    RandomForest<unsigned> rf;
    if (!rf_import_HDF5(rf, segmentation_ilp_path, rf_path)) {
      throw runtime_error("Could not load Random Forest classifier!");
    }

    // run segmentation for all tif files in the given folder
    // write results to <dataset_folder>/<dataset_sequence_segmentation>
    fs::directory_iterator end_itr;
    for (fs::directory_iterator dir_itr(tif_dir); dir_itr != end_itr; ++dir_itr) {
      string filename(dir_itr->path().string());
      if (dir_itr->path().extension().string().compare(".tif")) {
	continue;
      }
      if (!isImage(filename.c_str())) {
	continue;
      }
      // read image
      ImageImportInfo info(filename.c_str());
      Shape2 shape(info.width(), info.height());
      MultiArray<2, DATATYPE> src(shape);
      MultiArray<2, unsigned> labels(shape);
      
      importImage(info, destImage(src));
      // map tp [0,255] if neccessary
      remap<DATATYPE, 2>(src);
      
    }
    
    return 0;
  }
  
  catch (ArgumentError& e) {
    cout << e.what();
    cout << "Usage: " << argv[0] << " folder sequence" << endl;
    return 0;
  }
  
  catch (runtime_error& e) {
    cout << e.what();
    return 0;
  }
}
