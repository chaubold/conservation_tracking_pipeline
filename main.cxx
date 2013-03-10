// stl
#include <exception>
#include <stdexcept>
#include <iostream>
#include <string>
#include <vector>
#include <utility>
#include <fstream>

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
    vector<RandomForest<unsigned> > rfs;
    for (int n_rf = 0; n_rf < 10; ++n_rf) {
      string rf_path = "PixelClassification/ClassifierForests/Forest" + zero_padding(n_rf, 4);
      cout << rf_path << "\n";
      rfs.push_back(RandomForest<unsigned>());
      if (!rf_import_HDF5(rfs[n_rf], segmentation_ilp_path, rf_path)) {
	throw runtime_error("Could not load Random Forest classifier!");
      }
    }

    // get features used in project
    string feature_list_path = dataset_folder + "/features.txt";
    vector<pair<string, string> > feature_list;
    int read_status = read_features_from_file(feature_list_path, feature_list);
    if (read_status == 1) throw runtime_error("Could not open file " + feature_list_path);
    
    

    // run segmentation for all tif files in the given folder
    // write results to <dataset_folder>/<dataset_sequence_segmentation>
    fs::directory_iterator end_itr;
    vector<MultiArray<2, unsigned> > label_images;
    for (fs::directory_iterator dir_itr(tif_dir); dir_itr != end_itr; ++dir_itr) {
      string filename(dir_itr->path().string());
      cout << "processing " + filename + " ...\n";
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

      // calculate features
      vector<pair<string, string> >::iterator it = feature_list.begin();
      vector<vector<MultiArray<2, double> > > features;
      int feature_dim = 0;
      for (; it != feature_list.end(); ++it) {
	cout << " Calculating feature " + it->first + "(s=" + it->second + ")\n";
	feature_dim += feature_dim_lookup_size(it->first);
	features.push_back(vector<MultiArray<2, double> >());
	double scale = string_to_double(it->second);
	int feature_status = get_features(src, features[features.size()-1], it->first, scale);
	if (feature_status == 1)
	  throw runtime_error("get_features not implemented for feature " + it->first);
	else if (feature_status == 2)
	  throw runtime_error("vector passed to get_features is not a zero-length vector");
      }
    }

    // extract objects

    // calculate features

    // chaingrpah tracking

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
