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

// boost
#include <boost/filesystem.hpp>

// vigra
#include <vigra/impex.hxx>
#include <vigra/hdf5impex.hxx>
#include <vigra/random_forest_hdf5_impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/random_forest.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/labelimage.hxx>
#include <vigra/accumulator.hxx>

// pgmlink
#include <pgmlink/traxels.h>
#include <pgmlink/tracking.h>

// own
#include "pipeline_helpers.hxx"


// namespaces to be used;
using namespace std;
using namespace vigra;
using namespace vigra::acc;
namespace fs = boost::filesystem;

typedef unsigned DATATYPE;
typedef vigra::CoupledIteratorType<2, DATATYPE, DATATYPE>::type Iterator;
typedef Iterator::value_type Handle;
typedef AccumulatorChainArray<Handle,
				   Select<DataArg<1>, LabelArg<2>,
					  Count, Coord<Mean> >
			      > chain;


int main(int argc, char** argv) {
  // arg 1: dataset folder
  try {
    // check for correct number of arguments
    if (argc < 3) {
      throw ArgumentError();
    }
    bool presmoothing;
    if (argc == 4) {
      presmoothing = false;
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

    if (!fs::exists(fs::path(tif_dir_str + "_RES"))) {
      if (!fs::create_directory(fs::path(tif_dir_str + "_RES"))) {
	throw runtime_error("Could not create directory: " + tif_dir_str + "_RES");
      }
    }
      

    // int dim = 2;


    // load segmentation classifier
    string segmentation_ilp_path(dataset_folder + "/segment.ilp");
    vector<RandomForest<unsigned> > rfs;
    for (int n_rf = 0; n_rf < 10; ++n_rf) {
      string rf_path = "PixelClassification/ClassifierForests/Forest" + zero_padding(n_rf, 4);
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
    
    //for (vector<pair<string, string> >::iterator fl_it = feature_list.begin(); fl_it != feature_list.end(); ++fl_it) {
    //      maxSigma = max(string_to_double(fl_it->second), maxSigma);
    //}


    // set config
    map<string, double> options;
    string config_file_path = dataset_folder + "/config.txt";
    int config_status = read_config_from_file(config_file_path, options);
    if (config_status == 1) throw runtime_error("Could not open file " + config_file_path);

    

    // run segmentation for all tif files in the given folder
    // write results to <dataset_folder>/<dataset_sequence_segmentation>
    fs::directory_iterator end_itr;
    vector<MultiArray<2, unsigned> > label_images;
    int timestep = 1;
    pgmlink::TraxelStore ts;
    for (fs::directory_iterator dir_itr(tif_dir); dir_itr != end_itr; ++dir_itr, ++timestep) {
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
      MultiArray<2, DATATYPE> src_unsmoothed(shape);
      MultiArray<2, DATATYPE> src(shape);
      MultiArray<2, unsigned> labels(shape);
      
      importImage(info, destImage(src_unsmoothed));
      
      // map tp [0,255] if neccessary
      // remap<DATATYPE, 2>(src);
      if (options.count("min") && options.count("max"))
	cout << "normalizing image to [0, 255]\n";
	renormalize_to_8bit<2, DATATYPE>(src_unsmoothed, options["min"], options["max"]);

      // calculate features
      vector<pair<string, string> >::iterator it = feature_list.begin();
      vector<vector<MultiArray<2, FEATURETYPE> > > features;
      int feature_dim = 0;
      for (; it != feature_list.end(); ++it) {
	feature_dim += feature_dim_lookup_size(it->first);
	features.push_back(vector<MultiArray<2, FEATURETYPE> >());
	double scale = string_to_double(it->second);
	double presmooth_sigma = 1.0;
	if (scale <= 1.0 || !presmoothing)
          presmooth_sigma = scale;
        else
          presmooth_sigma = sqrt(scale*scale - presmooth_sigma*presmooth_sigma);
	cout << "  Presmoothing image (s=" << it->second << "," << presmooth_sigma << ")\n";
	vigra::ConvolutionOptions<2> opt;
	opt.filterWindowSize(3.5);
	if (presmoothing) {
	  gaussianSmoothMultiArray(srcMultiArrayRange(src_unsmoothed), destMultiArray(src), presmooth_sigma, opt);
	} else {
	  src = src_unsmoothed;
	}
	// gaussianSmoothing(srcImageRange(src_unsmoothed), destImage(src), presmooth_sigma);
	double feature_sigma = 1.0;
	if (scale <= 1.0 || !presmoothing)
          feature_sigma = scale;
	cout << "  Calculating feature " + it->first + "(s=" + it->second + "," << feature_sigma << ")\n";
	int feature_status = get_features<2>(src, features[features.size()-1], it->first, feature_sigma);
	for (unsigned index = 0; index < features[features.size()-1].size(); ++index) {
	  stringstream ss;
	  ss << features.size() << "_" << index;
	  string number = ss.str();
	  cout << "Maximum in Channel " + number + ": " << *argMax(features[features.size()-1][index].begin(), features[features.size()-1][index].end()) << "\n";
	  exportImage(srcImageRange(features[features.size()-1][index]), ImageExportInfo(("feature" + number + ".tif").c_str()));
	}
		      
	if (feature_status == 1)
	  throw runtime_error("get_features not implemented for feature " + it->first);
	else if (feature_status == 2)
	  throw runtime_error("vector passed to get_features is not a zero-length vector");
	// features[features.size()-1][0] *= 255.0/ *argMax(features[features.size()-1][0].begin(), features[features.size()-1][0].end());
      }


      
      MultiArray<2, FEATURETYPE> feat(Shape2(1, feature_dim));
      MultiArray<2, FEATURETYPE> res_ar(Shape2(1, 2));
      MultiArray<2, unsigned>::iterator label_it = labels.begin();
      unsigned step_count(0);
      for (; label_it != labels.end(); ++label_it, ++step_count) {
	int feat_index = 0;
	for (vector<vector<MultiArray<2, FEATURETYPE> > >::iterator ft_vec_it = features.begin(); ft_vec_it != features.end(); ++ft_vec_it) {
	  for (vector<MultiArray<2, FEATURETYPE> >::iterator ft_it = ft_vec_it->begin(); ft_it != ft_vec_it->end(); ++ft_it, ++feat_index) {
	    feat(0, feat_index) = *(ft_it->begin()+step_count);
	  }
	}
	double label_pred_0 = 0.0;
	double label_pred_1 = 0.0;
	for (vector<RandomForest<unsigned> >::iterator rf_it = rfs.begin(); rf_it != rfs.end(); ++rf_it) {
	  rf_it->predictProbabilities(feat, res_ar);
	  label_pred_0 += res_ar(0,0);
	  label_pred_1 += res_ar(0,1);
	}
	if (label_pred_1 > label_pred_0)
	  *label_it = 1;
	else
	  *label_it = 0;
      }
      string segmentation_result_path = tif_dir_str + "_RES/" + dir_itr->path().filename().string() + ".png";
      exportImage(srcImageRange(labels), ImageExportInfo(segmentation_result_path.c_str()));

      // extract objects
      MultiArray<2, unsigned> label_image(src.shape());
      int n_regions = labelImageWithBackground(srcImageRange(labels), destImage(label_image), 1, 0);
      exportImage(srcImageRange(label_image), ImageExportInfo("labels.png"));


      // calculate features and build TraxelStore
      chain accu_chain;
      Iterator start = createCoupledIterator(label_image, label_image);
      Iterator end = start.getEndIterator();
      extractFeatures(start, end, accu_chain);
      for (int i = 1; i <= n_regions; ++i) {
	float size = get<Count>(accu_chain, i);
	vector<float> com(get<Coord<Mean> >(accu_chain, i).begin(), get<Coord<Mean> >(accu_chain, i).end());
	if (com.size() == 2)
	  com.push_back(0);
	pgmlink::FeatureMap f_map;
	f_map["com"] = com;
	f_map["count"].push_back(size);
	pgmlink::Traxel trax(i, timestep, f_map);
	pgmlink::add(ts, trax);
      }
    }

    // chaingrpah tracking
    pgmlink::ChaingraphTracking tracker("none",
					10000., // appearance
					10000., // disappearance
					10., // detection
					500., // misdetection
					false, // cellness by rf
					10000., // opportunity cost
					0., // forbidden cost
					true, // with constraints
					false, // fixed detections
					20., // mean div dist
					0., // min angle
					0.05, // ep_gap
					2., // n neighbors
					false // alternative builder
					);
    
    vector<vector<pgmlink::Event> > events = tracker(ts);
    for (vector<vector<pgmlink::Event> >::iterator ts_it = events.begin(); ts_it != events.end(); ++ts_it) {
      for (vector<pgmlink::Event>::iterator ev_it = ts_it->begin(); ev_it != ts_it->end(); ++ev_it) {
	cout << *ev_it << "\n";
      }
    }					

    return 0;
  }
  
  catch (ArgumentError& e) {
    cout << e.what();
    cout << "Usage: " << argv[0] << " folder sequence" << endl;
    return 0;
  }
  
  catch (runtime_error& e) {
    cout << "Program crashed:\n";
    cout << e.what();
    return 0;
  }
}
