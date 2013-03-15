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
#include <vigra/accumulator.hxx>
#include <vigra/accessor.hxx>

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

typedef FEATURETYPE DATATYPE;
typedef vigra::CoupledIteratorType<2, unsigned, unsigned>::type Iterator;
typedef Iterator::value_type Handle;
typedef AccumulatorChainArray<Handle,
				   Select<DataArg<1>, LabelArg<2>,
					  Count, Coord<Mean> >
			      > chain;


int main(int argc, char** argv) {
  // arg 1: dataset folder
  try {
    // check for correct number of arguments
    if (argc < 6) {
      throw ArgumentError();
    }
    bool presmoothing = true;
    

    // dataset variables
    rstrip(argv[1], '/');
    rstrip(argv[2], '/');
    rstrip(argv[3], '/');
    rstrip(argv[4], '/');
    rstrip(argv[5], '/');
    string dataset_folder(argv[1]);
    string dataset_sequence(argv[2]);
    string tif_dir_str(dataset_folder + "/" + dataset_sequence);
    string res_dir_str(dataset_folder + "/" + dataset_sequence+ "_RES");
    string config_file_path(argv[3]);
    string rf_file_path(argv[4]);
    string feature_file_path(argv[5]);
    
    
    fs::path tif_dir = fs::system_complete(tif_dir_str);
    fs::path res_dir = fs::system_complete(res_dir_str);

    // check validity of dataset variables
    if (!fs::exists(tif_dir)) {
      throw runtime_error(tif_dir_str + " does not exist");
    }
    
    if (!fs::is_directory(tif_dir)) {
      throw runtime_error(tif_dir_str + " is not a directory");
    }

    if (!fs::exists(res_dir)) {
      if (!fs::create_directory(res_dir)) {
	throw runtime_error("Could not create directory: " + tif_dir_str + "_RES");
      }
    }


    // make unary for copying only filepaths that contain *.tif
    // std::unary_function<fs::directory_entry&, bool> tif_chooser = bind2nd(ptr_fun(contains_substring_boost_path), ".tif");
    boost::function<bool (fs::directory_entry&)> tif_chooser = bind(contains_substring_boost_path, _1, ".tif");
    // int dim = 2;


    // load segmentation classifier
    string segmentation_ilp_path(dataset_folder + "/segment.ilp");
    vector<RandomForest<unsigned> > rfs;
    if (!get_rfs_from_file(rfs, rf_file_path)) { // segmentation_ilp_path)) {
      throw runtime_error("Could not load Random Forest classifier!");
    }


    // get features used in project
    string feature_list_path = dataset_folder + "/features.txt";
    vector<pair<string, string> > feature_list;
    int read_status = read_features_from_file(feature_file_path, feature_list);
    if (read_status == 1) {
      throw runtime_error("Could not open file " + feature_list_path);
    }


    // set config
    map<string, double> options;
    // string config_file_path = dataset_folder + "/config.txt";
    int config_status = read_config_from_file(config_file_path, options);
    if (config_status == 1) {
      throw runtime_error("Could not open file " + config_file_path);
    }

    

    // run segmentation for all tif files in the given folder
    // write results to <dataset_folder>/<dataset_sequence_segmentation>
    fs::directory_iterator end_itr;
    vector<MultiArray<2, unsigned> > label_images;
    int timestep = 0;
    pgmlink::TraxelStore ts;
    // sort filenames
    vector<fs::path> fn_vec;
    copy_if_own(fs::directory_iterator(tif_dir), fs::directory_iterator(), back_inserter(fn_vec), tif_chooser);
    sort(fn_vec.begin(), fn_vec.end());
    for (vector<fs::path>::iterator dir_itr = fn_vec.begin(); dir_itr != fn_vec.end(); ++dir_itr, ++timestep) {
      string filename(dir_itr->string());
      cout << "processing " + filename + " ...\n";
      if (dir_itr->extension().string().compare(".tif")) {
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


      // calculate features
      vector<pair<string, string> >::iterator it = feature_list.begin();
      vector<vector<MultiArray<2, FEATURETYPE> > > features;
      int feature_dim = 0;
      for (; it != feature_list.end(); ++it) {
	feature_dim += feature_dim_lookup_size(it->first);
	features.push_back(vector<MultiArray<2, FEATURETYPE> >());
	double scale = string_to_double(it->second);
	double presmooth_sigma = 1.0;
	if (scale <= 1.0 || !presmoothing) {
          presmooth_sigma = scale;
        } else {          
          presmooth_sigma = sqrt(scale*scale - presmooth_sigma*presmooth_sigma);
        }
	
	vigra::ConvolutionOptions<2> opt;
	opt.filterWindowSize(3.5);
	if (presmoothing) {
	  gaussianSmoothMultiArray(srcMultiArrayRange(src_unsmoothed), destMultiArray(src), presmooth_sigma, opt);
          // cout << "  Presmoothing image (s=" << it->second << "," << presmooth_sigma << ")\n";
	} else {
	  src = src_unsmoothed;
	}
	// gaussianSmoothing(srcImageRange(src_unsmoothed), destImage(src), presmooth_sigma);
	double feature_sigma = 1.0;
        double window_size = 2.0;
	if (scale <= 1.0 || !presmoothing)
          feature_sigma = scale;
        if (!presmoothing) {
          window_size = 3.5;
        }
        
	// cout << "  Calculating feature " + it->first + "(s=" + it->second + "," << feature_sigma << ")\n";
	int feature_status = get_features<2>(src, features[features.size()-1], it->first, feature_sigma, window_size);
		      
	if (feature_status == 1)
	  throw runtime_error("get_features not implemented for feature " + it->first);
	else if (feature_status == 2)
	  throw runtime_error("vector passed to get_features is not a zero-length vector");
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
	if (label_pred_1 > label_pred_0) {
	  *label_it = 1;
        } else {
	  *label_it = 0;
        }
      }

      // extract objects
      MultiArray<2, unsigned> label_image(shape);
      int n_regions = labelImageWithBackground(srcImageRange(labels), destImage(label_image), 1, 0);
      if (options.count("border") > 0) {
        ignore_border_cc<2>(label_image, options["border"]);
      }
      // cout << "Detected " << n_regions << " connected components\n";
      stringstream segmentation_result_path;
      segmentation_result_path <<  tif_dir_str + "_RES/" + "mask" + zero_padding(timestep, 2) + ".tif";
      // exportImage(srcImageRange<unsigned, StandardConstAccessor<short> >(label_image), ImageExportInfo(segmentation_result_path.str().c_str())); // .setPixelType("INT16"));
      exportImage(srcImageRange(vigra::MultiArray<2, short>(label_image)), ImageExportInfo(segmentation_result_path.str().c_str()));


      // calculate features and build TraxelStore
      chain accu_chain;
      std::vector<unsigned> filtered_labels;
      Iterator start = createCoupledIterator(label_image, label_image);
      Iterator end = start.getEndIterator();
      extractFeatures(start, end, accu_chain);
      for (int i = 1; i <= n_regions; ++i) {
         float size = get<Count>(accu_chain, i);
         if ((options.count("size_from") > 0 && size < options["size_from"]) || options.count("size_to") > 0 && size > options["size_to"]) {
            filtered_labels.push_back(i);
            continue;
         }
 
         filtered_labels.push_back(size);
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


    vector<vector<pgmlink::Event> > events = track(ts, options);
    vector<vector<pgmlink::Event> >::const_iterator EV_IT = events.begin();
    for(; EV_IT != events.end(); ++EV_IT) {
      for (vector<pgmlink::Event>::const_iterator ev = EV_IT->begin(); ev != EV_IT->end(); ++ev) {
        cout << *ev << "\n";
      }
      cout << "\n";
    }
    vector<Lineage> lineage_vec;


    vector<fs::path> res_fn_vec;
    copy_if_own(fs::directory_iterator(res_dir), fs::directory_iterator(), back_inserter(res_fn_vec), tif_chooser);
    sort(res_fn_vec.begin(), res_fn_vec.end());
    cout << res_fn_vec[0].string() << "\n";


    ImageImportInfo info_general(res_fn_vec[0].string().c_str());
    MultiArrayShape<2>::type shape_general(info_general.shape());
    MultiArray<2, unsigned> base_img(shape_general);
    importImage(info_general, destImage(base_img));
    int max_l_id = initialize_lineages<2>(lineage_vec, base_img);

    transform_events<2>(events,
                        ++res_fn_vec.begin(),
                        res_fn_vec.end(),
                        lineage_vec,
                        shape_general,
                        max_l_id,
                        1);

    write_lineages(lineage_vec, res_dir_str + "/res_track.txt");

    


    // base_img = MultiArray<2, unsigned>();

    /* close_open_lineages(lineage_vec, 500);
    for (vector<Lineage>::iterator lm_it = lineage_vec.begin(); lm_it != lineage_vec.end(); ++lm_it) {
      cout << *lm_it << "\n";

      }*/ 
    
    

    /* vigra::MultiArray<2, unsigned> relabelll(shape_general);
    relabelll *= 0;
    cout << find_lineage_by_o_id(lineage_vec, 2) << "\n";
    relabel_image<2>(base_img, relabelll, 2, 50);
    exportImage(srcImageRange(relabelll), ImageExportInfo("relabel.tif")); */
    
    					

    return 0;
  }
  
  catch (ArgumentError& e) {
    cout << e.what();
    cout << "Usage: " << argv[0] << " folder sequence config_path rf_path feature_path" << endl;
    return 0;
  }
  
  catch (runtime_error& e) {
    cout << "Program crashed:\n";
    cout << e.what();
    return 0;
  }
}
