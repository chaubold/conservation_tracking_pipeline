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


// Aliases for convenience
namespace acc = vigra::acc;
namespace fs = boost::filesystem;

// Typedefs for convenience
// TODO Move some to pipeline_helpers eg. EventVector
//
// Scalar types
typedef FeatureType DataType;
// Container types of the standard library
typedef std::vector<std::pair<std::string, std::string> > StringPairVectorType;
typedef std::map<std::string, double> StringDoubleMapType;
// Vigra MultiArray types
typedef vigra::MultiArray<2, DataType> DataMatrixType;
typedef vigra::MultiArray<2, FeatureType> FeatureMatrixType;
typedef vigra::MultiArray<2, unsigned> UIntMatrixType;
// Further vigra types
typedef vigra::RandomForest<unsigned> RandomForestType;
typedef vigra::CoupledIteratorType<2, unsigned, unsigned>::type
  CoupledIteratorType;
typedef acc::AccumulatorChainArray<
  CoupledIteratorType::value_type,
  acc::Select<
    acc::DataArg<1>,
    acc::LabelArg<2>,
    acc::Count,
    acc::Coord<acc::Mean> >
> AccChainType;
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
    bool presmoothing = true;

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
    StringPairVectorType feature_list;
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

    std::vector<UIntMatrixType> label_images;
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

    // iterate over the filenames
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
      DataMatrixType src(shape);
      UIntMatrixType labels(shape);
      // read the image pixel data
      vigra::importImage(info, vigra::destImage(src_unsmoothed));

      // calculate features
      int feature_dim = 0;
      FeatureStorageType features; // type is vec<vec<Matrix> >
      for (
        StringPairVectorType::iterator it = feature_list.begin();
        it != feature_list.end();
        ++it)
      {
        feature_dim += feature_dim_lookup_size(it->first);
        features.push_back(std::vector<FeatureMatrixType>());
        double scale = string_to_double(it->second);
        double presmooth_sigma = 1.0;
        if (scale <= 1.0 || !presmoothing) {
          presmooth_sigma = scale;
        } else {
          presmooth_sigma = std::sqrt(
            scale*scale - presmooth_sigma*presmooth_sigma);
        }

        // gaussian presmoothing of the image
        vigra::ConvolutionOptions<2> opt;
        opt.filterWindowSize(3.5); // TODO hard coded value
        if (presmoothing) {
          vigra::gaussianSmoothMultiArray(
            vigra::srcMultiArrayRange(src_unsmoothed),
            vigra::destMultiArray(src),
            presmooth_sigma,
            opt);
        } else {
          src = src_unsmoothed;
        }

        // TODO hard coded values in the following
        double feature_sigma = 1.0;
        double window_size = 2.0;
        if (scale <= 1.0 || !presmoothing) {
          feature_sigma = scale;
        }
        if (!presmoothing) {
          window_size = 3.5;
        }

        // Calculate the features
        std::cout << "  Calculating feature " << it->first
          << " (s=" << it->second << "," << feature_sigma << ")\n";
        int feature_status = get_features<2>(
          src,
          features[features.size()-1],
          it->first,
          feature_sigma,
          window_size);
        // raise an error if the feature status is 1 or 2
        if (feature_status == 1) {
          throw std::runtime_error(
            "get_features not implemented for feature " + it->first);
        } else if (feature_status == 2) {
          throw std::runtime_error(
            "vector passed to get_features is not a zero-length vector");
        }
      }

      // initialize some variables outside the loop over all pixels
      // feat(1, feature_dim): feature vector for one pixel
      // res_ar(1, 2): classification results of random forests
      FeatureMatrixType feat(vigra::Shape2(1, feature_dim));
      FeatureMatrixType res_ar(vigra::Shape2(1, 2));
      unsigned step_count = 0;
      UIntMatrixType::iterator label_it = labels.begin();
      // TODO Have a closer look at the following code
      // iterate over all labels and get the segmentation classification
      for (; label_it != labels.end(); ++label_it, ++step_count) {
        // write all features into a feature vector
        int feat_index = 0;
        for (
          FeatureStorageType::iterator ft_vec_it = features.begin();
          ft_vec_it != features.end();
          ++ft_vec_it)
        {
          for (
            std::vector<FeatureMatrixType>::iterator ft_it = ft_vec_it->begin();
            ft_it != ft_vec_it->end();
            ++ft_it, ++feat_index
          ) {
            feat(0, feat_index) = *(ft_it->begin()+step_count);
          }
        }
        // get the rf probability prediction values
        double label_pred_0 = 0.0;
        double label_pred_1 = 0.0;
        for (
          std::vector<RandomForestType>::iterator rf_it = rfs.begin();
          rf_it != rfs.end();
          ++rf_it)
        {
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
      UIntMatrixType label_image(shape);
      int n_regions = vigra::labelImageWithBackground(
        vigra::srcImageRange(labels),
        vigra::destImage(label_image),
        1,
        0);
      if (options.count("border") > 0) {
        ignore_border_cc<2>(label_image, options["border"]);
      }
      // save the segmentation results
      std::stringstream segmentation_result_path;
      segmentation_result_path <<  tif_dir_str << "_RES/"
        << "mask" << zero_padding(timestep, 2) + ".tif";
      // TODO why are the following lines commented?
      // vigra::exportImage(
      //  vigra::srcImageRange<unsigned, vigra::StandardConstAccessor<short> >(label_image),
      //  vigra::ImageExportInfo(segmentation_result_path.str().c_str()));
      //  // .setPixelType("INT16"));

      // calculate size and com with an accumulator chain to build the
      // TraxelStore
      AccChainType accu_chain;
      // TODO why is the label image of the filtered labels updated for t == 0?
      // Remove?
      std::vector<unsigned> filtered_labels_at_0;
      CoupledIteratorType start = vigra::createCoupledIterator(
        label_image,
        label_image);
      CoupledIteratorType end = start.getEndIterator();
      // calculate the size and coordinate center of mass
      acc::extractFeatures(start, end, accu_chain);
      // variable for relabeling the filtered objects for t == 0
      int count_true_object = 1;
      // loop over all objects
      for (int i = 1; i <= n_regions; ++i) {
        // get the object size
        float size = acc::get<acc::Count>(accu_chain, i);
        // check if the object is within the desired size range
        bool fits_lower_limit = true;
        if (options.count("size_from") > 0) {
          fits_lower_limit = (size >= options["size_from"]);
        }
        bool fits_upper_limit = true;
        if (options.count("size_to") > 0) {
          fits_upper_limit = (size <= options["size_to"]);
        }
        if (!(fits_lower_limit && fits_upper_limit)) {
          if (timestep == 0) {
            filtered_labels_at_0.push_back(i);
            set_pixels_of_cc_to_value<2>(label_image, i, 0);
          }
          continue;
        }
        // create a traxel for each object that passed the size filter
        // TODO Remove the following if-else statement as well if the filtered
        // label image is removed.
        if (timestep == 0) {
          set_pixels_of_cc_to_value<2>(label_image, i, count_true_object);
          typedef acc::Coord<acc::Mean> CoordMeanType;
          std::vector<float> com(
            acc::get<CoordMeanType>(accu_chain, count_true_object).begin(),
            acc::get<CoordMeanType>(accu_chain, count_true_object).end());
          if (com.size() == 2) {
            com.push_back(0);
          }
          pgmlink::FeatureMap f_map;
          f_map["com"] = com;
          f_map["count"].push_back(size);
          pgmlink::Traxel trax(count_true_object, timestep, f_map);
          pgmlink::add(ts, trax);
          ++count_true_object;
        } else {
          // get center of mass for the current object
          typedef acc::Coord<acc::Mean> CoordMeanType;
          std::vector<float> com(
            acc::get<CoordMeanType>(accu_chain, i).begin(),
            acc::get<CoordMeanType>(accu_chain, i).end());
          if (com.size() == 2) {
            com.push_back(0);
          } else {
            throw std::runtime_error("Wrong dimension of COM");
          }
          // write size and com into a feature map
          pgmlink::FeatureMap f_map;
          f_map["com"] = com;
          f_map["count"].push_back(size);
          // create the traxel and add it to the feature store
          pgmlink::Traxel trax(i, timestep, f_map);
          pgmlink::add(ts, trax);
        }
      }
      // typedef for convenience
      typedef vigra::MultiArray<2, unsigned short> UShortMatrixType;
      // save the segmentation
      vigra::exportImage(
        vigra::srcImageRange(UShortMatrixType(label_image)),
        vigra::ImageExportInfo(segmentation_result_path.str().c_str()));
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
