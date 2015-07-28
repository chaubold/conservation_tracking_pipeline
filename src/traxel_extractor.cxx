/* TODO
 * implement border width
 * merge division feature extractor with traxel extractor
 */

#include "traxel_extractor.hxx"
#include <iostream>

namespace isbi_pipeline {

template<typename T>
void dump(const std::vector<T> vector) {
  std::cout << "(";
  for(float entry : vector) {
    std::cout << entry << ", ";
  }
  std::cout << ")";
}

void dump(const FeatureMapType& feature_map) {
  for(
    FeatureMapType::const_iterator f_it = feature_map.begin();
    f_it != feature_map.end();
    f_it++)
  {
    std::cout << f_it->first << ":";
    dump(f_it->second);
    std::cout << std::endl;
  }
}

template<typename T>
void set_feature(FeatureMapType& feature_map, const std::string& name, T value) {
  FeatureArrayType f;
  f.push_back(static_cast<FeatureType>(value));
  feature_map[name] = f;
}

template<>
void set_feature(
  FeatureMapType& feature_map,
  const std::string& name,
  vigra::TinyVector<double, 2> value)
{
  FeatureArrayType f;
  f.clear();
  f.push_back(static_cast<FeatureType>(value[0]));
  f.push_back(static_cast<FeatureType>(value[1]));
  feature_map[name] = f;
}

template<>
void set_feature(
  FeatureMapType& feature_map,
  const std::string& name,
  vigra::TinyVector<double, 3> value)
{
  FeatureArrayType f;
  f.clear();
  f.push_back(static_cast<FeatureType>(value[0]));
  f.push_back(static_cast<FeatureType>(value[1]));
  f.push_back(static_cast<FeatureType>(value[2]));
  feature_map[name] = f;
}

template<int N, typename T1, typename T2>
void set_feature_with_offset(
  FeatureMapType& feature_map,
  const std::string& name,
  vigra::TinyVector<T1,N> value,
  vigra::TinyVector<T2,N> offset)
{
  FeatureArrayType f;
  f.clear();
  for(int i = 0; i < N; i++) {
    f.push_back(static_cast<FeatureType>(value[i] + offset[i]));
  }
  feature_map[name] = f;
}

template<>
void set_feature(
  FeatureMapType& feature_map,
  const std::string& name,
  vigra::linalg::Matrix<DataType> value)
{
  FeatureArrayType f;
  f.clear();
  for(auto it = value.begin(); it != value.end(); ++it) {
    f.push_back(static_cast<FeatureType>(*it));
  }
  feature_map[name] = f;
}

template<unsigned int N, typename T>
void copy_vector(
  const std::vector<T>& vector,
  vigra::MultiArrayView<N, T>& multi_array_view)
{
  typename std::vector<T>::const_iterator v_it = vector.begin();
  typename vigra::MultiArrayView<N, T>::iterator m_it = multi_array_view.begin();
  for (; v_it != vector.end(); v_it++, m_it++) {
    *m_it = *v_it;
  }
}

////
//// class TraxelExtractor
////
template<int N>
TraxelExtractor<N>::TraxelExtractor(
    const std::vector<std::string> feature_selection,
    const RandomForestVectorType& random_forests,
    const TrackingOptions& options) :
  feature_selection_(feature_selection),
  random_forests_(random_forests),
  options_(options)
{
  // assertions?
  if (!options.get_option<std::string>("tracker").compare("ConsTracking")) {
    max_object_num_ = options.get_option<int>("maxObj");
  } else {
    max_object_num_ = 1;
  }
  border_distance_ = options.get_option<int>("borderWidth");
  lower_size_lim_ = options.get_option<int>("size_range_0");
  upper_size_lim_ = options.get_option<int>("size_range_1");
  x_scale_ = options.get_option<double>("scales_0");
  y_scale_ = options.get_option<double>("scales_1");
  z_scale_ = options.get_option<double>("scales_2");
}

template<int N>
int TraxelExtractor<N>::extract(
  const Segmentation<N>& segmentation,
  const vigra::MultiArrayView<N, DataType>& image,
  const int timestep,
  TraxelVectorType& traxels) const
{
  traxels.clear();
  int return_status = 0;
  // initialize the accumulator chain
  AccChainType acc_chain;
  acc_chain.ignoreLabel(0);
  // always enable com, count, mean and bounding box
  acc_chain.template activate<acc::RegionCenter>();
  acc_chain.template activate<acc::Count>();
  acc_chain.template activate<acc::Mean>();
  acc_chain.template activate<acc::Variance>();
  acc_chain.template activate<acc::Coord<acc::Minimum> >();
  acc_chain.template activate<acc::Coord<acc::Maximum> >();
  // select the other features
  select_features(acc_chain);
  // initialize the coupled iterator
  CoupledIteratorType start_it = vigra::createCoupledIterator(
    segmentation.label_image_,
    image);
  CoupledIteratorType end_it = start_it.getEndIterator();
  // extract the features
  vigra::acc::extractFeatures(start_it, end_it, acc_chain);
  // loop over all labels
  for (size_t label_id = 1; label_id <= segmentation.label_count_; label_id++) {
    extract_for_label(
      acc_chain,
      label_id,
      timestep,
      traxels);
  }
  return return_status;
}

template<int N>
int TraxelExtractor<N>::extract_for_label(
  const AccChainType& acc_chain,
  const size_t label_id,
  const int timestep,
  TraxelVectorType& traxels) const
{
  int return_status = 0;
  // get the feature map
  FeatureMapType feature_map;
  fill_feature_map(acc_chain, label_id, feature_map);
  // get the object size
  DataType size = acc::get<acc::Count>(acc_chain, label_id);
  // filter size
  bool fits_lower_lim = ((lower_size_lim_ == 0) or (size >= lower_size_lim_));
  bool fits_upper_lim = ((upper_size_lim_ == 0) or (size <= upper_size_lim_));
  if (fits_upper_lim and fits_lower_lim) {
    // get the count (for tracking and divion feature calculation)
    feature_map["count"].push_back(size);
    if (feature_map.count("Count") == 0) {
      feature_map["Count"].push_back(size);
    }
    // get the region center (maybe once again)
    set_feature(
      feature_map,
      "com",
      acc::get<acc::RegionCenter>(acc_chain, label_id));
    if (feature_map.count("RegionCenter") == 0) {
      set_feature(
        feature_map,
        "RegionCenter",
        acc::get<acc::RegionCenter>(acc_chain, label_id));
    }
    // get the bounding box
    set_feature(
      feature_map,
      "CoordMin",
      acc::get<acc::Coord<acc::Minimum> >(acc_chain, label_id));
    set_feature(
      feature_map,
      "CoordMax",
      acc::get<acc::Coord<acc::Maximum> >(acc_chain, label_id));
    if (feature_map["com"].size() == 2) {
      feature_map["com"].push_back(0.0);
    }
    // get the mean
    if (feature_map.count("Mean") == 0) {
      set_feature(
        feature_map,
        "Mean",
        acc::get<acc::Mean>(acc_chain, label_id));
    }
    // get the variance
    if (feature_map.count("Variance") == 0) {
      set_feature(
        feature_map,
        "Variance",
        acc::get<acc::Variance>(acc_chain, label_id));
    }
    if (random_forests_.size() > 0) {
      get_detection_probability(feature_map);
    }
    // Its ok to use "new" since the Traxel class handles the
    // destruction of the locator
    typedef pgmlink::ComLocator LocatorType;
    LocatorType* l_ptr = new LocatorType;
    l_ptr->x_scale = x_scale_;
    l_ptr->y_scale = y_scale_;
    l_ptr->z_scale = z_scale_;
    // create the traxel and add it to the traxel store
    pgmlink::Traxel traxel(label_id, timestep, feature_map, l_ptr);
    traxels.push_back(traxel);
  }
  return return_status;
}

template<int N>
int TraxelExtractor<N>::select_features(
  AccChainType& acc_chain) const
{
  int ret = 0;
  for (std::string feature : feature_selection_) {
    if (!feature.compare("Coord<Principal<Kurtosis> >")) {
      acc_chain.template activate<acc::Coord<acc::Principal<acc::Kurtosis> > >();
    } else if (!feature.compare("Coord<Principal<Skewness> >")) {
      acc_chain.template activate<acc::Coord<acc::Principal<acc::Skewness> > >();
    } else if (!feature.compare("Count")) {
      acc_chain.template activate<acc::Count>();
    } else if (!feature.compare("Kurtosis")) {
      acc_chain.template activate<acc::Kurtosis>();
    } else if (!feature.compare("Maximum")) {
      acc_chain.template activate<acc::Maximum>();
    } else if (!feature.compare("Mean")) {
      acc_chain.template activate<acc::Mean>();
    } else if (!feature.compare("Minimum")) {
      acc_chain.template activate<acc::Minimum>();
    } else if (!feature.compare("RegionCenter")) {
      acc_chain.template activate<acc::RegionCenter>();
    } else if (!feature.compare("RegionRadii")) {
      acc_chain.template activate<acc::RegionRadii>();
    } else if (!feature.compare("Skewness")) {
      acc_chain.template activate<acc::Skewness>();
    } else if (!feature.compare("Sum")) {
      acc_chain.template activate<acc::Sum>();
    } else if (!feature.compare("Variance")) {
      acc_chain.template activate<acc::Variance>();
    } else {
      throw std::runtime_error("Unknown feature \"" + feature + "\"");
      ret = 1;
    }
  }
  return ret;
}

template<int N>
int TraxelExtractor<N>::fill_feature_map(
  const AccChainType& acc_chain,
  const size_t label_id,
  FeatureMapType& feature_map) const
{
  int ret = 0;
  // local typedefs
  typedef acc::Coord<acc::Principal<acc::Kurtosis> > CoordPK;
  typedef acc::Coord<acc::Principal<acc::Skewness> > CoordPS;
  for (std::string feature : feature_selection_) {
    if (!feature.compare("Coord<Principal<Kurtosis> >")) {
      set_feature(
        feature_map,
        feature,
        acc::get<CoordPK>(acc_chain, label_id));
    } else if (!feature.compare("Coord<Principal<Skewness> >")) {
      set_feature(
        feature_map,
        feature,
        acc::get<CoordPS>(acc_chain, label_id));
    } else if (!feature.compare("Count")) {
      set_feature(
        feature_map,
        feature,
        acc::get<acc::Count>(acc_chain, label_id));
    } else if (!feature.compare("Kurtosis")) {
      set_feature(
        feature_map,
        feature,
        acc::get<acc::Kurtosis>(acc_chain, label_id));
    } else if (!feature.compare("Maximum")) {
      set_feature(
        feature_map,
        feature,
        acc::get<acc::Maximum>(acc_chain, label_id));
    } else if (!feature.compare("Mean")) {
      set_feature(
        feature_map,
        feature,
        acc::get<acc::Mean>(acc_chain, label_id));
    } else if (!feature.compare("Minimum")) {
      set_feature(
        feature_map,
        feature,
        acc::get<acc::Minimum>(acc_chain, label_id));
    } else if (!feature.compare("RegionCenter")) {
      set_feature(
        feature_map,
        feature,
        acc::get<acc::RegionCenter>(acc_chain, label_id));
      if (feature_map["com"].size() == 2) {
        feature_map["com"].push_back(0.0);
      }
    } else if (!feature.compare("RegionRadii")) {
      set_feature(
        feature_map,
        feature,
        acc::get<acc::RegionRadii>(acc_chain, label_id));
    } else if (!feature.compare("Skewness")) {
      set_feature(
        feature_map,
        feature,
        acc::get<acc::Skewness>(acc_chain, label_id));
    } else if (!feature.compare("Sum")) {
      set_feature(
        feature_map,
        feature,
        acc::get<acc::Sum>(acc_chain, label_id));
    } else if (!feature.compare("Variance")) {
      set_feature(
        feature_map,
        feature,
        acc::get<acc::Variance>(acc_chain, label_id));
    } else {
      throw std::runtime_error("Unknown region feature \"" + feature + "\"");
      ret = 1;
    }
  }
  return ret;
}

template<int N>
int TraxelExtractor<N>::get_detection_probability(
  FeatureMapType& feature_map) const
{
  if(random_forests_.size() < 1){
    throw std::runtime_error("Cannot extract detection probability without RF");
  }
  // get the size of the feature vector
  size_t feature_size = 0;
  for(std::string feature : feature_selection_) {
    FeatureMapType::const_iterator f_it = feature_map.find(feature);
    if (f_it == feature_map.end()) {
      throw std::runtime_error("Feature \"" + feature + "\" not found");
    } else {
      feature_size += (f_it->second).size();
    }
  }
  // get all features into one multi array
  size_t offset = 0;
  vigra::MultiArray<2, FeatureType> features(vigra::Shape2(1, feature_size));
  for(std::string feature : feature_selection_) {
    FeatureMapType::const_iterator f_it = feature_map.find(feature);
    size_t size = (f_it->second).size();
    vigra::MultiArrayView<2, FeatureType> features_view = features.subarray(
      vigra::Shape2(0, offset),
      vigra::Shape2(1, offset+size));
    copy_vector(f_it->second, features_view);
    offset += size;
  }
  // evaluate the random forests
  size_t num_classes = random_forests_[0].ext_param_.class_count_;
  vigra::MultiArray<2, FeatureType> probabilities(
    vigra::Shape2(1, num_classes),
    0.0);
  for (size_t n = 0; n < random_forests_.size(); n++) {
    vigra::MultiArray<2, FeatureType> probabilities_temp(
      vigra::Shape2(1, num_classes));
    random_forests_[n].predictProbabilities(features, probabilities_temp);
    probabilities += probabilities_temp;
  }
  // fill the feature map
  FeatureArrayType& det_array = feature_map["detProb"];
  det_array.clear();
  for (auto it = probabilities.begin(); it != probabilities.end(); it++) {
    det_array.push_back(*it);
  }
  return 0;
}

// explicit instantiation
template class TraxelExtractor<2>;
template class TraxelExtractor<3>;

} // end of namespace isbi_pipeline
