// vigra
#include <vigra/labelimage.hxx> /* for labelImageWithBackground */
#include <vigra/multi_tensorutilities.hxx>
#include <vigra/hdf5impex.hxx> /* for writeHDF5 */

#include "segmentation.hxx"

#ifdef NDEBUG
  #define LOG(message)
#else
  #include <iostream>
  #define LOG(message) std::cout << message << std::endl
#endif

namespace isbi_pipeline {

////
//// class FeatureCalculator
////
template<int N>
FeatureCalculator<N>::FeatureCalculator(
    const StringDoublePairVectorType& feature_scales,
    double window_size) :
  feature_scales_(feature_scales),
  window_size_(window_size)
{
  // initialize the feature dimension map
  feature_sizes_["GaussianSmoothing"] = 1;
  feature_sizes_["LaplacianOfGaussians"] = 1;
  feature_sizes_["StructureTensorEigenvalues"] = N;
  feature_sizes_["HessianOfGaussianEigenvalues"] = N;
  feature_sizes_["GaussianGradientMagnitude"] = 1;
  feature_sizes_["DifferenceOfGaussians"] = 1;
  // set the filter window size
  conv_options_.filterWindowSize(window_size);
}

template<int N>
size_t FeatureCalculator<N>::get_feature_size(
  const std::string& feature_name
) const {
  std::map<std::string, size_t>::const_iterator it;
  it = feature_sizes_.find(feature_name);
  if (it == feature_sizes_.end()) {
    return 0;
  } else {
    return it->second;
  }
}

template<int N>
size_t FeatureCalculator<N>::get_feature_size() const {
  size_t size = 0;
  for (
    StringDoublePairVectorType::const_iterator it = feature_scales_.begin();
    it != feature_scales_.end();
    it++
  ) {
    size += get_feature_size(it->first);
  }
  return size;
}

template<int N>
int FeatureCalculator<N>::calculate_gaussian_smoothing(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArrayView<N+1, DataType>& features,
  double feature_scale) const
{
  LOG("calculate gaussian smoothing (" << feature_scale << ")");
  vigra::MultiArrayView<N, DataType> results(features.bindAt(N,0));
  vigra::gaussianSmoothMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(results),
    feature_scale,
    conv_options_);
  return 0;
}

template<int N>
int FeatureCalculator<N>::calculate_laplacian_of_gaussians(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArrayView<N+1, DataType>& features,
  double feature_scale) const
{
  LOG("calculate laplacian of gaussians (" << feature_scale << ")");
  vigra::MultiArrayView<N, DataType> results(features.bindAt(N,0));
  vigra::laplacianOfGaussianMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(results),
    feature_scale,
    conv_options_);
  return 0;
}

template<int N>
int FeatureCalculator<N>::calculate_gaussian_gradient_magnitude(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArrayView<N+1, DataType>& features,
  double feature_scale) const
{
  LOG("calculate gaussian gradient magnitude (" << feature_scale << ")");
  vigra::MultiArrayView<N, DataType> results(features.bindAt(2, 0));
  vigra::MultiArray<N, vigra::TinyVector<DataType, N> > temp(image.shape());
  vigra::gaussianGradientMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(temp),
    feature_scale,
    conv_options_);
  vigra::VectorNormFunctor<vigra::TinyVector<DataType, N> > norm;
  vigra::transformMultiArray(
    srcMultiArrayRange(temp),
    destMultiArray(results),
    norm);
  return 0;
}

template<int N>
int FeatureCalculator<N>::calculate_difference_of_gaussians(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArrayView<N+1, DataType>& features,
  double feature_scale) const
{
  LOG("calculate difference of gaussians (" << feature_scale << ")");
  vigra::MultiArray<N, DataType> temp(image.shape());
  vigra::MultiArrayView<N, DataType> results(features.bindAt(N,0));
  vigra::gaussianSmoothMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(results),
    feature_scale,
    conv_options_);
  vigra::gaussianSmoothMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(temp),
    feature_scale * 0.66,
    conv_options_);
  results -= temp;
  return 0;
}

template<int N>
int FeatureCalculator<N>::calculate_structure_tensor_eigenvalues(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArrayView<N+1, DataType>& features,
  double feature_scale) const
{
  LOG("calculate structure tensor eigenvalues (" << feature_scale << ")");
  vigra::MultiArray<N, vigra::TinyVector<DataType, (N*(N+1))/2> > tensor(
    image.shape());
  vigra::MultiArray<N, vigra::TinyVector<DataType, N> > eigenvalues(
    image.shape());
  vigra::structureTensorMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(tensor),
    feature_scale,
    feature_scale * 0.5,
    conv_options_);
  vigra::tensorEigenvaluesMultiArray(
    srcMultiArrayRange(tensor),
    destMultiArray(eigenvalues));
  features = eigenvalues.expandElements(N);
  return 0;
}

template<int N>
int FeatureCalculator<N>::calculate_hessian_of_gaussian_eigenvalues(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArrayView<N+1, DataType>& features,
  double feature_scale) const
{
  LOG("calculate hessian of gaussian eigenvalues (" << feature_scale << ")");
  vigra::MultiArray<N, vigra::TinyVector<DataType, (N*(N+1))/2> > hessian(
    image.shape());
  vigra::MultiArray<N, vigra::TinyVector<DataType, N> > eigenvalues(
    image.shape());
  vigra::hessianOfGaussianMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(hessian),
    feature_scale,
    conv_options_);
  vigra::tensorEigenvaluesMultiArray(
    srcMultiArrayRange(hessian),
    destMultiArray(eigenvalues));
  features = eigenvalues.expandElements(N);
  return 0;
}

template<>
int FeatureCalculator<2>::calculate(
  const vigra::MultiArrayView<2, DataType>& image,
  vigra::MultiArray<3, DataType>& features)
{
  // initialize offset and size of the current features along the
  // feature vectors
  size_t offset = 0;
  size_t size = 0;
  features.reshape(
    vigra::Shape3(image.shape(0), image.shape(1), get_feature_size()));
  for (
    StringDoublePairVectorType::const_iterator it = feature_scales_.begin();
    it != feature_scales_.end();
    it++
  ) {
    // get the size of the current feature and create a view of all
    // feature vectors
    size = get_feature_size(it->first);
    vigra::MultiArrayView<3, DataType> features_v;
    features_v = features.subarray(
      vigra::Shape3(0, 0, offset),
      vigra::Shape3(image.shape(0), image.shape(1), offset + size));
    if (!it->first.compare("GaussianSmoothing")) {
      calculate_gaussian_smoothing(image, features_v, it->second);
    } else if (!it->first.compare("LaplacianOfGaussians")) {
      calculate_laplacian_of_gaussians(image, features_v, it->second);
    } else if (!it->first.compare("GaussianGradientMagnitude")) {
      calculate_gaussian_gradient_magnitude(image, features_v, it->second);
    } else if (!it->first.compare("DifferenceOfGaussians")) {
      calculate_difference_of_gaussians(image, features_v, it->second);
    } else if (!it->first.compare("StructureTensorEigenvalues")) {
      calculate_structure_tensor_eigenvalues(image, features_v, it->second);
    } else if (!it->first.compare("HessianOfGaussianEigenvalues")) {
      calculate_hessian_of_gaussian_eigenvalues(image, features_v, it->second);
    } else {
      return 1;
    }
    offset += size;
  }
  LOG("Feature calculation done");
  return 0;
}

template<>
int FeatureCalculator<3>::calculate(
  const vigra::MultiArrayView<3, DataType>& image,
  vigra::MultiArray<4, DataType>& features)
{
  // initialize offset and size of the current features along the
  // feature vectors
  size_t offset = 0;
  size_t size = 0;
  features.reshape(vigra::Shape4(
    image.shape(0), image.shape(1), image.shape(2), get_feature_size()));
  for (
    StringDoublePairVectorType::const_iterator it = feature_scales_.begin();
    it != feature_scales_.end();
    it++
  ) {
    // get the size of the current feature and create a view of all
    // feature vectors
    size = get_feature_size(it->first);
    vigra::MultiArrayView<4, DataType> features_v;
    features_v = features.subarray(
      vigra::Shape4(0, 0, 0, offset),
      vigra::Shape4(image.shape(0), image.shape(1), image.shape(2), offset+size));
    if (!it->first.compare("GaussianSmoothing")) {
      calculate_gaussian_smoothing(image, features_v, it->second);
    } else if (!it->first.compare("LaplacianOfGaussians")) {
      calculate_laplacian_of_gaussians(image, features_v, it->second);
    } else if (!it->first.compare("GaussianGradientMagnitude")) {
      calculate_gaussian_gradient_magnitude(image, features_v, it->second);
    } else if (!it->first.compare("DifferenceOfGaussians")) {
      calculate_difference_of_gaussians(image, features_v, it->second);
    } else if (!it->first.compare("StructureTensorEigenvalues")) {
      calculate_structure_tensor_eigenvalues(image, features_v, it->second);
    } else if (!it->first.compare("HessianOfGaussianEigenvalues")) {
      calculate_hessian_of_gaussian_eigenvalues(image, features_v, it->second);
    } else {
      return 1;
    }
    offset += size;
  }
  return 0;
}

// explicit instantiation
template class FeatureCalculator<2>;

////
//// struct Segmentation
////
template<int N>
void Segmentation<N>::initialize(const vigra::MultiArray<N, DataType>& image) {
  segmentation_image_.reshape(image.shape());
  label_image_.reshape(image.shape());
  // TODO ugly
  typename vigra::MultiArrayShape<N+1>::type prediction_map_shape;
  for (size_t i = 0; i < N; i++) {
    prediction_map_shape[i] = image.shape(i);
  }
  prediction_map_shape[N] = 2;
  prediction_map_.reshape(prediction_map_shape);
  prediction_map_.init(0.0);
}

template<int N>
int Segmentation<N>::export_hdf5(const std::string filename) {
  vigra::writeHDF5(filename.c_str(), "/segmentation/segmentation", segmentation_image_);
  vigra::writeHDF5(filename.c_str(), "/segmentation/labels", label_image_);
  vigra::writeHDF5(filename.c_str(), "/segmentation/features", feature_image_);
  vigra::writeHDF5(
    filename.c_str(),
    "/segmentation/prediction_map",
    prediction_map_);
  return 0;
}

template<int N>
int Segmentation<N>::read_hdf5(const std::string filename) {
  // get all import infos
  vigra::HDF5ImportInfo segmentation_info(
    filename.c_str(),
    "/segmentation/segmentation");
  vigra::HDF5ImportInfo labels_info(
    filename.c_str(),
    "/segmentation/labels");
  vigra::HDF5ImportInfo features_info(
    filename.c_str(),
    "/segmentation/features");
  vigra::HDF5ImportInfo prediction_map_info(
    filename.c_str(),
    "/segmentation/prediction_map");
  // assert sizes
  if (segmentation_info.numDimensions() != N) {
    throw std::runtime_error("Segmentation image has wrong dimension");
  }
  if (labels_info.numDimensions() != N) {
    throw std::runtime_error("Label image has wrong dimension");
  }
  if (features_info.numDimensions() != N+1) {
    throw std::runtime_error("Feature image has wrong dimension");
  }
  if (prediction_map_info.numDimensions() != N+1) {
    throw std::runtime_error("Prediction map has wrong dimension");
  }
  // reshape multi arrays
  typedef typename vigra::MultiArrayShape<N>::type NShapeType;
  typedef typename vigra::MultiArrayShape<N+1>::type N1ShapeType;
  NShapeType seg_shape(segmentation_info.shape().begin());
  segmentation_image_.reshape(seg_shape);
  NShapeType labels_shape(labels_info.shape().begin());
  label_image_.reshape(labels_shape);
  N1ShapeType features_shape(features_info.shape().begin());
  feature_image_.reshape(features_shape);
  N1ShapeType prediction_map_shape(prediction_map_info.shape().begin());
  prediction_map_.reshape(prediction_map_shape);
  // Debug
  LOG("Shape of segmentation image: " << seg_shape);
  LOG("Shape of label image: " << labels_shape);
  LOG("Shape of feature image: " << features_shape);
  LOG("Shape of prediction map: " << prediction_map_shape);
  // load data
  vigra::readHDF5(segmentation_info, segmentation_image_);
  vigra::readHDF5(labels_info, label_image_);
  vigra::readHDF5(features_info, feature_image_);
  vigra::readHDF5(prediction_map_info, prediction_map_);
  // read label count
  unsigned int min, max;
  label_image_.minmax(&min, &max);
  label_count_ = max;
  return 0;
}

// explicit instantiation
// TODO for dim = 3 as well
template class Segmentation<2>;

////
//// class SegmentationCalculator
////
template<int N>
SegmentationCalculator<N>::SegmentationCalculator(
    boost::shared_ptr<FeatureCalculator<N> > feature_calculator_ptr,
    const std::vector<RandomForestType>& random_forests) :
  feature_calculator_ptr_(feature_calculator_ptr),
  random_forests_(random_forests)
{
  // assertions?
}

template<int N>
int SegmentationCalculator<N>::calculate(
  const vigra::MultiArray<N, DataType>& image,
  Segmentation<N>& segmentation) const
{
  int return_status = 0;
  // initialize the segmentation
  segmentation.initialize(image);
  // calculate the features and reshape them
  feature_calculator_ptr_->calculate(image, segmentation.feature_image_);
  // get the count of pixels in image
  size_t pixel_count = 1;
  for (size_t dim = 0; dim < N; dim++) {
    pixel_count *= image.shape(dim);
  }
  // get the feature dimension
  size_t feature_dim = segmentation.feature_image_.shape(N);
  // create a view with all image dimensions flattened
  vigra::MultiArrayView<2, DataType> feature_view(
    vigra::Shape2(pixel_count, feature_dim),
    segmentation.feature_image_.data());
  vigra::MultiArrayView<2, DataType> prediction_map_view(
    vigra::Shape2(pixel_count, 2),
    segmentation.prediction_map_.data());
  vigra::MultiArray<2, DataType> prediction_temp(pixel_count, 2);
  // loop over all random forests for prediction probabilities
  LOG("Evaluate random forests");
  for (
    std::vector<RandomForestType>::const_iterator it = random_forests_.begin();
    it != random_forests_.end();
    it++
  ) {
    it->predictProbabilities(
      feature_view,
      prediction_temp);
    prediction_map_view += prediction_temp;
  }
  LOG("Assign the segmentation labels");
  // assign the labels
  typename vigra::MultiArrayView<N, unsigned>::iterator seg_it;
  seg_it = segmentation.segmentation_image_.begin();
  for (size_t n = 0; n < pixel_count; n++, seg_it++) {
    if (prediction_map_view(n, 1) > prediction_map_view(n, 0)) {
      *seg_it = 1;
    } else {
      *seg_it = 0;
    }
  }
  // extract objects
  segmentation.label_count_ = vigra::labelImageWithBackground(
    vigra::srcImageRange(segmentation.segmentation_image_),
    vigra::destImage(segmentation.label_image_),
    1,
    0);
  LOG("Found " << segmentation.label_count_ << " objects");
  // TODO:
  // if (options.count("border") > 0) {
  //   ignore_border_cc<2>(label_image, options["border"]);
  // }
  // done
  return return_status;
}

// explicit instantiation
template class SegmentationCalculator<2>;

////
//// class TraxelExtractor
////
template<int N>
TraxelExtractor<N>::TraxelExtractor(
    unsigned int border_distance,
    unsigned int lower_size_lim,
    unsigned int upper_size_lim) :
  border_distance_(border_distance),
  lower_size_lim_(lower_size_lim),
  upper_size_lim_(upper_size_lim)
{
  // assertions?
}

template<int N>
int TraxelExtractor<N>::extract(
  const Segmentation<N>& segmentation,
  const int timestep,
  pgmlink::TraxelStore& traxelstore) const
{
  LOG("Extract traxel");
  int return_status = 0;
  // calculate the size and coordinate center of mass
  AccChainType acc_chain;
  CoupledIteratorType start_it = vigra::createCoupledIterator(
    segmentation.label_image_,
    segmentation.label_image_);
  CoupledIteratorType end_it = start_it.getEndIterator();
  vigra::acc::extractFeatures(start_it, end_it, acc_chain);
  // loop over all labels
  for (size_t label_id = 1; label_id <= segmentation.label_count_; label_id++) {
    extract_for_label(acc_chain, label_id, timestep, traxelstore);
  }
  return return_status;
}

template<int N>
int TraxelExtractor<N>::extract_for_label(
  const AccChainType& acc_chain,
  const size_t label_id,
  const int timestep,
  pgmlink::TraxelStore& traxelstore) const
{
  // TODO border filter
  typedef vigra::acc::Coord<vigra::acc::Mean> CoordMeanType;
  int return_status = 0;
  // get the object size
  float size = vigra::acc::get<vigra::acc::Count>(acc_chain, label_id);
  // filter size
  bool fits_lower_lim = ((lower_size_lim_ == 0) or (size >= lower_size_lim_));
  bool fits_upper_lim = ((upper_size_lim_ == 0) or (size <= upper_size_lim_));
  if (fits_upper_lim and fits_lower_lim) {
    // get the com
    std::vector<float> com(
      vigra::acc::get<CoordMeanType>(acc_chain, label_id).begin(),
      vigra::acc::get<CoordMeanType>(acc_chain, label_id).end());
    if (N == 2) {
      com.push_back(0.0);
    }
    // fill the feature map
    pgmlink::FeatureMap feature_map;
    feature_map["com"] = com;
    feature_map["count"].push_back(size);
    // create the traxel and add it to the traxel store
    pgmlink::Traxel traxel(label_id, timestep, feature_map);
    pgmlink::add(traxelstore, traxel);
  }
  return return_status;
}

// explicit instantiation
template class TraxelExtractor<2>;

} // namespace isbi_pipeline
