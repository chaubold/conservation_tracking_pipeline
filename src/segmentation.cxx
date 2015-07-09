// vigra
#include <vigra/multi_labeling.hxx> /* for labelMultiArrayWithBackground */
#include <vigra/multi_tensorutilities.hxx>
#include <vigra/hdf5impex.hxx> /* for writeHDF5 */

#include "segmentation.hxx"

namespace isbi_pipeline {

////
//// local functions
////
template<int N>
typename vigra::MultiArrayShape<N+1>::type append_to_shape(
  const typename vigra::MultiArrayShape<N>::type& shape,
  const size_t value)
{
  typename vigra::MultiArrayShape<N+1>::type ret;
  for (size_t i = 0; i < N; i++) {
    ret[i] = shape[i];
  }
  ret[N] = value;
  return ret;
}


////
//// class FeatureCalculator
////
template<int N>
FeatureCalculator<N>::FeatureCalculator(
    const StringDataPairVectorType& feature_scales,
    DataType window_size) :
  feature_scales_(feature_scales),
  window_size_(window_size)
{
  // initialize the feature dimension map
  feature_sizes_["GaussianSmoothing"] = 1;
  feature_sizes_["LaplacianOfGaussian"] = 1;
  feature_sizes_["StructureTensorEigenvalues"] = N;
  feature_sizes_["HessianOfGaussianEigenvalues"] = N;
  feature_sizes_["GaussianGradientMagnitude"] = 1;
  feature_sizes_["DifferenceOfGaussians"] = 1;
  // set the filter window size
  conv_options_.filterWindowSize(window_size);
}

template<int N>
FeatureCalculator<N>::FeatureCalculator(
    const StringDataPairVectorType& feature_scales,
    const vigra::TinyVector<DataType, N> image_scales,
    DataType window_size) :
  FeatureCalculator(feature_scales, window_size)
{
  conv_options_.stepSize(image_scales);
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
    StringDataPairVectorType::const_iterator it = feature_scales_.begin();
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
  DataType feature_scale) const
{
  vigra::MultiArrayView<N, DataType> results(features.template bind<N>(0));
  vigra::gaussianSmoothMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(results),
    feature_scale,
    conv_options_);
  return 0;
}

template<int N>
int FeatureCalculator<N>::calculate_laplacian_of_gaussian(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArrayView<N+1, DataType>& features,
  DataType feature_scale) const
{
  vigra::MultiArrayView<N, DataType> results(features.template bind<N>(0));
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
  DataType feature_scale)
{
  vigra::MultiArrayView<N, DataType> results(features.template bind<N>(0));
  vigra::VectorNormFunctor<vigra::TinyVector<DataType, N> > norm;
#ifndef USE_PARALLEL_FEATURES
  // reuse eigenvalue_temp
  if (eigenvalue_temp_.shape() != image.shape()) {
    eigenvalue_temp_.reshape(image.shape());
  }
  vigra::gaussianGradientMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(eigenvalue_temp_),
    feature_scale,
    conv_options_);
  vigra::transformMultiArray(
    srcMultiArrayRange(eigenvalue_temp_),
    destMultiArray(results),
    norm);
#else
  vigra::MultiArray<N, vigra::TinyVector<DataType, N> > temp(image.shape());
  vigra::gaussianGradientMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(temp),
    feature_scale,
    conv_options_);
  vigra::transformMultiArray(
    srcMultiArrayRange(temp),
    destMultiArray(results),
    norm);
#endif
  return 0;
}

template<int N>
int FeatureCalculator<N>::calculate_difference_of_gaussians(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArrayView<N+1, DataType>& features,
  DataType feature_scale)
{
  vigra::MultiArrayView<N, DataType> results(features.template bind<N>(0));
  vigra::gaussianSmoothMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(results),
    feature_scale,
    conv_options_);
#ifndef USE_PARALLEL_FEATURES
  if (feature_temp_.shape() != image.shape()) {
    feature_temp_.reshape(image.shape());
  }
  vigra::gaussianSmoothMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(feature_temp_),
    feature_scale * 0.66,
    conv_options_);
  results -= feature_temp_;
#else
  vigra::MultiArray<N, DataType> temp(image.shape());
  vigra::gaussianSmoothMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(temp),
    feature_scale * 0.66,
    conv_options_);
  results -= temp;
#endif
  return 0;
}

template<int N>
int FeatureCalculator<N>::calculate_structure_tensor_eigenvalues(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArrayView<N+1, DataType>& features,
  DataType feature_scale)
{
#ifndef USE_PARALLEL_FEATURES
  if (tensor_temp_.shape() != image.shape()) {
    tensor_temp_.reshape(image.shape());
  }
  if (eigenvalue_temp_.shape() != image.shape()) {
    eigenvalue_temp_.reshape(image.shape());
  }
  vigra::structureTensorMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(tensor_temp_),
    feature_scale,
    feature_scale * 0.5,
    conv_options_);
  vigra::tensorEigenvaluesMultiArray(
    srcMultiArrayRange(tensor_temp_),
    destMultiArray(eigenvalue_temp_));
  features = eigenvalue_temp_.expandElements(N);
#else
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
#endif
  return 0;
}

template<int N>
int FeatureCalculator<N>::calculate_hessian_of_gaussian_eigenvalues(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArrayView<N+1, DataType>& features,
  DataType feature_scale)
{
#ifndef USE_PARALLEL_FEATURES
  // using tensor as hessian here
  if (tensor_temp_.shape() != image.shape()) {
    tensor_temp_.reshape(image.shape());
  }
  if (eigenvalue_temp_.shape() != image.shape()) {
    eigenvalue_temp_.reshape(image.shape());
  }
  vigra::hessianOfGaussianMultiArray(
    srcMultiArrayRange(image),
    destMultiArray(tensor_temp_),
    feature_scale,
    conv_options_);
  vigra::tensorEigenvaluesMultiArray(
    srcMultiArrayRange(tensor_temp_),
    destMultiArray(eigenvalue_temp_));
  features = eigenvalue_temp_.expandElements(N);
#else
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
#endif
  return 0;
}

template<int N>
int FeatureCalculator<N>::calculate(
  const vigra::MultiArrayView<N, DataType>& image,
  vigra::MultiArray<N+1, DataType>& features)
{
  std::cout << "\tcalculating " << get_feature_size() << " features" << std::endl;
  typedef typename vigra::MultiArrayShape<N+1>::type FeaturesShapeType;
  typedef typename vigra::MultiArrayShape<N>::type ImageShapeType;
  typedef typename vigra::MultiArrayView<N+1, DataType> FeaturesViewType;
  // initialize offset and size of the current features along the
  // feature vectors
  std::vector<size_t> offsets;
  FeaturesShapeType features_shape = append_to_shape<N>(
    image.shape(),
    get_feature_size());
  if (features.shape() != features_shape) {
    features.reshape(features_shape);
  }
  
  // store all offsets and keep scope of variable offset within the
  // for loop
  {
    size_t offset = 0;
    for (
      StringDataPairVectorType::const_iterator it = feature_scales_.begin();
      it != feature_scales_.end();
      it++
    ) {
      offsets.push_back(offset);
      offset += get_feature_size(it->first);
    }
  }

#ifdef USE_PARALLEL_FEATURES
  #pragma omp parallel for
#endif
  for(size_t i = 0; i < feature_scales_.size(); i++) {
    // get the offset and size of the current feature in the feature
    // arrays
    const size_t& offset = offsets[i];
    const size_t& size = get_feature_size(feature_scales_[i].first);
    // create the bounding box from box_min to box_max
    FeaturesShapeType box_min(0);
    box_min[N] = offset;
    FeaturesShapeType box_max = append_to_shape<N>(image.shape(), offset + size);
    // create a view to this bounding box
    FeaturesViewType features_view = features.subarray(box_min, box_max);
    // branch between the different features
    const std::string& feature_name = feature_scales_[i].first;
    const DataType& scale = feature_scales_[i].second;
    
    if (!feature_name.compare("GaussianSmoothing")) {
      calculate_gaussian_smoothing(image, features_view, scale);
    } else if (!feature_name.compare("LaplacianOfGaussian")) {
      calculate_laplacian_of_gaussian(image, features_view, scale);
    } else if (!feature_name.compare("GaussianGradientMagnitude")) {
      calculate_gaussian_gradient_magnitude(image, features_view, scale);
    } else if (!feature_name.compare("DifferenceOfGaussians")) {
      calculate_difference_of_gaussians(image, features_view, scale);
    } else if (!feature_name.compare("StructureTensorEigenvalues")) {
      calculate_structure_tensor_eigenvalues(image, features_view, scale);
    } else if (!feature_name.compare("HessianOfGaussianEigenvalues")) {
      calculate_hessian_of_gaussian_eigenvalues(image, features_view, scale);
    } else {
      throw std::runtime_error("Invalid feature name used");
    }
  }
  return 0;
}

// explicit instantiation
template class FeatureCalculator<2>;
template class FeatureCalculator<3>;

////
//// struct Segmentation
////
template<int N>
void Segmentation<N>::initialize(const vigra::MultiArray<N, DataType>& image, size_t num_classes) {
  if (segmentation_image_.shape() != image.shape()) {
    segmentation_image_.reshape(image.shape());
    label_image_.reshape(image.shape());
  }
  // TODO ugly
  typename vigra::MultiArrayShape<N+1>::type prediction_map_shape =
    append_to_shape<N>(image.shape(), num_classes);
  prediction_map_.reshape(prediction_map_shape, 0.0);
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
int Segmentation<N>::read_hdf5(
  const std::string filename,
  const bool segmentation_only)
{
  // load data
  read_hdf5_array<N, LabelType>(
    filename,
    "/segmentation/segmentation",
    segmentation_image_);
  read_hdf5_array<N, LabelType>(
    filename,
    "/segmentation/labels",
    label_image_);
  if (!segmentation_only) {
    read_hdf5_array<N+1, DataType>(
      filename,
      "/segmentation/features",
      feature_image_);
    read_hdf5_array<N+1, DataType>(
      filename,
      "/segmentation/prediction_map",
      prediction_map_);
  }
  // read label count
  LabelType min, max;
  label_image_.minmax(&min, &max);
  label_count_ = max;
  return 0;
}

// explicit instantiation
template class Segmentation<2>;
template class Segmentation<3>;

////
//// class SegmentationCalculator
////
template<int N>
SegmentationCalculator<N>::SegmentationCalculator(
    boost::shared_ptr<FeatureCalculator<N> > feature_calculator_ptr,
    const std::vector<RandomForestType>& random_forests,
    const TrackingOptions& options) :
  feature_calculator_ptr_(feature_calculator_ptr),
  random_forests_(random_forests),
  options_(options)
{
  // assertions?
}

template<int N>
int SegmentationCalculator<N>::calculate(
  const vigra::MultiArray<N, DataType>& image,
  Segmentation<N>& segmentation) const
{
  int return_status = 0;
  int num_pixel_classification_labels = options_.get_option<int>("NumPCLabels");
  int channel_index = options_.get_option<int>("Channel");
  DataType prob_threshold = options_.get_option<DataType>("SingleThreshold");
  // initialize the segmentation
  segmentation.initialize(image, num_pixel_classification_labels);
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
    vigra::Shape2(pixel_count, num_pixel_classification_labels),
    segmentation.prediction_map_.data());
  // loop over all random forests for prediction probabilities
  std::cout << "\tPixel Classification" << std::endl;
  #pragma omp parallel for
  for(size_t rf = 0; rf < random_forests_.size(); rf++)
  {
    vigra::MultiArray<2, DataType> prediction_temp(pixel_count, num_pixel_classification_labels);
    random_forests_[rf].predictProbabilities(
      feature_view,
      prediction_temp);

    #pragma omp critical
    {
      prediction_map_view += prediction_temp;
    }
  }

  // smooth prediction map
  if (options_.has_option<DataType>("PredictionMapSmoothing")) {
    vigra::ConvolutionOptions<N> conv_options;
    conv_options.filterWindowSize(2.0);
    vigra::MultiArrayView<N, DataType> prediction_map_channel_view =
      segmentation.prediction_map_.template bind<N>(channel_index);
      
    vigra::gaussianSmoothMultiArray(
      prediction_map_channel_view,
      prediction_map_channel_view,
      options_.get_option<DataType>("PredictionMapSmoothing"),
      conv_options);
  }

  // assign the labels
  std::cout << "\tThresholding" << std::endl;
  prob_threshold = prob_threshold * random_forests_.size();
  typename vigra::MultiArrayView<N, LabelType>::iterator seg_it;
  seg_it = segmentation.segmentation_image_.begin();
  for (size_t n = 0; n < pixel_count; n++, seg_it++) {
    if (prediction_map_view(n, channel_index) > prob_threshold) {
      *seg_it = 1;
    } else {
      *seg_it = 0;
    }
  }
  std::cout << "\tConnected Components" << std::endl;
  // extract objects
  segmentation.label_count_ = vigra::labelMultiArrayWithBackground(
    segmentation.segmentation_image_,
    segmentation.label_image_,
    vigra::IndirectNeighborhood,
    static_cast<LabelType>(0));
  return return_status;
}

// explicit instantiation
template class SegmentationCalculator<2>;
template class SegmentationCalculator<3>;

} // namespace isbi_pipeline
