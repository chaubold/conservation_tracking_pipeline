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
  vigra::MultiArrayView<N, DataType> results(features.template bind<N>(0));
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
  double feature_scale) const
{
  LOG("calculate gaussian gradient magnitude (" << feature_scale << ")");
  vigra::MultiArrayView<N, DataType> results(features.template bind<2>(0));
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
  vigra::MultiArrayView<N, DataType> results(features.template bind<N>(0));
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
  std::vector<size_t> offsets;
  size_t offset = 0;
  size_t size = 0;
  features.reshape(
    vigra::Shape3(image.shape(0), image.shape(1), get_feature_size()));
  
  // store all offsets
  for (
    StringDoublePairVectorType::const_iterator it = feature_scales_.begin();
    it != feature_scales_.end();
    it++
  ) 
  {
    offsets.push_back(offset);
    offset += get_feature_size(it->first);
  }

  // compute features in parallel
  #pragma omp parallel for
  for(size_t fs = 0; fs < feature_scales_.size(); fs++)
  {
    // get the size of the current feature and create a view of all
    // feature vectors
    size = get_feature_size(feature_scales_[fs].first);
    vigra::MultiArrayView<3, DataType> features_v;
    features_v = features.subarray(
      vigra::Shape3(0, 0, offsets[fs]),
      vigra::Shape3(image.shape(0), image.shape(1), offsets[fs] + size));
    if (!feature_scales_[fs].first.compare("GaussianSmoothing")) {
      calculate_gaussian_smoothing(image, features_v, feature_scales_[fs].second);
    } else if (!feature_scales_[fs].first.compare("LaplacianOfGaussian")) {
      calculate_laplacian_of_gaussians(image, features_v, feature_scales_[fs].second);
    } else if (!feature_scales_[fs].first.compare("GaussianGradientMagnitude")) {
      calculate_gaussian_gradient_magnitude(image, features_v, feature_scales_[fs].second);
    } else if (!feature_scales_[fs].first.compare("DifferenceOfGaussians")) {
      calculate_difference_of_gaussians(image, features_v, feature_scales_[fs].second);
    } else if (!feature_scales_[fs].first.compare("StructureTensorEigenvalues")) {
      calculate_structure_tensor_eigenvalues(image, features_v, feature_scales_[fs].second);
    } else if (!feature_scales_[fs].first.compare("HessianOfGaussianEigenvalues")) {
      calculate_hessian_of_gaussian_eigenvalues(image, features_v, feature_scales_[fs].second);
    } else {
      throw std::runtime_error("Invalid feature name used");
    }
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
int Segmentation<N>::read_hdf5(
  const std::string filename,
  const bool segmentation_only)
{
  // load data
  read_hdf5_array<N, unsigned>(
    filename,
    "/segmentation/segmentation",
    segmentation_image_);
  read_hdf5_array<N, unsigned>(
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
  // loop over all random forests for prediction probabilities
  LOG("Evaluate random forests");

  #pragma omp parallel for
  // for(
  //   std::vector<RandomForestType>::const_iterator it = random_forests_.begin();
  //   it != random_forests_.end();
  //   it++
  // ) {
  for(size_t rf = 0; rf < random_forests_.size(); rf++)
  {
    vigra::MultiArray<2, DataType> prediction_temp(pixel_count, 2);
    random_forests_[rf].predictProbabilities(
      feature_view,
      prediction_temp);

    #pragma omp critical
    {
      prediction_map_view += prediction_temp;
    }
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

} // namespace isbi_pipeline
