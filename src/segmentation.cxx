// vigra
#include <vigra/labelimage.hxx> /* for labelImageWithBackground */
#include <vigra/multi_tensorutilities.hxx>
#include <vigra/hdf5impex.hxx> /* for writeHDF5 */

#include "segmentation.hxx"

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
}

template<int N>
int Segmentation<N>::export_hdf5(const std::string filename) {
  vigra::writeHDF5(filename.c_str(), "/segmentation/segmentation", segmentation_image_);
  vigra::writeHDF5(filename.c_str(), "/segmentation/labels", label_image_);
  vigra::writeHDF5(filename.c_str(), "/segmentation/features", feature_image_);
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
  vigra::MultiArray<2, DataType> reshaped_features;
  feature_calculator_ptr_->calculate(image, segmentation.feature_image_);
  reshape_features(segmentation.feature_image_, reshaped_features);
  // initialize arrays for segmentation
  vigra::MultiArray<2, DataType> label_probabilities(
    vigra::Shape2(reshaped_features.shape(0), 2),
    0.0);
  vigra::MultiArray<2, DataType> label_probabilities_temp(
    vigra::Shape2(reshaped_features.shape(0), 2),
    0.0);
  // loop over all random forests for prediction probabilities
  for (
    std::vector<RandomForestType>::const_iterator it = random_forests_.begin();
    it != random_forests_.end();
    it++
  ) {
    it->predictProbabilities(reshaped_features, label_probabilities_temp);
    label_probabilities += label_probabilities_temp;
  }
  // assign the labels
  typename vigra::MultiArray<N, unsigned>::iterator label_it;
  label_it = segmentation.label_image_.begin();
  for (size_t n = 0; n < reshaped_features.shape(0); n++) {
    if (label_probabilities(n, 1) > label_probabilities(n, 0)) {
      *label_it = 1;
    } else {
      *label_it = 0;
    }
  }
  // extract objects
  segmentation.label_count_ = vigra::labelImageWithBackground(
    vigra::srcImageRange(segmentation.label_image_),
    vigra::destImage(segmentation.segmentation_image_),
    1,
    0);
  // TODO:
  // if (options.count("border") > 0) {
  //   ignore_border_cc<2>(label_image, options["border"]);
  // }
  // done
  return return_status;
}

// TODO: superugly
template<>
int SegmentationCalculator<2>::reshape_features(
  const vigra::MultiArray<3, DataType>& features,
  vigra::MultiArray<2, DataType>& features_reshaped) const
{
  size_t size_0 = features.size(0);
  size_t size_1 = features.size(1);
  size_t size_2 = features.size(2);
  size_t n = 0;
  features_reshaped.reshape(vigra::Shape2(size_0 * size_1, size_2));
  for (size_t index_0 = 0; index_0 < size_0; index_0++) {
    for (size_t index_1 = 0; index_1 < size_1; index_1++) {
      for (size_t index_2 = 0; index_2 < size_2; index_2++) {
        features_reshaped(n, index_2) = features(index_0, index_1, index_2);
      }
      n++;
    }
  }
  return 0;
}

template<>
int SegmentationCalculator<3>::reshape_features(
  const vigra::MultiArray<4, DataType>& features,
  vigra::MultiArray<2, DataType>& features_reshaped) const
{
  size_t size_0 = features.size(0);
  size_t size_1 = features.size(1);
  size_t size_2 = features.size(2);
  size_t size_3 = features.size(3);
  size_t n = 0;
  features_reshaped.reshape(vigra::Shape2(size_0 * size_1 * size_2, size_3));
  for (size_t index_0 = 0; index_0 < size_0; index_0++) {
    for (size_t index_1 = 0; index_1 < size_1; index_1++) {
      for (size_t index_2 = 0; index_2 < size_2; index_2++) {
        for (size_t index_3 = 0; index_3 < size_3; index_3++) {
          features_reshaped(n, index_3) = features(
            index_0,
            index_1,
            index_2,
            index_3);
        }
        n++;
      }
    }
  }
  return 0;
}

// explicit instantiation
// TODO for dim = 3 as well
template class SegmentationCalculator<2>;

} // namespace isbi_pipeline
