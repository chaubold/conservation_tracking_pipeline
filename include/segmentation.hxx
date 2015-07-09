#ifndef ISBI_SEGMENTATION_HXX
#define ISBI_SEGMENTATION_HXX

// stl
#include <map> /* for std::map */

// vigra
#include <vigra/multi_array.hxx> /* for MultiArray */
#include <vigra/multi_convolution.hxx>
#include <vigra/random_forest.hxx> /* for RandomForest */
#include <vigra/accumulator.hxx> /* for accumulator chain */
#include <vigra/hdf5impex.hxx> /* for hdf 5 import */

// boost
#include <boost/shared_ptr.hpp> /* for shared_ptr */

// own
#include "common.h"
#include "pipeline_helpers.hxx"

namespace isbi_pipeline {

namespace acc = vigra::acc;

template<int N, typename T>
int read_hdf5_array(
  const std::string filename,
  const std::string path_in_file,
  vigra::MultiArray<N, T>& multi_array);

template<int N>
class FeatureCalculator {
 public:
  FeatureCalculator(
    const StringDataPairVectorType& feature_scales,
    DataType window_size = 3.5);
  FeatureCalculator(
    const StringDataPairVectorType& feature_scales,
    const vigra::TinyVector<DataType, N> image_scales,
    DataType window_size = 3.5);
  size_t get_feature_size(const std::string& feature_name) const;
  size_t get_feature_size() const;
  int calculate(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArray<N+1, DataType>& features);
 private:
  int calculate_gaussian_smoothing(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    DataType feature_scale) const;
  int calculate_laplacian_of_gaussian(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    DataType feature_scale) const;
  int calculate_gaussian_gradient_magnitude(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    DataType feature_scale);
  int calculate_difference_of_gaussians(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    DataType feature_scale);
  int calculate_structure_tensor_eigenvalues(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    DataType feature_scale);
  int calculate_hessian_of_gaussian_eigenvalues(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    DataType feature_scale);
  
  // temporary storage for feature computation allocated only once
  vigra::MultiArray<N, DataType> feature_temp_;
  vigra::MultiArray<N, vigra::TinyVector<DataType, (N*(N+1))/2> > tensor_temp_;
  vigra::MultiArray<N, vigra::TinyVector<DataType, N> > eigenvalue_temp_;

  const StringDataPairVectorType& feature_scales_;
  DataType window_size_;
  std::map<std::string, size_t> feature_sizes_;
  vigra::ConvolutionOptions<N> conv_options_;
};

template<int N>
struct Segmentation {
  vigra::MultiArray<N, LabelType> segmentation_image_;
  vigra::MultiArray<N, LabelType> label_image_;
  vigra::MultiArray<N+1, DataType> feature_image_;
  vigra::MultiArray<N+1, DataType> prediction_map_;
  size_t label_count_;
  
  void initialize(const vigra::MultiArray<N, DataType>& image, size_t num_classes = 2);
  int export_hdf5(const std::string filename);
  int read_hdf5(
    const std::string filename,
    const bool segmentation_only = false);
};

template<int N>
class SegmentationCalculator {
 public:
  SegmentationCalculator(
    boost::shared_ptr<FeatureCalculator<N> > feature_calculator_ptr,
    const RandomForestVectorType& random_forests,
    const TrackingOptions& options);
  int calculate(
    const vigra::MultiArray<N, DataType>& image,
    Segmentation<N>& segmentation) const;
 private:
  boost::shared_ptr<FeatureCalculator<N> > feature_calculator_ptr_;
  const RandomForestVectorType random_forests_;
  const TrackingOptions& options_;
};

/*=============================================================================
  Implementation
=============================================================================*/

// TODO check if file and dataset exist
template<int N, typename T>
int read_hdf5_array(
  const std::string filename,
  const std::string path_in_file,
  vigra::MultiArray<N, T>& multi_array)
{
  vigra::HDF5ImportInfo import_info(
    filename.c_str(),
    path_in_file.c_str());
  if(import_info.numDimensions() != N) {
    throw std::runtime_error("Dataset has wrong dimension");
  }
  typedef typename vigra::MultiArrayShape<N>::type NShapeType;
  NShapeType shape(import_info.shape().begin());
  multi_array.reshape(shape);
  vigra::readHDF5(import_info, multi_array);
  return 0;
}

} // end of namespace segmentation

#endif // ISBI_SEGMENTATION_HXX
