#ifndef ISBI_SEGMENTATION_HXX
#define ISBI_SEGMENTATION_HXX

// stl
#include <map> /* for std::map */

// vigra
#include <vigra/multi_array.hxx> /* for MultiArray */
#include <vigra/multi_convolution.hxx>
#include <vigra/random_forest.hxx> /* for RandomForest */
#include <vigra/accumulator.hxx> /* for accumulator chain */

// boost
#include <boost/shared_ptr.hpp> /* for shared_ptr */

// pgmlink
#include <pgmlink/traxels.h> /* for Traxel and TraxelStore */

namespace isbi_pipeline {

typedef float DataType;
typedef std::vector<std::pair<std::string, double> > StringDoublePairVectorType;
typedef vigra::RandomForest<unsigned> RandomForestType;

template<int N>
class FeatureCalculator {
 public:
  FeatureCalculator(
    const StringDoublePairVectorType& feature_scales,
    double window_size = 3.5);
  size_t get_feature_size(const std::string& feature_name) const;
  size_t get_feature_size() const;
  int calculate(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArray<N+1, DataType>& features);
 private:
  int calculate_gaussian_smoothing(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    double feature_scale) const;
  int calculate_laplacian_of_gaussians(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    double feature_scale) const;
  int calculate_gaussian_gradient_magnitude(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    double feature_scale) const;
  int calculate_difference_of_gaussians(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    double feature_scale) const;
  int calculate_structure_tensor_eigenvalues(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    double feature_scale) const;
  int calculate_hessian_of_gaussian_eigenvalues(
    const vigra::MultiArrayView<N, DataType>& image,
    vigra::MultiArrayView<N+1, DataType>& features,
    double feature_scale) const;
  
  const StringDoublePairVectorType& feature_scales_;
  double window_size_;
  std::map<std::string, size_t> feature_sizes_;
  vigra::ConvolutionOptions<N> conv_options_;
};

template<int N>
struct Segmentation {
  vigra::MultiArray<N, unsigned> segmentation_image_;
  vigra::MultiArray<N, unsigned> label_image_;
  vigra::MultiArray<N+1, DataType> feature_image_;
  vigra::MultiArray<N+1, DataType> prediction_map_;
  size_t label_count_;
  
  void initialize(const vigra::MultiArray<N, DataType>& image);
  int export_hdf5(const std::string filename);
  int read_hdf5(const std::string filename);
};

template<int N>
class SegmentationCalculator {
 public:
  SegmentationCalculator(
    boost::shared_ptr<FeatureCalculator<N> > feature_calculator_ptr,
    const std::vector<RandomForestType>& random_forests);
  int calculate(
    const vigra::MultiArray<N, DataType>& image,
    Segmentation<N>& segmentation) const;
 private:
  boost::shared_ptr<FeatureCalculator<N> > feature_calculator_ptr_;
  const std::vector<RandomForestType> random_forests_;
};

template<int N>
class TraxelExtractor {
 public:
  typedef typename vigra::CoupledIteratorType<N, unsigned, unsigned>::type
    CoupledIteratorType;
  typedef typename vigra::acc::AccumulatorChainArray<
    vigra::CoupledArrays<N, unsigned, unsigned>,
    vigra::acc::Select<
      vigra::acc::DataArg<1>,
      vigra::acc::LabelArg<2>,
      vigra::acc::Count,
      vigra::acc::Coord<vigra::acc::Mean> >
  > AccChainType;
  TraxelExtractor(
    unsigned int border_distance = 0,
    unsigned int lower_size_lim = 0,
    unsigned int upper_size_lim = 0);
  int extract(
    const Segmentation<N>& segmentation,
    const int timestep,
    pgmlink::TraxelStore& traxelstore) const;
 private:
  int extract_for_label(
    const AccChainType& acc_chain,
    const size_t label_id,
    const int timestep,
    pgmlink::TraxelStore& traxelstore) const;
  unsigned int border_distance_;
  unsigned int lower_size_lim_;
  unsigned int upper_size_lim_;
};

} // end of namespace segmentation

#endif // ISBI_SEGMENTATION_HXX
