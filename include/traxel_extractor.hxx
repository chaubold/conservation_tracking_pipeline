#ifndef ISBI_TRAXEL_EXTRACTOR_HXX
#define ISBI_TRAXEL_EXTRACTOR_HXX

// stl
#include <vector>
#include <string>

// vigra
#include <vigra/multi_iterator_coupled.hxx> /* for CoupledIterator */
#include <vigra/accumulator.hxx> /* for AccumulatorChainArray */

// pgmlink
#include <pgmlink/traxels.h> /* for Traxels and TraxelStore */

// own
#include "common.h"
#include "segmentation.hxx"

namespace isbi_pipeline {

template<int N>
class TraxelExtractor {
 public:
  typedef typename vigra::CoupledIteratorType<N, LabelType, DataType>::type
    CoupledIteratorType;
  typedef typename vigra::acc::DynamicAccumulatorChainArray<
    vigra::CoupledArrays<N, LabelType, DataType>,
    acc::Select<
      acc::LabelArg<1>,
      acc::DataArg<2>,
      acc::Coord<acc::Principal<acc::Kurtosis> >,
      acc::Coord<acc::Principal<acc::Skewness> >,
      acc::Count,
      acc::Kurtosis,
      acc::Maximum,
      acc::Mean,
      acc::Minimum,
      acc::RegionCenter,
      acc::RegionRadii,
      acc::Skewness,
      acc::Sum,
      acc::Variance >
  > AccChainType;
  TraxelExtractor(
    unsigned int max_object_num,
    const std::vector<std::string> feature_selection,
    const RandomForestVectorType& random_forests,
    unsigned int border_distance = 0,
    unsigned int lower_size_lim = 0,
    unsigned int upper_size_lim = 0);
  int extract(
    const Segmentation<N>& segmentation,
    const vigra::MultiArrayView<N, DataType>& image,
    const int timestep,
    TraxelVectorType& traxels) const;
 private:
  int select_features(AccChainType& acc_chain) const;
  int extract_for_label(
    const AccChainType& acc_chain,
    const size_t label_id,
    const int timestep,
    TraxelVectorType& traxels) const;
  int fill_feature_map(
    const AccChainType& acc_chain,
    const size_t label_id,
    FeatureMapType& feature_map) const;
  int get_detection_probability(FeatureMapType& feature_map) const;
  unsigned int max_object_num_;
  const std::vector<std::string> feature_selection_;
  const RandomForestVectorType& random_forests_;
  unsigned int border_distance_;
  unsigned int lower_size_lim_;
  unsigned int upper_size_lim_;
};

} // end of namespace isbi_pipeline

#endif // ISBI_TRAXEL_EXTRACTOR
