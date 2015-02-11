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
#include "segmentation.hxx"

namespace isbi_pipeline {

namespace acc = vigra::acc;

typedef pgmlink::feature_type FeatureType;

template<int N>
class TraxelExtractor {
 public:
  typedef typename vigra::CoupledIteratorType<N, unsigned, DataType>::type
    CoupledIteratorType;
  typedef typename vigra::acc::DynamicAccumulatorChainArray<
    vigra::CoupledArrays<N, unsigned, DataType>,
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
    unsigned int border_distance = 0,
    unsigned int lower_size_lim = 0,
    unsigned int upper_size_lim = 0);
  int extract(
    const Segmentation<N>& segmentation,
    const vigra::MultiArrayView<N, DataType>& image,
    const int timestep,
    const std::vector<std::string> feature_selection,
    pgmlink::TraxelStore& traxelstore) const;
 private:
  int select_features(
    const std::vector<std::string> feature_selection,
    AccChainType& acc_chain) const;
  int extract_for_label(
    const AccChainType& acc_chain,
    const size_t label_id,
    const int timestep,
    const std::vector<std::string> feature_selection,
    pgmlink::TraxelStore& traxelstore) const;
  int fill_feature_map(
    const AccChainType& acc_chain,
    const size_t label_id,
    const std::vector<std::string> feature_selection,
    pgmlink::FeatureMap& feature_map) const;
  unsigned int border_distance_;
  unsigned int lower_size_lim_;
  unsigned int upper_size_lim_;
};

} // end of namespace isbi_pipeline

#endif // ISBI_TRAXEL_EXTRACTOR
