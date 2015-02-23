#ifndef ISBI_TRAXEL_EXTRACTOR_HXX
#define ISBI_TRAXEL_EXTRACTOR_HXX

// stl
#include <vector>
#include <string>

// vigra
#include <vigra/multi_array.hxx> /* for MultiArray */
#include <vigra/multi_iterator_coupled.hxx> /* for CoupledIterator */
#include <vigra/accumulator.hxx> /* for AccumulatorChainArray */

// pgmlink
#include <pgmlink/traxels.h> /* for Traxels and TraxelStore */
#include <pgmlink/merger_resolving.h> /* for extract_coordintes */

// own
#include "common.h"
#include "segmentation.hxx"
#include "pipeline_helpers.hxx" /* for options */

namespace isbi_pipeline {

template<unsigned int N>
void fill_coordinate_map(
  const TraxelVectorType& traxels,
  const vigra::MultiArray<N, LabelType>& label_image,
  CoordinateMapPtrType coordinate_map);

template<int N>
class TraxelExtractor {
 public:
  typedef typename vigra::CoupledIteratorType<N, LabelType, DataType>::type
    CoupledIteratorType;
  typedef typename acc::DynamicAccumulatorChainArray<
    vigra::CoupledArrays<N, LabelType, DataType>,
    acc::Select<
      acc::LabelArg<1>,
      acc::DataArg<2>,
      acc::Coord<acc::Maximum>,
      acc::Coord<acc::Minimum>,
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
    const std::vector<std::string> feature_selection,
    const RandomForestVectorType& random_forests,
    const TrackingOptions& options);
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
  const std::vector<std::string> feature_selection_;
  const RandomForestVectorType& random_forests_;
  const TrackingOptions& options_;
  unsigned int max_object_num_;
  unsigned int border_distance_;
  unsigned int lower_size_lim_;
  unsigned int upper_size_lim_;
  DataType x_scale_, y_scale_, z_scale_;
};

/*=============================================================================
  Implementation
=============================================================================*/

template<unsigned int N>
void fill_coordinate_map(
  const TraxelVectorType& traxels,
  const vigra::MultiArray<N, LabelType>& label_image,
  CoordinateMapPtrType coordinate_map_ptr)
{
  // iterate over all traxels
  for(const pgmlink::Traxel traxel : traxels) {
    const FeatureMapType& feature_map = traxel.features;
    // get the upper left and lower right of this traxel
    FeatureMapType::const_iterator c_min_it = feature_map.find("CoordMin");
    FeatureMapType::const_iterator c_max_it = feature_map.find("CoordMax");
    if (c_min_it == feature_map.end() or c_max_it == feature_map.end()) {
      throw std::runtime_error(
        "CoordMin and CoordMax not found in feature map for traxel");
    }
    // convert to tiny vectors of long int
    vigra::TinyVector<long int, int(N)> c_min_vec, c_max_vec;
    for(size_t n = 0; n < N; n++) {
      c_min_vec[n] = static_cast<long int>((c_min_it->second)[n]);
      c_max_vec[n] = static_cast<long int>((c_max_it->second)[n]) + 1;
    }
    // get a view of this traxel on the label_image
    vigra::MultiArrayView<N, LabelType> traxel_view = label_image.subarray(
      c_min_vec,
      c_max_vec);
    // extract the coordinates
    pgmlink::extract_coordinates<N, LabelType>(
      coordinate_map_ptr,
      traxel_view,
      c_min_vec,
      traxel);
    // debug
    TraxelIndexType t_index(traxel.Timestep, traxel.Id);
    CoordinateMapType::const_iterator cm_it = coordinate_map_ptr->find(t_index);
  }
}


} // end of namespace isbi_pipeline

#endif // ISBI_TRAXEL_EXTRACTOR
