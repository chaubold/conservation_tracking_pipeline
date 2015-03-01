#ifndef ISBI_LINEAGE_HXX
#define ISBI_LINEAGE_HXX

// stl
#include <vector> /* for std::vector */
#include <map> /* for std::map */
#include <utility> /* for std::pair */
#include <ostream> /* for overloading << */

// vigra
#include <vigra/multi_array.hxx> /* for MultiArray */

// pgmlink
#include <pgmlink/event.h> /* for pgmlink::Event */

// own
#include "common.h"

namespace isbi_pipeline {

class Lineage {
 public:
  Lineage(const EventVectorVectorType& events, size_t timeframe_offset = 0);
  template<int N>
  void relabel(
    vigra::MultiArrayView<N, LabelType>& label_image,
    const int timestep,
    const CoordinateMapPtrType coordinate_map_ptr
      = CoordinateMapPtrType()) const;
  template<int N>
  void restrict_to_bounding_box(
    const vigra::TinyVector<LabelType, N>& coord_min,
    const vigra::TinyVector<LabelType, N>& coord_max,
    const TraxelStoreType& traxelstore);
  void restrict_to_traxel_descendants(const TraxelVectorType traxels);
 private:
  void handle_event(const pgmlink::Event& event, const int timestep);
  void handle_appearance(const pgmlink::Event& event, const int timestep);
  void handle_disappearance(const pgmlink::Event& event, const int timestep);
  void handle_move(const pgmlink::Event& event, const int timestep);
  void handle_division(const pgmlink::Event& event, const int timestep);
  void handle_resolvedto(const pgmlink::Event& event, const int timestep);
  void start_track(
    const TraxelIndexType& traxel_index,
    const LabelType parent_track_index = 0);
  std::vector<LabelType> find_children_tracks(
    const LabelType track) const;
  // call clean_up() after remove(), otherwise there are inconsistencies within
  // a Lineage object
  void remove(const TraxelIndexType& traxel_index);
  void clean_up();

  LabelType track_count_;
  size_t timeframe_offset_;
  static const LabelType track_index_offset_ = 1;
  TraxelTrackIndexMapType traxel_track_map_;
  TrackTraxelIndexMapType track_traxel_map_;
  std::map<LabelType, LabelType> track_track_parent_map_;
  std::map<TraxelIndexType, std::vector<TraxelIndexType> > resolved_map_;
  friend std::ostream& operator<<(std::ostream& s, const Lineage& lineage);
};

// implementation
template<int N>
void Lineage::relabel(
  vigra::MultiArrayView<N, LabelType>& label_image,
  const int timestep,
  const CoordinateMapPtrType coordinate_map_ptr) const
{
  if (coordinate_map_ptr) {
    for(auto it = resolved_map_.begin(); it != resolved_map_.end(); it++) {
      const TraxelIndexType& traxel_index = it->first;
      if (traxel_index.first == timestep) {
        for (TraxelIndexType new_index : it->second) {
          pgmlink::update_labelimage<N, LabelType>(
            coordinate_map_ptr,
            label_image,
            new_index.first,
            new_index.second);
        }
      }
    }
  }
  TraxelTrackIndexMapType::const_iterator traxel_track_it;
  typedef typename vigra::MultiArrayView<N, LabelType>::iterator ImageItType;
  for (ImageItType it = label_image.begin(); it != label_image.end(); it++) {
    TraxelIndexType traxel_index(timestep, *it);
    traxel_track_it = traxel_track_map_.find(traxel_index);
    if (traxel_track_it != traxel_track_map_.end()){
      *it = traxel_track_it->second;
    } else {
      *it = 0;
    }
  }
}

template<int N>
void Lineage::restrict_to_bounding_box(
  const vigra::TinyVector<LabelType, N>& coord_min,
  const vigra::TinyVector<LabelType, N>& coord_max,
  const TraxelStoreType& traxelstore)
{
  for(
    TraxelStoreType::iterator t_it = traxelstore.begin();
    t_it != traxelstore.end();
    t_it++)
  {
    const FeatureMapType& feature_map = t_it->features;
    FeatureMapType::const_iterator f_min_it = feature_map.find("CoordMin");
    FeatureMapType::const_iterator f_max_it = feature_map.find("CoordMax");
    if ((f_min_it == feature_map.end()) or (f_max_it == feature_map.end())) {
      throw std::runtime_error(
        "cannot find \"CoordMin\" and \"CoordMax\" in feature map");
    }
    const FeatureArrayType& coord_min_trax = f_min_it->second;
    const FeatureArrayType& coord_max_trax = f_max_it->second;
    bool in_bb = true;
    for(size_t n = 0; n < N; n++) {
      in_bb = in_bb and (coord_max_trax[n] >= coord_min[n]);
      in_bb = in_bb and (coord_min_trax[n] <= coord_max[n]);
    }
    if(!in_bb) {
      // remove traxel and all dependent variables from lineage class
      TraxelIndexType traxel_index(t_it->Timestep, t_it->Id);
      remove(traxel_index);
    }
  }
  clean_up();
}

} // namespace isbi_pipeline

#endif //ISBI_LINEAGE_HXX
