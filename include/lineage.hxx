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
    vigra::MultiArrayView<N, LabelType>& segmentation_image,
    vigra::MultiArrayView<N, vigra::UInt16>& label_image,
    const int timestep,
    const CoordinateMapPtrType coordinate_map_ptr
      = CoordinateMapPtrType()) const;
  template<int N>
  void dilate(
    vigra::MultiArrayView<N, LabelType>& label_image,
    const int timestep,
    const TraxelStoreType& traxelstore,
    int radius) const;
  template<int N>
  void restrict_to_bounding_box(
    const vigra::TinyVector<LabelType, N>& coord_min,
    const vigra::TinyVector<LabelType, N>& coord_max,
    const TraxelStoreType& traxelstore);
  void restrict_to_traxel_descendants(const TraxelVectorType traxels);
 private:
  template<int N>
  void dilate_for_traxel(
    vigra::MultiArrayView<N, LabelType>& label_image,
    const pgmlink::Traxel& traxel,
    int radius) const;
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
  void new_splitted_track(
    TraxelIndexVectorType::iterator it_new_begin,
    TraxelIndexVectorType::iterator it_new_end,
    LabelType old_track_id);
  void check_track(
    LabelType track_id,
    TraxelIndexVectorType& traxels);
  void clean_up();

  LabelType track_count_;
  size_t timeframe_offset_;
  static const LabelType track_index_offset_ = 1;
  TraxelTrackIndexMapType traxel_track_map_;
  TrackTraxelIndexMapType track_traxel_map_;
  std::map<LabelType, LabelType> track_track_parent_map_;
  std::map<TraxelIndexType, std::vector<TraxelIndexType> > resolved_map_;
  // min_resolved_id_map_ stores the smallest "merger-resolved-to"-id for each
  // timestep
  std::map<LabelType, LabelType> min_resolved_id_map_;
  friend std::ostream& operator<<(std::ostream& s, const Lineage& lineage);
};

// implementation
template<int N>
void Lineage::relabel(
  vigra::MultiArrayView<N, LabelType>& segmentation_image,
  vigra::MultiArrayView<N, vigra::UInt16>& label_image,
  const int timestep,
  const CoordinateMapPtrType coordinate_map_ptr) const
{
  typedef typename vigra::MultiArrayView<N, LabelType>::iterator SegItType;
  typedef typename vigra::MultiArrayView<N, vigra::UInt16>::iterator LabelItType;
  assert(label_image.shape() == segmentation_image.shape());
  // remove all objects that were to small for the size filter and whose index
  // could be given to a new resolved traxel by pgmlink
  std::map<LabelType, LabelType>::const_iterator min_it =
    min_resolved_id_map_.find(timestep - timeframe_offset_);
  if (min_it == min_resolved_id_map_.end()) {
    throw std::runtime_error("timestep not in Lineage::min_resolved_id_map_");
  }
  for(
    SegItType s_it = segmentation_image.begin();
    s_it != segmentation_image.end();
    s_it++)
  {
    if (*s_it >= (min_it->second)) {
      *s_it = 0;
    }
  }
  if (coordinate_map_ptr) {
    for(auto it = resolved_map_.begin(); it != resolved_map_.end(); it++) {
      const TraxelIndexType& traxel_index = it->first;
      if (traxel_index.first + timeframe_offset_ == timestep) {
        for (TraxelIndexType new_index : it->second) {
          pgmlink::update_labelimage<N, LabelType>(
            coordinate_map_ptr,
            segmentation_image,
            new_index.first + timeframe_offset_,
            new_index.second);
        }
      }
    }
  }
  // now actually relabel the image with the track ids
  TraxelTrackIndexMapType::const_iterator traxel_track_it;
  LabelItType l_it = label_image.begin();
  for (SegItType s_it = segmentation_image.begin();
       s_it != segmentation_image.end();
       s_it++, l_it++) {
    if (*s_it == 0)
      continue;

    TraxelIndexType traxel_index(timestep - timeframe_offset_, *s_it);
    traxel_track_it = traxel_track_map_.find(traxel_index);
    if (traxel_track_it != traxel_track_map_.end()){
      *l_it = traxel_track_it->second;
    } else {
//      std::cout << "WARNING: could not find frame " << timestep << " traxel "
//                << *it << ". Tracking might be inconsistent!" << std::endl;
      *l_it = 0;
    }
  }
}

template<int N>
void Lineage::restrict_to_bounding_box(
  const vigra::TinyVector<LabelType, N>& bb_min,
  const vigra::TinyVector<LabelType, N>& bb_max,
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
      in_bb = in_bb and (coord_max_trax[n] >= bb_min[n]);
      in_bb = in_bb and (coord_min_trax[n] <= bb_max[n]);
    }
    if(!in_bb) {
      // remove traxel and all dependent variables from lineage class
      TraxelIndexType traxel_index(t_it->Timestep, t_it->Id);
      remove(traxel_index);
    }
  }
  clean_up();
}

template<int N>
void Lineage::dilate(
  vigra::MultiArrayView<N, LabelType>& label_image,
  const int timestep,
  const TraxelStoreType& ts,
  int radius) const
{
  const pgmlink::TraxelStoreByTimestep& traxels_by_timestep =
    ts.get<pgmlink::by_timestep>();
  std::pair<
    pgmlink::TraxelStoreByTimestep::const_iterator,
    pgmlink::TraxelStoreByTimestep::const_iterator> traxels_at =
      traxels_by_timestep.equal_range(timestep);
  for (auto it = traxels_at.first; it != traxels_at.second; it++) {
    dilate_for_traxel<N>(label_image, *it, radius);
  }
}

} // namespace isbi_pipeline

#endif //ISBI_LINEAGE_HXX
