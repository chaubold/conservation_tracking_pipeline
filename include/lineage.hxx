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
  Lineage(const EventVectorVectorType& events);
  TraxelIndexVectorType get_traxel_ids(const LabelType track_index) const;
  LabelType get_track_id(const TraxelIndexType traxel_index) const;
  template<int N>
  void relabel(
    vigra::MultiArrayView<N, LabelType>& label_image,
    const int timestep) const;
 private:
  void handle_event(const pgmlink::Event& event, const int timestep);
  void handle_appearance(const pgmlink::Event& event, const int timestep);
  void handle_move(const pgmlink::Event& event, const int timestep);
  void handle_division(const pgmlink::Event& event, const int timestep);
  void start_track(
    const TraxelIndexType& traxel_index,
    const LabelType parent_track_index = 0);

  LabelType track_count_;
  static const LabelType track_index_offset_ = 1;
  TraxelTrackIndexMapType traxel_track_map_;
  TrackTraxelIndexMapType track_traxel_map_;
  std::map<LabelType, LabelType> track_track_parent_map_;
  friend std::ostream& operator<<(std::ostream& s, const Lineage& lineage);
};

// implementation
template<int N>
void Lineage::relabel(
  vigra::MultiArrayView<N, LabelType>& label_image,
  const int timestep) const
{
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


} // namespace isbi_pipeline

#endif //ISBI_LINEAGE_HXX
