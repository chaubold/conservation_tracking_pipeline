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

namespace isbi_pipeline {

typedef std::vector<pgmlink::Event> EventVectorType;
typedef std::vector<EventVectorType> EventVectorVectorType;
typedef std::pair<int, unsigned> TraxelIndexType;
typedef std::map<TraxelIndexType, unsigned> TraxelTrackIndexMapType;
typedef std::vector<TraxelIndexType> TraxelIndexVectorType;
typedef std::map<unsigned, TraxelIndexVectorType> TrackTraxelIndexMapType;

class Lineage {
 public:
  Lineage(const EventVectorVectorType& events);
  TraxelIndexVectorType get_traxel_ids(const unsigned track_index) const;
  unsigned get_track_id(const TraxelIndexType traxel_index) const;
  template<int N>
  void relabel(
    vigra::MultiArrayView<N, unsigned>& label_image,
    const int timestep) const;
 private:
  void handle_event(const pgmlink::Event& event, const int timestep);
  void handle_appearance(const pgmlink::Event& event, const int timestep);
  void handle_move(const pgmlink::Event& event, const int timestep);
  void handle_division(const pgmlink::Event& event, const int timestep);
  void start_track(
    const TraxelIndexType& traxel_index,
    const unsigned parent_track_index = 0);

  unsigned track_count_;
  static const unsigned track_index_offset_ = 1;
  TraxelTrackIndexMapType traxel_track_map_;
  TrackTraxelIndexMapType track_traxel_map_;
  std::map<unsigned, unsigned> track_track_parent_map_;
  friend std::ostream& operator<<(std::ostream& s, const Lineage& lineage);
};

// implementation
template<int N>
void Lineage::relabel(
  vigra::MultiArrayView<N, unsigned>& label_image,
  const int timestep) const
{
  TraxelTrackIndexMapType::const_iterator traxel_track_it;
  typedef typename vigra::MultiArrayView<N, unsigned>::iterator ImageItType;
  for (ImageItType it = label_image.begin(); it != label_image.end(); it++) {
    std::pair<int, unsigned> traxel_index(timestep, *it);
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
