#include <sstream> /* for std::stringstream */

#include "lineage.hxx"

namespace isbi_pipeline {

////
//// class Lineage
////
Lineage::Lineage(const EventVectorVectorType& events) {
  track_count_ = 0;
  for (int timestep = 0; timestep < int(events.size()); timestep++) {
    for (
      EventVectorType::const_iterator event_it = events[timestep].begin();
      event_it != events[timestep].end();
      event_it++ )
    {
      handle_event(*event_it, timestep);
    }
  }
}

void Lineage::handle_event(const pgmlink::Event& event, const int timestep) {
  switch(event.type) {
  case pgmlink::Event::Move:
    handle_move(event, timestep);
    break;
  case pgmlink::Event::Division:
    handle_division(event, timestep);
    break;
  case pgmlink::Event::Appearance:
    handle_appearance(event, timestep);
    break;
  default:
    break;
  }
}

void Lineage::handle_appearance(
  const pgmlink::Event& event,
  const int timestep)
{
  TraxelIndexType traxel_index(timestep, event.traxel_ids[0]);
  // start a new track for this traxel index
  start_track(traxel_index);
}

void Lineage::handle_move(const pgmlink::Event& event, const int timestep) {
  // create the traxel index for the child and parent traxel
  TraxelIndexType parent_index(timestep - 1, event.traxel_ids[0]);
  TraxelIndexType child_index(timestep, event.traxel_ids[1]);
  // check if there already exists a track
  TraxelTrackIndexMapType::const_iterator track_it = traxel_track_map_.find(
    parent_index);
  // start a new track if no parent exists
  if (track_it == traxel_track_map_.end()) {
    std::cout << "No parent track found for move at timstep "
      << timestep << std::endl;
    start_track(parent_index);
  }
  // handle the move
  LabelType track_index = traxel_track_map_[parent_index];
  track_traxel_map_[track_index].push_back(child_index);
  traxel_track_map_[child_index] = track_index;
}

void Lineage::handle_division(const pgmlink::Event& event, const int timestep) {
  TraxelIndexType parent_index(timestep - 1, event.traxel_ids[0]);
  TraxelIndexType lchild_index(timestep, event.traxel_ids[1]);
  TraxelIndexType rchild_index(timestep, event.traxel_ids[2]);
  // check if there already exists a parent track
  TraxelTrackIndexMapType::const_iterator track_it = traxel_track_map_.find(
    parent_index);
  // start a new track if no parent exists
  if (track_it == traxel_track_map_.end()) {
    std::cout << "No parent track found for division at timstep "
      << timestep << std::endl;
    start_track(parent_index);
  }
  // start the new tracks and set the parent map correctly
  start_track(lchild_index, traxel_track_map_[parent_index]);
  start_track(rchild_index, traxel_track_map_[parent_index]);
}

void Lineage::start_track(
  const TraxelIndexType& traxel_index,
  const LabelType parent_track_index)
{
  // get the new unique index of this track
  LabelType track_index = track_count_ + track_index_offset_;
  track_count_++;
  // map the traxel index to this track index
  traxel_track_map_[traxel_index] = track_index;
  // set the parent to zero for this track
  track_track_parent_map_[track_index] = parent_track_index;
  // create a new vector of traxel indexes
  track_traxel_map_[track_index].push_back(traxel_index);
}

std::ostream& operator<<(std::ostream& s, const Lineage& lineage) {
  std::stringstream sstream;
  const TrackTraxelIndexMapType& track_traxel_map = lineage.track_traxel_map_;
  for (
    TrackTraxelIndexMapType::const_iterator it = track_traxel_map.begin();
    it != track_traxel_map.end();
    it++ )
  {
    const TraxelIndexVectorType& traxel_index_vector = it->second;
    int e_timestep = traxel_index_vector.front().first;
    int l_timestep = traxel_index_vector.back().first;
    std::map<LabelType, LabelType>::const_iterator parent_track_it =
      lineage.track_track_parent_map_.find(it->first);
    if (parent_track_it == lineage.track_track_parent_map_.end()) {
      throw std::runtime_error("Parent track id does not exist");
    }
    sstream << it->first << " " << e_timestep << " " << l_timestep
      << " " << parent_track_it->second << std::endl;
  }
  return s << sstream.str();
}

} // namespace isbi_pipeline

