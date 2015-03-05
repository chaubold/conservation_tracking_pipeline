#include <sstream> /* for std::stringstream */
#include <algorithm> /* for std::remove */

#include <vigra/flatmorphology.hxx> /* for DiskDilationWithMask */

#include "lineage.hxx"

namespace isbi_pipeline {

////
//// class Lineage
////
Lineage::Lineage(const EventVectorVectorType& events, size_t timeframe_offset):
    track_count_(0),
    timeframe_offset_(timeframe_offset)
{
  for (int timestep = 0; timestep < int(events.size()); timestep++) {
    // std::cout << "Timestep " << timestep << std::endl;
    for (
      EventVectorType::const_iterator event_it = events[timestep].begin();
      event_it != events[timestep].end();
      event_it++ )
    {
      // std::cout << "\t" << *event_it << std::endl;
      handle_event(*event_it, timestep);
    }
  }
}

// TODO not very efficient
std::vector<LabelType> Lineage::find_children_tracks(
  const LabelType track) const
{
  std::vector<LabelType> ret;
  for(std::pair<LabelType, LabelType> pair : track_track_parent_map_) {
    if (pair.second == track) {
      ret.push_back(pair.first);
    }
  }
  return ret;
}

void Lineage::restrict_to_traxel_descendants(const TraxelVectorType traxels) {
  // get all track ids that contain the traxels descendants
  std::set<LabelType> tracks_to_keep;
  for(pgmlink::Traxel traxel : traxels) {
    TraxelIndexType traxel_index(traxel.Timestep, traxel.Id);
    // get the track id that contains this traxel
    if (traxel_track_map_.count(traxel_index) != 0) {
      LabelType track = traxel_track_map_[traxel_index];
      // keep this track
      tracks_to_keep.insert(track);
      // keep all childrent tracks as well
      std::vector<LabelType> children_tracks = find_children_tracks(track);
      for(LabelType child_track : children_tracks) {
        tracks_to_keep.insert(child_track);
      }
    }
  }
  // remove all traxel from the traxel index map if the corresponding track is
  // not in the tracks_to_keep set
  {
    TraxelTrackIndexMapType::iterator tt_it = traxel_track_map_.begin();
    while (tt_it != traxel_track_map_.end()) {
      TraxelTrackIndexMapType::iterator tt_curr_it = tt_it;
      tt_it++;
      if (tracks_to_keep.count(tt_curr_it->second) == 0) {
        // TODO what about the resolved_map_?
        traxel_track_map_.erase(tt_curr_it->first);
      }
    }
  }
  // clean up (make lineage object consistent again)
  clean_up();
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
  case pgmlink::Event::Disappearance:
    handle_disappearance(event, timestep);
  case pgmlink::Event::ResolvedTo:
    handle_resolvedto(event, timestep);
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

void Lineage::handle_disappearance(
  const pgmlink::Event& event,
  const int timestep)
{
  TraxelIndexType traxel_index(timestep - 1, event.traxel_ids[0]);
  // check if the parent track exists
  TraxelTrackIndexMapType::const_iterator track_it = traxel_track_map_.find(
    traxel_index);
  // start a new track for the parent traxel if no track exists
  if (track_it == traxel_track_map_.end()) {
    // std::cout << "No parent track found for disappearance at timestep "
    //   << timestep << std::endl;
    start_track(traxel_index);
  }
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
    // std::cout << "No parent track found for move at timstep "
    //   << timestep << std::endl;
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
    // std::cout << "No parent track found for division at timstep "
    //   << timestep << std::endl;
    start_track(parent_index);
  }
  // start the new tracks and set the parent map correctly
  start_track(lchild_index, traxel_track_map_[parent_index]);
  start_track(rchild_index, traxel_track_map_[parent_index]);
}

void Lineage::handle_resolvedto(
  const pgmlink::Event& event, const int timestep)
{
  TraxelIndexType old_index(timestep, event.traxel_ids[0]);
  resolved_map_[old_index].clear();
  for (size_t n = 1; n < event.traxel_ids.size(); n++) {
    TraxelIndexType new_index(timestep, event.traxel_ids[n]);
    resolved_map_[old_index].push_back(new_index);
  }
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

void Lineage::remove(const TraxelIndexType& traxel_index) {
  // get the iterator to the resolved map
  typedef std::map<TraxelIndexType, std::vector<TraxelIndexType> >::iterator
    ResolvedMapItType;
  ResolvedMapItType rmap_it = resolved_map_.find(traxel_index);
  if (rmap_it != resolved_map_.end()) {
    // if the traxel is a merger remove all new traxel ids
    for(TraxelIndexType resolved_index : (rmap_it->second)) {
      remove(resolved_index);
    }
    resolved_map_.erase(rmap_it);
  } else {
    // remove a traxel
    TraxelTrackIndexMapType::iterator it = traxel_track_map_.find(traxel_index);
    if (it != traxel_track_map_.end()) {
      // remove the traxel from the traxel_track_map
      traxel_track_map_.erase(it);
    }
  }
}

void Lineage::clean_up() {
  // lambda function to check if a traxel is removed from the lineage class
  auto not_in_lineage = [&] (const TraxelIndexType traxel_index) {
    return (traxel_track_map_.count(traxel_index) == 0);
  };
  // loop over all tracks and remove traxels which are not in the
  // traxel_track_map_
  {
    TrackTraxelIndexMapType::iterator tt_it = track_traxel_map_.begin();
    while(tt_it != track_traxel_map_.end()) {
      TrackTraxelIndexMapType::iterator tt_curr_it = tt_it;
      tt_it++;
      // remove the traxels in the vector if they do not appear in the
      // traxel_track_map_
      TraxelIndexVectorType& trax_vec = tt_curr_it->second;
      trax_vec.erase(
        std::remove_if(trax_vec.begin(), trax_vec.end(), not_in_lineage),
        trax_vec.end());
      if (trax_vec.size() == 0) {
        track_traxel_map_.erase(tt_curr_it);
      }
    }
  }
  // remove all tracks from the parent map if they do not exists in the track
  // traxel map and set the references to zero if applicable
  {
    typedef std::map<LabelType, LabelType>::iterator TrackTrackMapIt;
    TrackTrackMapIt tt_it = track_track_parent_map_.begin();
    while(tt_it != track_track_parent_map_.end()) {
      TrackTrackMapIt tt_curr_it = tt_it;
      tt_it++;
      // remove links to invalid parents
      if (track_traxel_map_.count(tt_curr_it->second) == 0) {
        tt_curr_it->second = 0;
      }
      // remove the tracks
      if (track_traxel_map_.count(tt_curr_it->first) == 0) {
        track_track_parent_map_.erase(tt_curr_it);
      }
    }
  }
}

template<>
void Lineage::dilate_for_traxel<3>(
  vigra::MultiArrayView<3, LabelType>& label_image,
  const pgmlink::Traxel& traxel,
  int radius) const
{
  throw std::runtime_error("Lineage::dilate_for_traxel<3> not implemented");
}

template<>
void Lineage::dilate_for_traxel<2>(
    vigra::MultiArrayView<2, LabelType>& label_image,
    const pgmlink::Traxel& traxel,
    int radius) const
{
  TraxelIndexType traxel_index(traxel.Timestep - timeframe_offset_, traxel.Id);
  TraxelTrackIndexMapType::const_iterator track_id_it = traxel_track_map_.find(
    traxel_index);
  if (track_id_it != traxel_track_map_.end()) {
    LabelType track_id = track_id_it->second;
    vigra::TinyVector<LabelType, 2> coord_min;
    vigra::TinyVector<LabelType, 2> coord_max;
    const FeatureMapType& feature_map = traxel.features;
    FeatureMapType::const_iterator f_min_it = feature_map.find("CoordMin");
    FeatureMapType::const_iterator f_max_it = feature_map.find("CoordMax");
    if ((f_min_it == feature_map.end()) or (f_max_it == feature_map.end())) {
      throw std::runtime_error(
        "cannot find \"CoordMin\" and \"CoordMax\" in feature map");
    }
    const FeatureArrayType& coord_min_trax = f_min_it->second;
    const FeatureArrayType& coord_max_trax = f_max_it->second;
    int width = radius + 1;
    for(size_t n = 0; n < 2; n++) {
      coord_min[n] = std::max<int>(coord_min_trax[n] - width, 0);
      coord_max[n] = std::min<int>(coord_max_trax[n] + width, label_image.shape(n));
    }
    vigra::MultiArrayView<2, LabelType> subarray = label_image.subarray(
      coord_min,
      coord_max);
    // std::cout << "Subarray is" << std::endl << subarray;
    vigra::MultiArray<2, LabelType> mask(coord_max - coord_min);
    auto equals_id = [&] (const LabelType x) {return x == track_id;};
    vigra::transformImage(srcImageRange(subarray), destImage(mask), equals_id);
    // std::cout << "Mask is" << std::endl << mask;
    vigra::MultiArray<2, LabelType> mask_dilated(coord_max - coord_min);
    vigra::discDilation(mask, mask_dilated, radius);
    // std::cout << "Mask dilated is" << std::endl << mask_dilated;
    auto set_id = [&] (const LabelType x, const LabelType y) {
      if (x) {
        return static_cast<LabelType>(track_id);
      } else {
        return static_cast<LabelType>(y);
      }
    };
    vigra::combineTwoImages(mask_dilated, subarray, subarray, set_id);
    // std::cout << "New subarray is" << std::endl << subarray;
  }
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
    sstream << it->first << " " << e_timestep + lineage.timeframe_offset_
            << " " << l_timestep + lineage.timeframe_offset_
            << " " << parent_track_it->second << std::endl;
  }
  return s << sstream.str();
}

} // namespace isbi_pipeline

