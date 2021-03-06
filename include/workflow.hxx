/* TODO
 * throw correct argument error
 * rename read_region_features_from_file
 * write images and intermediate results to prereserved memory
 */

#ifndef ISBI_WORKFLOW_HXX
#define ISBI_WORKFLOW_HXX
// stl
#include <stdexcept>
#include <set>
#include <algorithm>

// boost
#include <boost/filesystem.hpp>
#include <boost/shared_ptr.hpp>

// vigra
#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>
#include <vigra/multi_math.hxx>

// pgmlink
#include <pgmlink/tracking.h>

// own
#include "common.h"
#include "pipeline_helpers.hxx"
#include "segmentation.hxx"
#include "traxel_extractor.hxx"
#include "lineage.hxx"
#include "division_feature_extractor.hxx"

namespace isbi_pipeline {

namespace fs = boost::filesystem;

template<int N>
void load_multi_array(
  vigra::MultiArray<N, LabelType>& multi_array,
  const PathType& path);

template<int N>
void load_multi_array(
  vigra::MultiArray<N, DataType>& multi_array,
  const PathType& path);

template<int N>
void save_multi_array(
  vigra::MultiArray<N, LabelType>& multi_array,
  const PathType& path);

template<int N>
void save_multi_array(
  vigra::MultiArray<N, DataType>& multi_array,
  const PathType& path);

#ifdef USE_32_BIT_LABELS
template<int N>
void save_multi_array(
  vigra::MultiArray<N, vigra::UInt16>& multi_array,
  const PathType& path);
#endif

class Workflow {
 public:
  Workflow(bool calculate_segmentation, bool segmentation_dump = false);
  void init(int argc, char* argv[]);
  template<int N> Lineage run();
  template<int N> void extract_masked_traxels(
      const vigra::MultiArray<N, LabelType> &segmentation,
      const TraxelVectorType& traxels);
  void dump_traxelstore(TraxelStoreType& ts);
private:
  bool calculate_segmentation_;
  bool segmentation_dump_;
  int num_args_;
  TrackingOptions options_;
  // classifier config variables
  StringDataPairVectorType pix_feature_list_;
  std::vector<std::string> cnt_feature_list_;
  std::vector<std::string> div_feature_list_;
  RandomForestVectorType pix_feature_rfs_;
  RandomForestVectorType cnt_feature_rfs_;
  RandomForestVectorType div_feature_rfs_;
  // filename variables
  std::vector<PathType> raw_path_vec_;
  std::vector<PathType> seg_path_vec_;
  std::vector<PathType> res_path_vec_;
  PathType res_path_; // for lineage
  PathType traxelstore_dump_path_;
  // first frame masking:
  bool has_mask_image_;
  PathType mask_image_file_;
  std::vector<pgmlink::Traxel> traxels_to_keep_;
};

/*=============================================================================
  Implementation
=============================================================================*/
template<int N>
Lineage Workflow::run() {
  if (raw_path_vec_.size() == 0) {
    throw std::runtime_error("no images selected");
  }
  /*=========================
    Initialization
  =========================*/
  // initialize segmentation calculator if necessary
  boost::shared_ptr<SegmentationCalculator<N> > segmentation_calc_ptr;
  if (calculate_segmentation_) {
    // get the image scales
    vigra::TinyVector<DataType, N> image_scales
      = options_.get_vector_option<DataType, N>("scales");
    // create the feature calculator
    boost::shared_ptr<FeatureCalculator<N> > feature_calc_ptr(
      new FeatureCalculator<N>(pix_feature_list_, image_scales));
    // assign an instance to the segmentation calculator pointer
    segmentation_calc_ptr = boost::make_shared<SegmentationCalculator<N> >(
      feature_calc_ptr, pix_feature_rfs_, options_);
  }
  // initialize the traxel extractor
  TraxelExtractor<N> traxel_extractor(
    cnt_feature_list_,
    cnt_feature_rfs_,
    options_);
  // initialize the division feature extractor
  double template_size = options_.get_option<double>("templateSize");
  DivisionFeatureExtractor<N, LabelType> div_feature_extractor(template_size);
  // initialize the coordinate map
  CoordinateMapPtrType coordinate_map_ptr(new CoordinateMapType);
  // initialize the traxelstore
  TraxelStoreType ts;
  /*=========================
    Loop over timesteps
  =========================*/
  // for segmentation and traxel generation
  TraxelVectorType traxels_temp[2]; // temporary storage for traxels
  size_t curr_frame_index = 1;
  size_t prev_frame_index = 0;
  std::vector<PathType>::const_iterator raw_path_it = raw_path_vec_.begin();
  std::vector<PathType>::const_iterator seg_path_it = seg_path_vec_.begin();

  // memory for the segmentation
  Segmentation<N> segmentation;

  for(
    size_t timestep = 0;
    raw_path_it != raw_path_vec_.end();
    raw_path_it++, seg_path_it++, timestep++)
  {
    if (timestep >= 1 + options_.get_option<size_t>("time_range_1"))
      break;

    std::cout << "processing " << raw_path_it->string() << std::endl;
    // create references to the traxels of the previous and the
    // current frame
    std::swap(curr_frame_index, prev_frame_index);
    TraxelVectorType& traxels_curr_frame = traxels_temp[curr_frame_index];
    TraxelVectorType& traxels_prev_frame = traxels_temp[prev_frame_index];
    // load the raw image
    vigra::MultiArray<N, DataType> raw_image;
    load_multi_array<N>(raw_image, *raw_path_it);
    
    if (calculate_segmentation_) {
      // calculate the segmentation
      std::cout << "calculate segmentation" << std::endl;
      segmentation_calc_ptr->calculate(raw_image, segmentation);
      // save the segmentation
      save_multi_array<N>(segmentation.label_image_, *seg_path_it);
      // save the segmentation as a hdf5
      if (segmentation_dump_) {
        PathType h5_seg_path = fs::change_extension(*seg_path_it, ".h5");
        segmentation.export_hdf5(h5_seg_path.string());
      }
    } else {
      // load the segmentation from a file
      std::cout << "load labels from " << seg_path_it->string() << std::endl;
      load_multi_array<N>(segmentation.label_image_, *seg_path_it);
      LabelType min, max;
      segmentation.label_image_.minmax(&min, &max);
      segmentation.label_count_ = max;
    }
    // extract the traxel if this frame is within the range to be tracked:
    if (timestep < options_.get_option<size_t>("time_range_0")
        || timestep > options_.get_option<size_t>("time_range_1")) {
      // if we jumped over the last frame to track, then we still need to add its traxels to the TS
      if (timestep == 1 + options_.get_option<size_t>("time_range_1")) {
        for(pgmlink::Traxel& t : traxels_prev_frame) {
          pgmlink::add(ts, t);
        }
      }
      traxels_curr_frame.clear();
      continue;
    }

    std::cout << "extract traxel" << std::endl;
    traxel_extractor.extract(
      segmentation,
      raw_image,
      timestep,
      traxels_curr_frame);
    // get the coordinate map
    fill_coordinate_map(
      traxels_curr_frame,
      segmentation.label_image_,
      coordinate_map_ptr);
    // extract the division features
    std::cout << "extract division probabilities" << std::endl;
    // compute division features and add them to the traxelstore if
    // this is not the first frame
    if(raw_path_it != raw_path_vec_.begin()) {
      if (options_.get_option<bool>("withDivisions")) {
        div_feature_extractor.extract(
          traxels_prev_frame,
          traxels_curr_frame,
          segmentation.label_image_);
        div_feature_extractor.compute_div_prob(
          traxels_prev_frame,
          div_feature_list_,
          div_feature_rfs_);
      }
      // add the traxels of the previous frame to the traxelstore
      for(pgmlink::Traxel& t : traxels_prev_frame) {
        pgmlink::add(ts, t);
      }
    } else if(has_mask_image_) {
      // for the first frame, if a mask image was specified, get the set of marked traxels
      extract_masked_traxels<N>(segmentation.label_image_, traxels_curr_frame);
    }
  }
  // add the remaining traxels from the last frame
  for(pgmlink::Traxel& t : traxels_temp[curr_frame_index]) {
    pgmlink::add(ts, t);
  }

  // dump traxelstore
  dump_traxelstore(ts);

  /*=========================
    tracking
  =========================*/
  // EventVectorVectorType events = track(ts, options_, coordinate_map_ptr, traxels_to_keep_);
  EventVectorVectorType events = track(ts, options_, coordinate_map_ptr);
  Lineage lineage(events, options_.get_option<size_t>("time_range_0"));
  /*========================
    filter events
  ========================*/
  // filter to bounding box
  vigra::TinyVector<LabelType, N> bb_min, bb_max;
  get_bounding_box<N>(options_, bb_min, bb_max);
  lineage.restrict_to_bounding_box<N>(bb_min, bb_max, ts);
  // filter with traxels_to_keep_
  if (traxels_to_keep_.size() != 0) {
    lineage.restrict_to_traxel_descendants(traxels_to_keep_);
  }
  /*=========================
    relabeling
  =========================*/
  #pragma omp parallel for
  for (
    size_t timestep = options_.get_option<size_t>("time_range_0");
    timestep <= options_.get_option<size_t>("time_range_1");
    timestep++)
  {
    // do not attempt to save image if we were moving further than we have filenames
    if(timestep - options_.get_option<size_t>("time_range_0") >= res_path_vec_.size())
    {
      std::cout << "WARNING: tried to access invalid filename" << std::endl;
      continue;
    }
    std::vector<PathType>::const_iterator seg_path_it = seg_path_vec_.begin() + timestep;
    std::vector<PathType>::const_iterator res_path_it = res_path_vec_.begin() + timestep;

    // read the label image
    vigra::MultiArray<N, LabelType> segmentation_image;
    load_multi_array<N>(segmentation_image, *seg_path_it);

#ifdef USE_32_BIT_LABELS
    vigra::MultiArray<N, vigra::UInt16> label_image(segmentation_image.shape());
    // relabel the image
    lineage.relabel<N>(segmentation_image, label_image, timestep, coordinate_map_ptr);
    // save results
    std::cout << "save results to " << res_path_it->string() << std::endl;
    save_multi_array<N>(label_image, *res_path_it);
#else
    // relabel the image
    lineage.relabel<N>(segmentation_image, segmentation_image, timestep, coordinate_map_ptr);
    if(options_.has_option<int>("dilateResult") && options_.get_option<int>("dilateResult") > 0) {
      std::cout << "Dilate result" << std::endl;
      int radius = options_.get_option<int>("dilateResult");
      lineage.dilate<N>(segmentation_image, timestep, ts, radius);
    }
    // save results
    std::cout << "save results to " << res_path_it->string() << std::endl;
    save_multi_array<N>(segmentation_image, *res_path_it);
#endif
  }
  if(options_.has_option<std::string>("trackPositionExportLocation"))
  {
    std::cout << "saving track positions to " 
              << options_.get_option<std::string>("trackPositionExportLocation") 
              << std::endl;
    lineage.export_track_positions(options_.get_option<std::string>("trackPositionExportLocation"), ts);
  }

  // save lineage
  std::cout << "save lineage to " << res_path_.string() << std::endl;
  std::ofstream res_ofstream(res_path_.string());
  if (res_ofstream.is_open()) {
    res_ofstream << lineage;
    res_ofstream.close();
  } else {
    throw std::runtime_error("cannot open " + res_path_.string());
  }
  return lineage;
}

template<int N>
void Workflow::extract_masked_traxels(
    const vigra::MultiArray<N, LabelType>& segmentation,
    const TraxelVectorType& traxels
)
{
  using namespace vigra::multi_math;
  vigra::MultiArray<N, LabelType> mask_image;
  load_multi_array<N>(mask_image, mask_image_file_);
  // make mask image binary, background has label 0!
  mask_image = signi(mask_image);
  // extract part of segmentation that has been masked
  mask_image = segmentation * mask_image;
  // find which labels are present
  std::set<LabelType> masked_labels_in_first_frame =
      DivisionFeatureExtractor<N, LabelType>::find_unique_labels_in_roi(mask_image);
  // find corresponding traxels
  size_t missed_labels = 0;
  for (LabelType label : masked_labels_in_first_frame)
  {
    auto compare_predicate = [=](const pgmlink::Traxel& t) -> bool {
      return t.Timestep == 0 && t.Id == label;
    };
    TraxelVectorType::const_iterator traxel_it = std::find_if(traxels.begin(), traxels.end(), compare_predicate);
    if (traxel_it != traxels.end()) {
      traxels_to_keep_.push_back(*traxel_it);
    }
    else {
      std::cout << "Could not find traxel for selected label " << label 
                << ". Missed " << ++missed_labels << " / " << masked_labels_in_first_frame.size() << std::endl;
    }
  }

  std::cout << "Mask selected " << masked_labels_in_first_frame.size() << " traxels in first frame for tracking." << std::endl;
}

}

#endif // ISBI_WORKFLOW_HXX
