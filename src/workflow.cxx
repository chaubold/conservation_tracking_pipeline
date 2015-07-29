#include "workflow.hxx"

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>

namespace fs = boost::filesystem;

namespace isbi_pipeline {

template<>
void load_multi_array<2>(
  vigra::MultiArray<2, DataType>& multi_array,
  const PathType& path)
{
  vigra::importImage(path.string().c_str(), multi_array);
}

template<>
void load_multi_array<3>(
  vigra::MultiArray<3, DataType>& multi_array,
  const PathType& path)
{
  vigra::ImageImportInfo info(path.string().c_str());
  vigra::Shape3 shape(info.shape()[0], info.shape()[1], info.numImages());
  multi_array.reshape(shape);
  for(int i = 0; i < info.numImages(); i++) {
    info.setImageIndex(i);
    vigra::importImage(info, multi_array.bindOuter(i));
  }
}

template<>
void load_multi_array<2>(
  vigra::MultiArray<2, LabelType>& multi_array,
  const PathType& path)
{
  vigra::importImage(path.string().c_str(), multi_array);
}

template<>
void load_multi_array<3>(
  vigra::MultiArray<3, LabelType>& multi_array,
  const PathType& path)
{
  vigra::ImageImportInfo info(path.string().c_str());
  vigra::Shape3 shape(info.shape()[0], info.shape()[1], info.numImages());
  multi_array.reshape(shape);
  for(int i = 0; i < info.numImages(); i++) {
    info.setImageIndex(i);
    vigra::importImage(info, multi_array.bindOuter(i));
  }
}

template<>
void save_multi_array<2>(
  vigra::MultiArray<2, DataType>& multi_array,
  const PathType& path)
{
  vigra::exportImage(multi_array, path.string());
}

template<>
void save_multi_array<3>(
  vigra::MultiArray<3, DataType>& multi_array,
  const PathType& path)
{
  vigra::VolumeExportInfo export_info(path.string().c_str());
  vigra::exportVolume(multi_array, export_info);
}

template<>
void save_multi_array<2>(
  vigra::MultiArray<2, LabelType>& multi_array,
  const PathType& path)
{
  vigra::ImageExportInfo export_info(path.string().c_str());
  vigra::exportImage(multi_array, export_info);
}

template<>
void save_multi_array<3>(
  vigra::MultiArray<3, LabelType>& multi_array,
  const PathType& path)
{
  vigra::VolumeExportInfo export_info(path.string().c_str());
  vigra::exportVolume(multi_array, export_info);
}

#ifdef USE_32_BIT_LABELS
template<>
void save_multi_array<3>(
  vigra::MultiArray<3, vigra::UInt16>& multi_array,
  const PathType& path)
{
  vigra::VolumeExportInfo export_info(path.string().c_str());
  vigra::exportVolume(multi_array, export_info);
}
#endif

void load_forests(
  RandomForestVectorType& rfs,
  const PathType path,
  const std::string name)
{
  std::string group = name + "/ClassifierForests/Forest";
  int num_leading_zeros = 4;
  int read_status = get_rfs_from_file(
    rfs, 
    path.string(),
    group,
    num_leading_zeros);
  if (!read_status) {
    throw std::runtime_error(
      "failed to open classifier forests in " + group);
  }
}

void load_features(
  StringDataPairVectorType& feature_list,
  const PathType path)
{
  int read_status = read_features_from_file(path.string(), feature_list);
  if (read_status == 1) {
    throw std::runtime_error("could not read feature file " + path.string());
  }
}

void load_features(
  std::vector<std::string>& feature_list,
  const PathType path)
{
  int read_status = read_region_features_from_file(path.string(), feature_list);
  if (read_status == 1) {
    throw std::runtime_error("could not read feature file " + path.string());
  }
}

Workflow::Workflow(bool calculate_segmentation, bool segmentation_dump) :
  calculate_segmentation_(calculate_segmentation),
  segmentation_dump_(segmentation_dump),
  has_mask_image_(false)
{
  if (calculate_segmentation_) {
    num_args_ = 9;
  } else {
    num_args_ = 8;
  }
}

void Workflow::init(int argc, char* argv[]) {
  if (argc < num_args_) {
    throw ArgumentError();
  }
  // get arguments
  size_t arg_index = 1;
  // directory of raw images
  PathType raw_dir = fs::system_complete(argv[arg_index]); arg_index++;
  PathType seg_dir = fs::system_complete(argv[arg_index]); arg_index++;
  PathType res_dir = fs::system_complete(argv[arg_index]); arg_index++;
  PathType cfg_file = fs::system_complete(argv[arg_index]); arg_index++;
  PathType classifier_file = fs::system_complete(argv[arg_index]); arg_index++;
  PathType pix_feature_file;
  if (calculate_segmentation_) {
    pix_feature_file = fs::system_complete(argv[arg_index]); arg_index++;
  }
  PathType cnt_feature_file = fs::system_complete(argv[arg_index]); arg_index++;
  PathType div_feature_file = fs::system_complete(argv[arg_index]); arg_index++;
  if (static_cast<size_t>(argc) > arg_index) {
    has_mask_image_ = true;
    mask_image_file_ = fs::system_complete(argv[arg_index]); arg_index++;
    std::cout << "Found Mask image option: " << mask_image_file_.string() << std::endl;
  }
  // check directories
  check_directory(raw_dir, false);
  if (calculate_segmentation_) {
    check_directory(seg_dir, true);
  } else {
    check_directory(seg_dir, false);
  }
  check_directory(res_dir, true);
  // load tracking config
  check_file(cfg_file);
  options_.load(cfg_file.string());
  if (!options_.is_legal()) {
    throw std::runtime_error("incomplete options for tracking");
  }
  // load the classifier
  check_file(classifier_file);
  if (calculate_segmentation_) {
    load_forests(pix_feature_rfs_, classifier_file, "PixelClassification");
  }
  load_forests(cnt_feature_rfs_, classifier_file, "CountClassification");
  load_forests(div_feature_rfs_, classifier_file, "DivisionDetection");
  // load the feature files
  if (calculate_segmentation_) {
    load_features(pix_feature_list_, pix_feature_file);
  }
  load_features(cnt_feature_list_, cnt_feature_file);
  load_features(div_feature_list_, div_feature_file);
  // get all filenames
  std::string placeholder = "###";
  if(options_.has_option<std::string>("FileNumberingPlaceholder"))
	  placeholder = options_.get_option<std::string>("FileNumberingPlaceholder");

  raw_path_vec_ = get_files(raw_dir, ".tif", true);
  if (calculate_segmentation_) {
    seg_path_vec_ = create_filenames(seg_dir, "seg" + placeholder + ".tif", raw_path_vec_.size());
  } else {
    seg_path_vec_ = get_files(seg_dir, ".tif", true);
    if (seg_path_vec_.size() != raw_path_vec_.size() 
      && seg_path_vec_.size() < (options_.get_option<int>("time_range_1") - options_.get_option<int>("time_range_0"))) {
      throw std::runtime_error(
        "count of segmentation and raw images not the same or too low");
    }
  }
  res_path_vec_ = create_filenames(res_dir, "mask" + placeholder + ".tif", raw_path_vec_.size());
  res_path_ = fs::system_complete(res_dir.string() + "/res_track.txt");
  traxelstore_dump_path_ = fs::system_complete(seg_dir.string() + "/traxelstore.dump");
}

void Workflow::dump_traxelstore(TraxelStoreType& ts) {
  std::ofstream dump(traxelstore_dump_path_.string());
  boost::archive::binary_oarchive a(dump);
  a << ts;
}

}
