/* TODO
 * const correctness for the old functions
 * limit code width
 */
#ifndef PIPELINE_HELPERS_HXX
#define PIPELINE_HELPERS_HXX

//stl
#include <string>
#include <vector>
#include <utility>
#include <exception>
#include <map>

// vigra
#include <vigra/random_forest.hxx>

// boost
#include <boost/filesystem.hpp>
#include <boost/lexical_cast.hpp>

// pgmlink
#include <pgmlink/tracking.h>
#include <pgmlink/traxels.h>

// own
#include "common.h"

namespace isbi_pipeline {

template<typename T> bool convertible_to(const std::string value);

// ArgumentError class
class ArgumentError : public std::exception {
public:
  virtual const char* what() const throw();
};

class TrackingOptions {
 public:
  TrackingOptions() {};
  TrackingOptions(const std::string path);
  void load(const std::string path);
  bool is_legal() const;
  template<typename T> bool has_option(const std::string key) const;
  template<typename T> bool check_option(const std::string key) const;
  template<typename T> T get_option(const std::string key) const;
  template<typename T, int N>
  vigra::TinyVector<T, N> get_vector_option(const std::string key) const;
 private:
  StringStringMapType options_map_;
};

template<int N>
void get_bounding_box(
  const TrackingOptions& options,
  vigra::TinyVector<LabelType, N>& bb_min,
  vigra::TinyVector<LabelType, N>& bb_max);


// for logging of the traxelstore
void print_traxelstore(std::ostream& stream, const TraxelStoreType& ts);


// strip trailing slash
// works only for null terminated cstrings
void rstrip(char* c, char r);


// zero padding int to string
std::string zero_padding(int num, int n_zeros);


// read features from csv_style file
int read_features_from_file(
  const std::string path,
  StringDataPairVectorType& features);

// read region feature list from file
int read_region_features_from_file(
  const std::string path,
  std::vector<std::string>& feature_list);

// get random_forests from file
bool get_rfs_from_file(
  RandomForestVectorType& rfs,
  std::string fn,
  std::string path_in_file = "PixelClassification/ClassifierForests/Forest",
  int n_leading_zeros = 4);


// do the tracking
EventVectorVectorType track(
  TraxelStoreType& ts,
  const TrackingOptions& options,
  const CoordinateMapPtrType& coordinate_map_ptr = CoordinateMapPtrType(),
  const std::vector<pgmlink::Traxel>& traxels_to_keep_in_first_frame = {});


// helper function to iterate over tif only
bool contains_substring(std::string str, std::string substr);


// same for boost_path
bool contains_substring_boost_path(
  const DirectoryEntryType& p,
  std::string substr);


// check if a directory exists
void check_directory(const PathType& path, bool create_if_not = false);
void check_file(const PathType& path);


// get files in a path
std::vector<PathType> get_files(
  const PathType& path,
  const std::string extension_filter,
  bool sort = false);

std::vector<PathType> create_filenames(
  const PathType& path,
  const std::string mask,
  size_t size,
  size_t offset = 0);


// copy_if_own because not using c++11
template <class InputIterator, class OutputIterator, class UnaryPredicate>
OutputIterator copy_if_own (
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryPredicate pred);


template<typename T>
void read_volume(T& volume, const std::string& filename);

/* -------------------------------------------------- */
/*                   IMPLEMENTATION                   */
/* -------------------------------------------------- */
template<typename T> bool convertible_to(const std::string str) {
  bool ret = true;
  try {
    static_cast<void>(boost::lexical_cast<T>(str));
  } catch (const boost::bad_lexical_cast& error) {
    ret = false;
  }
  return ret;
}

template<typename T>
bool TrackingOptions::has_option(const std::string key) const {
  typedef StringStringMapType::const_iterator StringStringMapConstItType;
  StringStringMapConstItType it = options_map_.find(key);
  if (it != options_map_.end()) {
    return convertible_to<T>(it->second);
  } else {
    return false;
  }
}

template<typename T>
bool TrackingOptions::check_option(const std::string key) const {
  if (has_option<T>(key)) {
    return true;
  } else {
    std::cout << "Option \"" << key << "\" not legal" << std::endl;
    return false;
  }
}

template<typename T>
T TrackingOptions::get_option(const std::string key) const {
  typedef StringStringMapType::const_iterator StringStringMapConstItType;
  StringStringMapConstItType it = options_map_.find(key);
  if (it != options_map_.end()) {
    return boost::lexical_cast<T>(it->second);
  } else {
    throw std::runtime_error("Option \"" + key + "\" not set");
  }
}

template<typename T, int N>
vigra::TinyVector<T, N> TrackingOptions::get_vector_option(const std::string key) const {
  vigra::TinyVector<T, N> ret;
  for (int i = 0; i < N; i++) {
    std::stringstream sstream;
    sstream << key << "_" << i;
    ret[i] = get_option<T>(sstream.str());
  }
  return ret;
}

template <class InputIterator, class OutputIterator, class UnaryPredicate>
OutputIterator copy_if_own(
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  UnaryPredicate pred)
{
  while (first!=last) {
    if (pred(*first)) {
      *result = *first;
      ++result;
    }
    ++first;
  }
  return result;
}

// workaround since Destructor of VolumeImportInfo gives Segfaul:
template<typename T>
void read_volume(T& volume, const std::string& filename) {
  vigra::ImageImportInfo info(filename.c_str());
  vigra::Shape3 shape(info.shape()[0], info.shape()[1], info.numImages());
  volume.reshape(shape);
  for(int i = 0; i < info.numImages(); i++) {
    info.setImageIndex(i);
    vigra::importImage(info, volume.bindOuter(i));
  }
}

} // namespace isbi_pipeline

#endif /* PIPELINE_HELPERS_HXX */
