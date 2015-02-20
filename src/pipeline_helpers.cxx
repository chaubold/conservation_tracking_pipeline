/* TODO
 *  - Reduce function read_features_from_file and read_config_from_file
 *    to one function.
 */

//stl
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdexcept>

// vigra
#include <vigra/random_forest_hdf5_impex.hxx>

// boost
#include <boost/tokenizer.hpp>
#include <boost/lexical_cast.hpp>

// own
#include "pipeline_helpers.hxx"

namespace isbi_pipeline {

namespace fs = boost::filesystem;

// local typdef
typedef StringStringMapType::const_iterator StringStringMapConstItType;

// local functions
template<typename T> bool convertible_to(const std::string& str) {
  bool ret = true;
  try {
    static_cast<void>(boost::lexical_cast<T>(str));
  } catch (const boost::bad_lexical_cast&) {
    ret = false;
  }
  return ret;
}

////
//// ArgumentError
////
const char* ArgumentError::what() const throw() {
  return "Dataset folder and seqeunce not specified!\n";
}

////
//// class TrackingOptions
////
TrackingOptions::TrackingOptions(const std::string path) {
  std::ifstream file(path.c_str());
  if (!file.is_open()) {
    std::runtime_error("Could not open file \"" + path + "\"");
  }
  typedef boost::tokenizer<boost::escaped_list_separator<char> > TokenizerType;
  std::string line;
  while(std::getline(file, line)) {
    TokenizerType tokenizer(line);
    options_map_[*tokenizer.begin()] = *(++tokenizer.begin());
  }
  file.close();
}

template<typename T>
bool TrackingOptions::has_option(const std::string key) const {
  StringStringMapConstItType it = options_map_.find(key);
  if (it != options_map_.end()) {
    return convertible_to<T>(it->second);
  } else {
    return false;
  }
}

template<>
bool TrackingOptions::has_option<std::string>(const std::string key) const {
  return options_map_.count(key);
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
  StringStringMapConstItType it = options_map_.find(key);
  if (it != options_map_.end()) {
    return boost::lexical_cast<T>(it->second);
  } else {
    throw std::runtime_error("Option \"" + key + "\" not set");
  }
}

template<>
std::string TrackingOptions::get_option<std::string>(
  const std::string key
) const {
  StringStringMapConstItType it = options_map_.find(key);
  if (it != options_map_.end()) {
    return it->second;
  } else {
    throw std::runtime_error("Option \"" + key + "\" not set");
  }
}

bool TrackingOptions::is_legal() const {
  bool ret = check_option<std::string>("tracker");
  ret = ret and check_option<int>("borderWidth");
  ret = ret and check_option<int>("size_range_0");
  ret = ret and check_option<int>("size_range_1");
  ret = ret and check_option<int>("templateSize");
  if (ret) {
    const std::string tracker_type = get_option<std::string>("tracker");
    // check the tracker options
    if (!tracker_type.compare("ChaingraphTracking")) {
      ret = true;
      ret = ret and check_option<double>("appearanceCost");
      ret = ret and check_option<double>("disappearanceCost");
      ret = ret and check_option<double>("detection");
      ret = ret and check_option<double>("misdetection");
      ret = ret and check_option<double>("opportunityCost");
      ret = ret and check_option<double>("forbiddenCost");
      ret = ret and check_option<bool>  ("fixedDetections");
      ret = ret and check_option<double>("meanDivisionDist");
      ret = ret and check_option<double>("minAngle");
      ret = ret and check_option<double>("epGap");
      ret = ret and check_option<int>   ("nNeighbors");
      ret = ret and check_option<bool>  ("withDivisions");
      ret = ret and check_option<double>("cplex_timeout");
    } else if (!tracker_type.compare("ConsTracking")) {
      ret = true;
      ret = ret and check_option<double>("time_range_0");
      ret = ret and check_option<double>("x_range_0");
      ret = ret and check_option<double>("y_range_0");
      ret = ret and check_option<double>("z_range_0");
      ret = ret and check_option<double>("time_range_1");
      ret = ret and check_option<double>("x_range_1");
      ret = ret and check_option<double>("y_range_1");
      ret = ret and check_option<double>("z_range_1");
      ret = ret and check_option<int   >("maxObj");
      ret = ret and check_option<bool  >("sizeDependent");
      ret = ret and check_option<double>("avgSize");
      ret = ret and check_option<double>("maxDist");
      ret = ret and check_option<bool  >("withDivisions");
      ret = ret and check_option<double>("divThreshold");
      ret = ret and check_option<double>("forbiddenCost");
      ret = ret and check_option<double>("epGap");
      ret = ret and check_option<bool  >("withTracklets");
      ret = ret and check_option<double>("divWeight");
      ret = ret and check_option<double>("transWeight");
      ret = ret and check_option<double>("disappearanceCost");
      ret = ret and check_option<double>("appearanceCost");
      ret = ret and check_option<int   >("nDim");
      ret = ret and check_option<double>("transParameter");
      ret = ret and check_option<double>("borderAwareWidth");
      ret = ret and check_option<bool  >("withConstraints");
      ret = ret and check_option<double>("cplex_timeout");
    } else {
      std::cout << "Unknown tracker \"" << tracker_type << "\"" << std::endl;
      ret = false;
    }
  }
  return ret;
}


template<typename T>
void print_vector(std::ostream& stream, const std::vector<T>& vec) {
  if (vec.size() != 0) {
    stream << "(";
    for (size_t n = 0; n < (vec.size() - 1); n++) {
      stream << vec[n] << ", ";
    }
    stream << (vec.back()) << ")";
  } else {
    stream << "()";
  }
}

void print_traxelstore(std::ostream& stream, const TraxelStoreType& ts) {
  for(
    TraxelStoreType::const_iterator ts_it = ts.begin();
    ts_it != ts.end();
    ts_it++)
  {
    stream << *ts_it;
    FeatureMapType::const_iterator fmap_it = (ts_it->features).find("com");
    if (fmap_it != (ts_it->features).end()) {
      stream << " at ";
      print_vector(stream, fmap_it->second);
    }
    fmap_it = (ts_it->features).find("detProb");
    if (fmap_it != (ts_it->features).end()) {
      stream << ", detProb: ";
      print_vector(stream, fmap_it->second);
    }
    fmap_it = (ts_it->features).find("divProb");
    if (fmap_it != (ts_it->features).end()) {
      stream << ", divProb: ";
      print_vector(stream, fmap_it->second);
    }
    stream << std::endl;
  }
}

/** @brief Removes a single trailing character from the string
 */
void rstrip(char* c, char r) {
  while (*c != '\0') {
    if (*(c+1) == '\0' && *c == r) {
      *c = '\0';
    }
    c++;
  }
  return;
}

std::string zero_padding(int num, int n_zeros) {
  std::ostringstream ss;
  ss << std::setw(n_zeros) << std::setfill('0') << num;
  return ss.str();
}

/** @brief Read a csv table into a vector of string double pairs.
 */
int read_features_from_file(
  const std::string path,
  StringDataPairVectorType& features)
{
  std::ifstream f(path.c_str());
  if (!f.is_open()) {
    return 1;
  }
  typedef boost::tokenizer<boost::escaped_list_separator<char> > Tokenizer;
  std::string line;
  while(std::getline(f, line)) {
    Tokenizer tok(line);
    std::string key = *tok.begin();
    double value = boost::lexical_cast<double>(*(++tok.begin()));
    features.push_back(std::make_pair(key, value));
  }
  f.close();
  return 0;
}

int read_region_features_from_file(
  const std::string path,
  std::vector<std::string>& feature_list)
{
  std::ifstream f(path.c_str());
  if (!f.is_open()) {
    return 1;
  }
  feature_list.clear();
  std::string feature;
  while(std::getline(f, feature)) {
    feature_list.push_back(feature);
  }
  return 0;
}

/** @brief Read the random forests from the hdf5 files.
 */
bool get_rfs_from_file(
  RandomForestVectorType& rfs,
  std::string fn,
  std::string path_in_file,
  int n_forests,
  int n_leading_zeros)
{
  bool read_successful = false;
  for (int n = 0; n < n_forests; ++n) {
    std::string rf_path = path_in_file + zero_padding(n, n_leading_zeros);
    rfs.push_back(RandomForestType());
    read_successful = vigra::rf_import_HDF5(rfs[n], fn, rf_path);
  }
  return read_successful;
}

/** @brief Do the tracking.
 */
// TODO the whole following code is ugly -> will someone please come
// up with a cool idea?
EventVectorVectorType track(
  TraxelStoreType& ts,
  const TrackingOptions& options,
  const CoordinateMapPtrType& coordinate_map_ptr)
{
  const std::string& tracker_type = options.get_option<std::string>("tracker");
  // create the ChaingraphTracking or ConsTracking class and call the ()-
  // operator
  if (!tracker_type.compare("ChaingraphTracking")) {
    pgmlink::ChaingraphTracking tracker(
      "none", // random forest filename
      options.get_option<double>("appearanceCost"),
      options.get_option<double>("disappearanceCost"),
      options.get_option<double>("detection"),
      options.get_option<double>("misdetection"),
      false, // cellness by rf
      options.get_option<double>("opportunityCost"),
      options.get_option<double>("forbiddenCost"),
      true, // with constraints
      options.get_option<bool>  ("fixedDetections"),
      options.get_option<double>("meanDivisionDist"),
      options.get_option<double>("minAngle"),
      options.get_option<double>("epGap"),
      options.get_option<int>   ("nNeighbors"),
      options.get_option<bool>  ("withDivisions"),
      options.get_option<double>("cplex_timeout"),
      false); // alternative builder
    return tracker(ts);
  } else if (!tracker_type.compare("ConsTracking")) {
    // create the field of view
    pgmlink::FieldOfView field_of_view(
      options.get_option<double>("time_range_0"),
      options.get_option<double>("x_range_0"),
      options.get_option<double>("y_range_0"),
      options.get_option<double>("z_range_0"),
      options.get_option<double>("time_range_1"),
      options.get_option<double>("x_range_1"),
      options.get_option<double>("y_range_1"),
      options.get_option<double>("z_range_1"));
    // build the tracker
    pgmlink::ConsTracking tracker(
      options.get_option<int   >("maxObj"),
      options.get_option<bool  >("sizeDependent"),
      options.get_option<double>("avgSize"),
      options.get_option<double>("maxDist"),
      options.get_option<bool  >("withDivisions"),
      options.get_option<double>("divThreshold"),
      "none", // random forest filename
      field_of_view,
      "none"); // event_vector_dump_filename

    // std::cout << "Creating ConsTracking with parameters:" << std::endl;
    // std::cout << "\tmaxObj: " << options.get_option<int   >("maxObj") << std::endl;
    // std::cout << "\tsizeDependent: " << options.get_option<bool  >("sizeDependent") << std::endl;
    // std::cout << "\tavgSize: " << options.get_option<double>("avgSize") << std::endl;
    // std::cout << "\tmaxDist: " << options.get_option<double>("maxDist") << std::endl;
    // std::cout << "\twithDivisions: " << options.get_option<bool  >("withDivisions") << std::endl;
    // std::cout << "\tdivThreshold: " << options.get_option<double>("divThreshold") << std::endl;

    // build the hypotheses graph
    tracker.build_hypo_graph(ts);
    // track
    boost::shared_ptr<EventVectorVectorType> ret_ptr(
      new EventVectorVectorType(tracker.track(
        options.get_option<double>("forbiddenCost"),
        options.get_option<double>("epGap"),
        options.get_option<bool  >("withTracklets"),
        options.get_option<double>("divWeight"),
        options.get_option<double>("transWeight"),
        options.get_option<double>("disappearanceCost"),
        options.get_option<double>("appearanceCost"),
        options.get_option<int   >("nDim"),
        options.get_option<double>("transParameter"),
        options.get_option<double>("borderAwareWidth"),
        options.get_option<bool  >("withConstraints"),
        options.get_option<double>("cplex_timeout"))));

    // std::cout << "\n\nCalling tracking with parameters:" << std::endl;
    // std::cout << "\tforbiddenCost: " << options.get_option<double>("forbiddenCost") << std::endl;
    // std::cout << "\tepGap: " << options.get_option<double>("epGap") << std::endl;
    // std::cout << "\twithTracklets: " << options.get_option<bool  >("withTracklets") << std::endl;
    // std::cout << "\tdivWeight: " << options.get_option<double>("divWeight") << std::endl;
    // std::cout << "\ttransWeight: " << options.get_option<double>("transWeight") << std::endl;
    // std::cout << "\tdisappearanceCost: " << options.get_option<double>("disappearanceCost") << std::endl;
    // std::cout << "\tappearanceCost: " << options.get_option<double>("appearanceCost") << std::endl;
    // std::cout << "\tnDim: " << options.get_option<int   >("nDim") << std::endl;
    // std::cout << "\ttransParameter: " << options.get_option<double>("transParameter") << std::endl;
    // std::cout << "\tborderAwareWidth: " << options.get_option<double>("borderAwareWidth") << std::endl;
    // std::cout << "\twithConstraints: " << options.get_option<bool  >("withConstraints") << std::endl;
    // std::cout << "\tcplex_timeout: " << options.get_option<double>("cplex_timeout") << std::endl;

    // merger resolving
    return tracker.resolve_mergers(
      ret_ptr,
      coordinate_map_ptr,
      options.get_option<double>("epGap"),
      options.get_option<double>("transWeight"),
      options.get_option<bool  >("withTracklets"),
      options.get_option<int   >("nDim"),
      options.get_option<double>("transParameter"),
      options.get_option<bool  >("withConstraints"),
      false); // with multi frame moves
  } else {
    // throw an error
    throw std::runtime_error("Unknown tracker \"" + tracker_type + "\"");
    // return an empty vector to remove compiler warnings
    EventVectorVectorType ev;
    return ev;
  }
}

/** @brief Check if a string contains a substring.
 */
bool contains_substring(std::string str, std::string substr) {
  std::string::size_type index = str.find(substr);
  return index != std::string::npos;
}

/** @brief Check if a boost path name contains a substring.
 */
bool contains_substring_boost_path(
  const DirectoryEntryType& p,
  std::string substr)
{
  return contains_substring(p.path().string(), substr);
}

void check_directory(const PathType& path, bool create_if_not) {
  // does path exist?
  if (!fs::exists(path)) {
    // if not, check if it should be created
    if (create_if_not) {
      // create directory
      if (!fs::create_directory(path)) {
        throw std::runtime_error("Could not create directory: " + path.string());
      }
    } else {
      // else throw error
      throw std::runtime_error(path.string() + " does not exist");
    }
  } else if (!fs::is_directory(path)) {
    // throw an error if the path exists but is not a directory
    throw std::runtime_error(path.string() + " is not a directory");
  }
}

std::vector<PathType> get_files(
  const PathType& path,
  const std::string extension_filter,
  bool sort)
{
  std::vector<PathType> ret;
  for(DirectoryIteratorType it(path); it != DirectoryIteratorType(); it++) {
    std::string extension((it->path()).extension().string());
    if (!extension.compare(extension_filter)) {
      ret.push_back(*it);
    }
  }
  if (sort) {
    std::sort(ret.begin(), ret.end());
  }
  return ret;
}

} // namespace isbi_pipeline
