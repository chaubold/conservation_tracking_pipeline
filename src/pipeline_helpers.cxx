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
// local typdef
typedef std::map<std::string, std::string> StringStringMapType;
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
      ret = ret and check_option<double>("cplexTimeout");
    } else {
      std::cout << "Unknown tracker \"" << tracker_type << "\"" << std::endl;
      ret = false;
    }
  }
  return ret;
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
  StringDoublePairVectorType& features)
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
bool get_rfs_from_file(std::vector<vigra::RandomForest<unsigned> >& rfs, std::string fn, std::string path_in_file, int n_forests, int n_leading_zeros) {
  bool read_successful = false;
  for (int n = 0; n < n_forests; ++n) {
    std::string rf_path = path_in_file + zero_padding(n, n_leading_zeros);
    rfs.push_back(vigra::RandomForest<unsigned>());
    read_successful = vigra::rf_import_HDF5(rfs[n], fn, rf_path);
  }
  return read_successful;
}

/** @brief Do the tracking.
 */
// TODO the whole following code is ugly -> will someone please come
// up with a cool idea?
EventVectorVectorType track(
  pgmlink::TraxelStore& ts,
  const TrackingOptions& options)
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
      options.get_option<double>("t_range_1"),
      options.get_option<double>("t_range_1"),
      options.get_option<double>("t_range_1"));
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
    // build the hypotheses graph
    tracker.build_hypo_graph(ts);
    // track
    EventVectorVectorType ret = tracker.track(
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
      options.get_option<double>("cplexTimeout"));
    // TODO handle merger resolving
    return ret;
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
bool contains_substring_boost_path(const boost::filesystem::directory_entry& p, std::string substr) {
  return contains_substring(p.path().string(), substr);
}

} // namespace isbi_pipeline
