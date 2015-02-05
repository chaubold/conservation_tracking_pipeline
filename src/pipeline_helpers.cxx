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
      ret = ret and check_option<std::string>("rf_filename");
      ret = ret and check_option<double>("app");
      ret = ret and check_option<double>("dis");
      ret = ret and check_option<double>("det");
      ret = ret and check_option<double>("mis");
      ret = ret and check_option<bool>  ("cellness_by_rf");
      ret = ret and check_option<double>("opp");
      ret = ret and check_option<double>("forbidden_cost");
      ret = ret and check_option<bool>  ("with_constraints");
      ret = ret and check_option<bool>  ("fixed_det");
      ret = ret and check_option<double>("mean_div_dist");
      ret = ret and check_option<double>("min_angle");
      ret = ret and check_option<double>("ep_gap");
      ret = ret and check_option<int>   ("n_neighbors");
      ret = ret and check_option<bool>  ("with_div");
      ret = ret and check_option<double>("cplex_timeout");
      ret = ret and check_option<bool>  ("alternative_builder");
    } else if (!tracker_type.compare("ConsTracking")) {
      ret = true;
      ret = ret and check_option<double>("lt");
      ret = ret and check_option<double>("lx");
      ret = ret and check_option<double>("ly");
      ret = ret and check_option<double>("lz");
      ret = ret and check_option<double>("ut");
      ret = ret and check_option<double>("ux");
      ret = ret and check_option<double>("uy");
      ret = ret and check_option<double>("uz");
      ret = ret and check_option<int>        ("max_number_obj");
      ret = ret and check_option<double>     ("max_neighbor_dist");
      ret = ret and check_option<double>     ("div_threshold");
      ret = ret and check_option<std::string>("rf_filename");
      ret = ret and check_option<bool>       ("size_dep_det_prob");
      ret = ret and check_option<double>     ("forbidden_cost");
      ret = ret and check_option<double>     ("ep_gap");
      ret = ret and check_option<double>     ("avg_obj_size");
      ret = ret and check_option<bool>       ("with_tracklets");
      ret = ret and check_option<double>     ("div_weight");
      ret = ret and check_option<double>     ("trans_weight");
      ret = ret and check_option<bool>       ("with_div");
      ret = ret and check_option<double>     ("dis_cost");
      ret = ret and check_option<double>     ("app_cost");
      ret = ret and check_option<bool>       ("with_merger_res");
      ret = ret and check_option<int>        ("n_dim");
      ret = ret and check_option<double>     ("trans_param");
      ret = ret and check_option<double>     ("border_width");
      ret = ret and check_option<bool>       ("with_constraints");
      ret = ret and check_option<double>     ("cplex_timeout");
      ret = ret and check_option<std::string>("ev_dump_file");
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
EventVectorVectorType track(
  pgmlink::TraxelStore& ts,
  const TrackingOptions& options)
{
  const std::string& tracker_type = options.get_option<std::string>("tracker");
  // create the ChaingraphTracking or ConsTracking class and call the ()-
  // operator
  if (!tracker_type.compare("ChaingraphTracking")) {
    pgmlink::ChaingraphTracking tracker(
      options.get_option<std::string>("rf_filename"), // random forest filename
      options.get_option<double>("app"),              // appearance
      options.get_option<double>("dis"),              // disappearance
      options.get_option<double>("det"),              // detection
      options.get_option<double>("mis"),              // misdetection
      options.get_option<bool>  ("cellness_by_rf"),   // cellness by rf
      options.get_option<double>("opp"),              // opportunity cost
      options.get_option<double>("forbidden_cost"),   // forbidden cost
      options.get_option<bool>  ("with_constraints"), // with constraints
      options.get_option<bool>  ("fixed_det"),        // fixed detections
      options.get_option<double>("mean_div_dist"),    // mean div dist
      options.get_option<double>("min_angle"),        // min angle
      options.get_option<double>("ep_gap"),           // ep_gap
      options.get_option<int>   ("n_neighbors"),      // n neighbors
      options.get_option<bool>  ("with_div"),         // with divisions
      options.get_option<double>("cplex_timeout"),    // cplex timeout
      options.get_option<bool>  ("alternative_builder")); // alternative builder
    return tracker(ts);
  } else if (!tracker_type.compare("ConsTracking")) {
    pgmlink::FieldOfView field_of_view(
      options.get_option<double>("lt"),
      options.get_option<double>("lx"),
      options.get_option<double>("ly"),
      options.get_option<double>("lz"),
      options.get_option<double>("ut"),
      options.get_option<double>("ux"),
      options.get_option<double>("uy"),
      options.get_option<double>("uz"));
    pgmlink::ConsTracking tracker(
      options.get_option<int>        ("max_number_obj"),
      options.get_option<double>     ("max_neighbor_dist"),
      options.get_option<double>     ("div_threshold"),
      options.get_option<std::string>("rf_filename"),
      options.get_option<bool>       ("size_dep_det_prob"),
      options.get_option<double>     ("forbidden_cost"),
      options.get_option<double>     ("ep_gap"),
      options.get_option<double>     ("avg_obj_size"),
      options.get_option<bool>       ("with_tracklets"),
      options.get_option<double>     ("div_weight"),
      options.get_option<double>     ("trans_weight"),
      options.get_option<bool>       ("with_div"),
      options.get_option<double>     ("dis_cost"),
      options.get_option<double>     ("app_cost"),
      options.get_option<bool>       ("with_merger_res"),
      options.get_option<int>        ("n_dim"),
      options.get_option<double>     ("trans_param"),
      options.get_option<double>     ("border_width"),
      field_of_view,
      options.get_option<bool>       ("with_constraints"),
      options.get_option<double>     ("cplex_timeout"),
      options.get_option<std::string>("ev_dump_file"));
    return tracker(ts);
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
