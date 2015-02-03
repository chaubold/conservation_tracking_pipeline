/* TODO
 *  - Reduce function read_features_from_file and read_config_from_file
 *    to one function.
 */

//stl
#include <sstream>
#include <iomanip>
#include <vector>
#include <stdexcept>

// vigra
#include <vigra/random_forest_hdf5_impex.hxx>

// boost
#include <boost/tokenizer.hpp>

// own
#include "pipeline_helpers.hxx"

namespace isbi_pipeline {

////
//// ArgumentError
////
const char* ArgumentError::what() const throw() {
  return "Dataset folder and seqeunce not specified!\n";
}

/** @brief Convert a string to a double
 */
inline double string_to_double(std::string str) {
  std::istringstream is(str);
  double x;
  if (!(is >> x))
    throw std::runtime_error("Not convertable to double: " + str);
  return x;
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

/** @brief Read a csv table into a vector of string pairs.
 */
int read_features_from_file(std::string path, std::vector<std::pair<std::string, double> >& features) {
  std::ifstream f(path.c_str());
  if (!f.is_open()) {
    return 1;
  }
  typedef boost::tokenizer<boost::escaped_list_separator<char> > Tokenizer;
  std::string line;
  while(getline(f, line)) {
    Tokenizer tok(line);
    features.push_back(
      std::make_pair(*tok.begin(), string_to_double(*(++tok.begin()))));
  }
  f.close();
  return 0;
}

/** @brief Read a csv table into a map from string to double.
 */
int read_config_from_file(const std::string& path, std::map<std::string, double>& options) {
  std::ifstream f(path.c_str());
  if (!f.is_open()) return 1;

  typedef boost::tokenizer<boost::escaped_list_separator<char> > Tokenizer;
  std::string line;
  while(getline(f, line)) {
    Tokenizer tok(line);
    options[*tok.begin()] = string_to_double(*(++tok.begin()));
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

/** @brief Check if all required options are available.
 * TODO move hard coded parameters to header.
 */
bool has_required_options(const std::map<std::string, double>& options) {
  bool ret = true;
  ret = ret && (options.count("app")         != 0);
  ret = ret && (options.count("dis")         != 0);
  ret = ret && (options.count("det")         != 0);
  ret = ret && (options.count("mis")         != 0);
  ret = ret && (options.count("opp")         != 0);
  ret = ret && (options.count("for")         != 0);
  ret = ret && (options.count("mdd")         != 0);
  ret = ret && (options.count("min_angle")   != 0);
  ret = ret && (options.count("ep_gap")      != 0);
  ret = ret && (options.count("n_neighbors") != 0);
  return ret;
}

/** @brief Do the tracking.
 */
std::vector<std::vector<pgmlink::Event> > track(pgmlink::TraxelStore& ts, std::map<std::string, double> options) {
  if (!has_required_options(options)) {
    throw std::runtime_error("Options for chaingraph missing!");
  }
  pgmlink::ChaingraphTracking tracker(
    "none",                 // random forest filename
    options["app"],         // appearance
    options["dis"],         // disappearance
    options["det"],         // detection
    options["mis"],         // misdetection
    false,                  // cellness by rf
    options["opp"],         // opportunity cost
    options["for"],         // forbidden cost
    true,                   // with constraints
    false,                  // fixed detections
    options["mdd"],         // mean div dist
    options["min_angle"],   // min angle
    options["ep_gap"],      // ep_gap
    options["n_neighbors"], // n neighbors
    true,                   // with divisions
    1e+75,                  // cplex timeout
    false);                 // alternative builder
  return tracker(ts);
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
