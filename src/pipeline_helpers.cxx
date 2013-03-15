//stl
#include <string>
#include <sstream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <utility>
#include <stdexcept>
#include <map>
#include <algorithm>

// vigra
#include <vigra/multi_array.hxx>
#include <vigra/convolution.hxx>
#include <vigra/multi_convolution.hxx>
#include <vigra/tinyvector.hxx>
#include <vigra/functorexpression.hxx>

// boost
#include <boost/tokenizer.hpp>
#include <boost/filesystem.hpp>

// pgmlink
#include <pgmlink/traxels.h>
#include <pgmlink/tracking.h>

// own
#include "pipeline_helpers.hxx"


////
//// ArgumentError
////
const char* ArgumentError::what() const throw() {return "Dataset folder and seqeunce not specified!\n";}


////
//// Lineage
////
Lineage::Lineage(int id, int t_start, int t_end, int parent, int o_id) :
  id_(id), t_start_(t_start), t_end_(t_end), parent_(parent), o_id_(o_id) {
  // do nothing, just initialize member variables
}


bool operator<(const Lineage& lhs, const Lineage& rhs) {
  return lhs.id_ < rhs.id_;
}

bool operator==(const Lineage& lhs, const int& rhs) {
  return lhs.id_ == rhs;
}

/*std::stringstream& operator<<(std::stringstream& ss, const Lineage& lin) {
  //ss << lin.id_; // << " " << lin.t_start_ << " " << lin.t_end_ << " " << lin.parent_;
  ss << lin.t_start_;
  return ss;
  }*/

std::ostream& operator<<(std::ostream& os, const Lineage& lin) {
  os << lin.id_ << " " << lin.t_start_ << " " << lin.t_end_ << " " << lin.parent_;
  return os;
}


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


int read_features_from_file(std::string path, std::vector<std::pair<std::string, std::string> >& features) {
  std::ifstream f(path.c_str());
  if (!f.is_open()) return 1;

  typedef boost::tokenizer<boost::escaped_list_separator<char> > Tokenizer;
  std::string line;
  while(getline(f, line)) {
    Tokenizer tok(line);
    features.push_back(std::make_pair(*tok.begin(), *(++tok.begin())));
  }
  f.close();
  return 0;
}


int feature_dim_lookup_size(std::string feature) {
  std::map<std::string, int> dims;
  dims["GaussianSmoothing"] = 1;
  dims["LaplacianOfGaussians"] = 1;
  dims["StructureTensorEigenvalues"] = 3;
  dims["HessianOfGaussianEigenvalues"] = 3;
  dims["GaussianGradientMagnitude"] = 1;
  dims["DifferenceOfGaussians"] = 1;
  std::map<std::string, int>::iterator it = dims.find(feature);
  if (it == dims.end()) return 0;
  else return it->second;
}


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


// do the tracking
std::vector<std::vector<pgmlink::Event> > track(pgmlink::TraxelStore& ts, std::map<std::string, double> options) {
  if (options.count("app") == 0 ||
      options.count("dis") == 0 ||
      options.count("det") == 0 ||
      options.count("mis") == 0 ||
      options.count("opp") == 0 ||
      options.count("for") == 0 ||
      options.count("mdd") == 0 ||
      options.count("min_angle") == 0 ||
      options.count("ep_gap") == 0 ||
      options.count("n_neighbors") == 0) {
    throw std::runtime_error("Options for chaingraph missing!");
  }
  pgmlink::ChaingraphTracking tracker("none",
                                      options["app"], // appearance
                                      options["dis"], // disappearance
                                      options["det"], // detection
                                      options["mis"], // misdetection
                                      false, // cellness by rf
                                      options["opp"], // opportunity cost
                                      options["for"], // forbidden cost
                                      true, // with constraints
                                      false, // fixed detections
                                      options["mdd"], // mean div dist
                                      options["min_angle"], // min angle
                                      options["ep_gap"], // ep_gap
                                      options["n_neighbors"], // n neighbors
                                      false // alternative builder
                                      );
  return tracker(ts);
}


int find_lineage_by_o_id(const std::vector<Lineage>& lineage_vec, int o_id) {
  return std::find(lineage_vec.begin(), lineage_vec.end(), o_id) - lineage_vec.begin();
}


void close_open_lineages(std::vector<Lineage>& lineage_vec, int t_end) {
  for (std::vector<Lineage>::iterator it = lineage_vec.begin(); it != lineage_vec.end(); ++it) {
    if (it->t_end_ == -1) {
      it->t_end_ = t_end;
    } else {
      continue;
    }
  }
}


int handle_move(std::vector<Lineage>& lineage_vec, std::vector<std::pair<unsigned, int> >& lineages_to_be_relabeled, const event_array& move) {
  int from = move[0];
  int to = move[1];
  int lineage_index = find_lineage_by_o_id(lineage_vec, from);
  if (static_cast<unsigned>(lineage_index) == lineage_vec.size()) {
    return 1;
  }
  
  lineages_to_be_relabeled.push_back(std::make_pair<unsigned, int>(lineage_index, to));
  return 0;
}


int handle_split(std::vector<Lineage>& lineage_vec, std::vector<std::pair<unsigned, int> >& lineages_to_be_relabeled, const event_array& split, int timestep, int& max_l_id) {
  int from = split[0];
  int lineage_index = find_lineage_by_o_id(lineage_vec, from);
  if (static_cast<unsigned>(lineage_index) == lineage_vec.size()) {
    return 1;
  }
  const Lineage& lin = lineage_vec[lineage_index];
  for (event_array::const_iterator it = ++split.begin(); it != split.end(); ++it) {
    Lineage child(max_l_id, timestep, -1, lin.id_, *it);
    lineages_to_be_relabeled.push_back(std::make_pair<unsigned, int>(lineage_vec.size(), *it));
    lineage_vec.push_back(child);
    max_l_id += 1;
  }
  lineage_vec[lineage_index].t_end_ = timestep - 1;
  lineage_vec[lineage_index].o_id_ = -1;
  return 0;
}


int handle_disappearance(std::vector<Lineage>& lineage_vec, const event_array& disappearance, int timestep) {
  int dis = disappearance[0];
  int lineage_index = find_lineage_by_o_id(lineage_vec, dis);
  if (static_cast<unsigned>(lineage_index) == lineage_vec.size()) {
    return 1;
  }
  // Lineage& lin = lineage_vec[lineage_index];
  lineage_vec[lineage_index].t_end_ = timestep - 1;
  lineage_vec[lineage_index].o_id_ = -1;
  return 0;
}


int handle_appearance(std::vector<Lineage>& lineage_vec, std::vector<std::pair<unsigned, int> >& lineages_to_be_relabeled, const event_array& appearance, int timestep, int& max_l_id) {
  int app = appearance[0];
  int lineage_index = find_lineage_by_o_id(lineage_vec, app);
  if (static_cast<unsigned>(lineage_index) != lineage_vec.size()) {
    return 1;
  }
  Lineage lin(max_l_id, timestep, -1, 0, app);
  lineage_vec.push_back(lin);
  lineages_to_be_relabeled.push_back(std::make_pair<unsigned, int>(lineage_index, app));
  max_l_id += 1;
  return 0;
}


int write_lineages(const std::vector<Lineage>& lineage_vec, std::string filename) {
  std::ofstream f(filename.c_str());
  if (!f.is_open()) {
    return 1;
  }

  for (std::vector<Lineage>::const_iterator it = lineage_vec.begin(); it != lineage_vec.end(); ++it) {
    f << *it << "\n";
  }
  f.close();
  return 0;
}


bool contains_substring(std::string str, std::string substr) {
  std::string::size_type index = str.find(substr);
  return index != std::string::npos;
}


bool contains_substring_boost_path(const boost::filesystem::directory_entry& p, std::string substr) {
  return contains_substring(p.path().string(), substr);
}
