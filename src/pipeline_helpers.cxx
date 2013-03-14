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

// own
#include "pipeline_helpers.hxx"


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
