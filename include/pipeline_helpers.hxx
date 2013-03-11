#ifndef PIPELINE_HELPERS_HXX
#define PIPELINE_HELPERS_HXX

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

// boost
#include <boost/tokenizer.hpp>


typedef vigra::MultiArray<2, double> feature_image;


// ArgumentError class
class ArgumentError : public std::exception {
public:
  virtual const char* what() const throw() {return "Dataset folder and seqeunce not specified!\n";}
};


// strip trailing slash
// works only for null terminated cstrings
void rstrip(char* c, char r) {
  while (*c != '\0') {
    if (*(c+1) == '\0' && *c == r) {
      *c = '\0';
    }
    c++;
  }
  return;
}


// remap image to [0,255]
template <typename T, int N>
void remap(vigra::MultiArray<N, T>& src) {
  T max_value = *vigra::argMax(src.begin(), src.end());
  if (!(max_value > 255)) return;
  T min_value = *vigra::argMin(src.begin(), src.end());
  T range = max_value - min_value;
  typename vigra::MultiArray<N, T>::iterator it = src.begin();
  for (; it != src.end(); ++it) {
    *it = 255*(*it - min_value)/(range);
  }
}


// zero padding int to string
std::string zero_padding(int num, int n_zeros) {
  std::ostringstream ss;
  ss << std::setw(n_zeros) << std::setfill('0') << num;
  return ss.str();
}


// read features from csv_style file
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

// lookup the size of given feature
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

// get features from string
int get_features(const vigra::MultiArray<2, unsigned>& src, std::vector<vigra::MultiArray<2, double> >& dest, std::string feature, double scale) {
  if (dest.size()) return 2;
  if (!feature.compare("GaussianSmoothing")) {
    dest.push_back(vigra::MultiArray<2, double>(src.shape()));
    vigra::gaussianSmoothing(srcImageRange(src), destImage(dest[0]), scale);
  } else if (!feature.compare("LaplacianOfGaussians")) {
    dest.push_back(vigra::MultiArray<2, double>(src.shape()));
    vigra::laplacianOfGaussian(srcImageRange(src), destImage(dest[0]), scale);
  } else if (!feature.compare("StructureTensorEigenvalues")) {
    for (int i = 0; i < 3; ++i) dest.push_back(vigra::MultiArray<2, double>(src.shape()));
    vigra::structureTensor(srcImageRange(src), destImage(dest[0]), destImage(dest[1]), destImage(dest[2]), scale, scale);
  } else if (!feature.compare("HessianOfGaussianEigenvalues")) {
    for (int i = 0; i < 3; ++i) dest.push_back(vigra::MultiArray<2, double>(src.shape()));
    vigra::hessianMatrixOfGaussian(srcImageRange(src), destImage(dest[0]), destImage(dest[1]), destImage(dest[2]), scale);
  } else if (!feature.compare("GaussianGradientMagnitude")) {
    dest.push_back(vigra::MultiArray<2, double>(src.shape()));
    vigra::gaussianGradientMagnitude(srcImageRange(src), destImage(dest[0]), scale);
  } else if (!feature.compare("DifferenceOfGaussians")) {
    dest.push_back(vigra::MultiArray<2, double>(src.shape()));
    vigra::laplacianOfGaussian(srcImageRange(src), destImage(dest[0]), scale);
  } else return 1;
  return 0;
}

// extract number from string
inline double string_to_double(std::string str) {
  std::istringstream is(str);
  double x;
  if (!(is >> x))
    throw std::runtime_error("Not convertable to double: " + str);
  return x;
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

template <int N, class T>
void renormalize_to_8bit(vigra::MultiArrayView<N, T>& array, double min, double max) {
  double range = (max - min);
  double val;
  typename vigra::MultiArrayView<N, T>::iterator it = array.begin();
  for (; it != array.end(); ++it) {
    val = 255.0*(*it - min)/range;
    *it = static_cast<T>(std::min(std::max(val, 0.0), 255.0));
  }
}


#endif /* PIPELINE_HELPERS_HXX */
