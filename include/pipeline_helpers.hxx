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
#include <vigra/multi_convolution.hxx>
#include <vigra/tinyvector.hxx>
#include <vigra/functorexpression.hxx>
#include <vigra/multi_tensorutilities.hxx>

// boost
#include <boost/tokenizer.hpp>


typedef vigra::MultiArray<2, double> feature_image;
typedef float FEATURETYPE;


// ArgumentError class
class ArgumentError;


// strip trailing slash
// works only for null terminated cstrings
void rstrip(char* c, char r);


// remap image to [0,255]
template <typename T, int N>
void remap(vigra::MultiArray<N, T>& src);


// zero padding int to string
std::string zero_padding(int num, int n_zeros);


// read features from csv_style file
int read_features_from_file(std::string path, std::vector<std::pair<std::string, std::string> >& features);


// lookup the size of given feature
int feature_dim_lookup_size(std::string feature);


// ilastik pixelclassification style difference of gaussians
template <int N>
void differenceOfGaussians(const vigra::MultiArray<N, unsigned>& src, vigra::MultiArray<N, FEATURETYPE>& dest, double scale0, double scale1, vigra::ConvolutionOptions<N> opt);


// gaussianGradientMagnitude for MultiArrays
template <int N>
void gaussianGradientMagnitudeOwn(const vigra::MultiArray<N, unsigned>& src, vigra::MultiArray<N, FEATURETYPE>& dest, double scale, vigra::ConvolutionOptions<N> opt);


// structureTensorEigenvalues for vector<MultiArray> instead of MultiArray<N, TinyVector>
template <int N>
void structureTensorEigenValuesOwn(const vigra::MultiArray<N, unsigned>& src, std::vector<vigra::MultiArray<N, FEATURETYPE> >& dest, double inner_scale, double outer_scale, vigra::ConvolutionOptions<N> opt);


// hessianOfGaussianEigenvalues for vector<MultiArray> instead of MultiArray<N, TinyVector>
template <int N>
void hessianOfGaussianEigenvaluesOwn(const vigra::MultiArray<N, unsigned>& src, std::vector<vigra::MultiArray<N, FEATURETYPE> >& dest, double scale, vigra::ConvolutionOptions<N> opt);


// MultiArray<N, TinyVector<T, U> > to vector<MultiArray<N, T> >
template <int N, class T, int U>
void multi_array_of_tiny_vec_to_vec_of_multi_array(const vigra::MultiArray<N, vigra::TinyVector<T, U> >& src,
						   std::vector<vigra::MultiArray<N, T> >& dest);


// get features from string storing in vector of MultiArrays
template <int N>
int get_features(const vigra::MultiArray<N, unsigned>& src, std::vector<vigra::MultiArray<N, FEATURETYPE> >& dest, std::string feature, double scale);


// get features from string storing in MultiArray with Dim+1
template <int N>
int get_features(const vigra::MultiArray<N, unsigned>& src, vigra::MultiArrayView<N+1, FEATURETYPE> dest, std::string feature, double scale, int pos_in_res_array, double window_size = 2.0);


// extract number from string
double string_to_double(std::string str);


// read config from file (such as renormalization parameters,...)
int read_config_from_file(const std::string& path, std::map<std::string, double>& options);


// renormalize multiarray to 8bit
template <int N, class T>
void renormalize_to_8bit(vigra::MultiArrayView<N, T>& array, double min, double max);



/* -------------------------------------------------- */
/*                   IMPLEMENTATION                   */
/* -------------------------------------------------- */


class ArgumentError : public std::exception {
public:
  virtual const char* what() const throw() {return "Dataset folder and seqeunce not specified!\n";}
};


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


template <int N>
void differenceOfGaussians(const vigra::MultiArray<N, unsigned>& src, vigra::MultiArray<N, FEATURETYPE>& dest, double scale0, double scale1, vigra::ConvolutionOptions<N> opt) {
  vigra::MultiArray<N, FEATURETYPE> tmp(src.shape());
  vigra::gaussianSmoothMultiArray(srcMultiArrayRange(src), destMultiArray(dest), scale0, opt);
  vigra::gaussianSmoothMultiArray(srcMultiArrayRange(src), destMultiArray(tmp), scale1, opt);
  dest -= tmp;
}


template <int N>
void gaussianGradientMagnitudeOwn(const vigra::MultiArray<N, unsigned>& src, vigra::MultiArray<N, FEATURETYPE>& dest, double scale, vigra::ConvolutionOptions<N> opt) {
  vigra::MultiArray<N, vigra::TinyVector<FEATURETYPE, N> > tmp(src.shape());
  vigra::gaussianGradientMultiArray(srcMultiArrayRange(src), destMultiArray(tmp), scale, opt);
  if (N == 2)
    vigra::combineTwoMultiArrays(srcMultiArrayRange(tmp),
				 srcMultiArray(dest),
				 destMultiArray(dest),
				 vigra::functor::squaredNorm(vigra::functor::Arg1()));
  else
    throw std::runtime_error("gaussianGradientMagnitude only implemented for N == 2 and N == 3");
  vigra::transformMultiArray(srcMultiArrayRange(dest),
			     destMultiArray(dest),
			     vigra::functor::sqrt(vigra::functor::Arg1()));
  
}


template <int N>
void structureTensorEigenvaluesOwn(const vigra::MultiArray<N, unsigned>& src, std::vector<vigra::MultiArray<N, FEATURETYPE> >& dest, double inner_scale, double outer_scale, vigra::ConvolutionOptions<N> opt) {
  vigra::MultiArray<N, vigra::TinyVector<FEATURETYPE, (N*(N+1))/2> > tmp1(src.shape());
  vigra::MultiArray<N, vigra::TinyVector<FEATURETYPE, N> > tmp2(src.shape());
  vigra::structureTensorMultiArray(srcMultiArrayRange(src), destMultiArray(tmp1), inner_scale, outer_scale, opt);
  vigra::tensorEigenvaluesMultiArray(srcMultiArrayRange(tmp1), destMultiArray(tmp2));
  multi_array_of_tiny_vec_to_vec_of_multi_array<N, FEATURETYPE, N>(tmp2, dest);
}


template <int N>
void hessianOfGaussianEigenvaluesOwn(const vigra::MultiArray<N, unsigned>& src, std::vector<vigra::MultiArray<N, FEATURETYPE> >& dest, double scale, vigra::ConvolutionOptions<N> opt) {
  vigra::MultiArray<N, vigra::TinyVector<FEATURETYPE, (N*(N+1))/2> > tmp1(src.shape());
  vigra::MultiArray<N, vigra::TinyVector<FEATURETYPE, N> > tmp2(src.shape());
  vigra::hessianOfGaussianMultiArray(srcMultiArrayRange(src), destMultiArray(tmp1), scale, opt);
  vigra::tensorEigenvaluesMultiArray(srcMultiArrayRange(tmp1), destMultiArray(tmp2));
  multi_array_of_tiny_vec_to_vec_of_multi_array<N, FEATURETYPE, N>(tmp2, dest);
}


template <int N, class T, int U>
void multi_array_of_tiny_vec_to_vec_of_multi_array(const vigra::MultiArray<N, vigra::TinyVector<T, U> >& src,
						   std::vector<vigra::MultiArray<N, T> >& dest) {
  vigra::MultiArrayView<N+1, T, vigra::StridedArrayTag> tmp = src.expandElements(N);
  for (int i = 0; i < U; ++i) {
    dest.push_back(tmp.bindOuter(i));
  }
  //vigra::combineTwoMultiArrays(srcMultiArrayRange(src), srcMultiArray(dest), destMultiArray(dest),
}


template <int N>
int get_features(const vigra::MultiArray<N, unsigned>& src, std::vector<vigra::MultiArray<N, FEATURETYPE> >& dest, std::string feature, double scale) {
  if (dest.size()) return 2;
  vigra::ConvolutionOptions<N> opt;
  opt.filterWindowSize(2.0);
  if (!feature.compare("GaussianSmoothing")) {
    dest.push_back(vigra::MultiArray<N, FEATURETYPE>(src.shape()));
    vigra::gaussianSmoothMultiArray(srcMultiArrayRange(src), destMultiArray(dest[0]), scale, opt);
  } else if (!feature.compare("LaplacianOfGaussians")) {
    dest.push_back(vigra::MultiArray<N, FEATURETYPE>(src.shape()));
    vigra::laplacianOfGaussianMultiArray(srcMultiArrayRange(src), destMultiArray(dest[0]), scale, opt);
  } else if (!feature.compare("StructureTensorEigenvalues")) {
    structureTensorEigenvaluesOwn<N>(src, dest, scale, 0.5*scale, opt);
  } else if (!feature.compare("HessianOfGaussianEigenvalues")) {
    hessianOfGaussianEigenvaluesOwn<N>(src, dest, scale, opt);
  } else if (!feature.compare("GaussianGradientMagnitude")) {
    dest.push_back(vigra::MultiArray<N, FEATURETYPE>(src.shape()));
    gaussianGradientMagnitudeOwn<N>(src, dest[0], scale, opt);
  } else if (!feature.compare("DifferenceOfGaussians")) {
    dest.push_back(vigra::MultiArray<N, FEATURETYPE>(src.shape()));
    differenceOfGaussians<N>(src, dest[0], scale, 0.66*scale, opt);
  } else return 1;
  return 0;			     
}


template <int N>
int get_features(const vigra::MultiArray<N, unsigned>& src, vigra::MultiArrayView<N+1, FEATURETYPE>& dest, std::string feature, double scale, double window_size) {
  vigra::ConvolutionOptions<N> opt;
  opt.filterWindowSize(window_size);
  if (feature.compare("GaussianSmoothing") == 0) {
    vigra::gaussianSmoothMultiArray(srcMultiArrayRange(src), destMultiArray(dest.bindOuter(0)), scale, opt);
  } else if (feature.compare("LaplacianOfGaussians") == 0) {
    vigra::laplacianOfGaussianMultiArray(srcMultiArrayRange(src), destMultiArray(dest.bindOuter(0)), scale, opt);
  } else if (feature.compare("StructureTensorEigenvalues") == 0) {
    return 1;
  } else if (feature.compare("HessianOfGaussianEigenvalues") == 0) {
    return 1;
  } else if (feature.compare("GaussianGradientMagnitude") == 0) {
    gaussianGradientMagnitudeOwn<N>(src, dest.binOuter(0), scale, opt);
  } else if (feature.compare("DifferenceOfGaussians") == 0) {
    differenceOfGaussians<N>(src, dest.bindOuter(0), scale, 0.66*scale, opt);
  } else {
    return 1;
  }
  return 0;
}


inline double string_to_double(std::string str) {
  std::istringstream is(str);
  double x;
  if (!(is >> x))
    throw std::runtime_error("Not convertable to double: " + str);
  return x;
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
