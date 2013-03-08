#ifndef PIPELINE_HELPERS_HXX
#define PIPELINE_HELPERS_HXX

#include <vigra/multi_array.hxx>



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

#endif /* PIPELINE_HELPERS_HXX */
