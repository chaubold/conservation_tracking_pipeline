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

// pgmlink
#include <pgmlink/tracking.h>
#include <pgmlink/traxels.h>

namespace isbi_pipeline {

typedef float FeatureType;
typedef std::vector<pgmlink::Event> EventVectorType;
typedef std::vector<EventVectorType> EventVectorVectorType;
typedef std::vector<std::pair<std::string, double> > StringDoublePairVectorType;

// ArgumentError class
class ArgumentError : public std::exception {
public:
  virtual const char* what() const throw();
};

class TrackingOptions {
 public:
  TrackingOptions(const std::string path);
  bool is_legal() const;
  template<typename T> bool has_option(const std::string key) const;
  template<typename T> bool check_option(const std::string key) const;
  template<typename T> T get_option(const std::string key) const;
 private:
  std::map<std::string, std::string> options_map_;
};


// strip trailing slash
// works only for null terminated cstrings
void rstrip(char* c, char r);


// zero padding int to string
std::string zero_padding(int num, int n_zeros);


// read features from csv_style file
int read_features_from_file(
  const std::string path,
  StringDoublePairVectorType& features);

// read region feature list from file
int read_region_features_from_file(
  const std::string path,
  std::vector<std::string>& feature_list);

// read tif image
int read_tif_image(
  const std::string path,
  vigra::MultiArray<2, unsigned>& image);

// save tif image
int save_tif_image(
  const std::string path,
  const vigra::MultiArray<2, unsigned>& image);

// get random_forests from file
bool get_rfs_from_file(std::vector<vigra::RandomForest<unsigned> >& rfs,
                       std::string fn,
                       std::string path_in_file = "PixelClassification/ClassifierForests/Forest",
                       int n_forests = 10,
                       int n_leading_zeros = 4);


// do the tracking
EventVectorVectorType track(pgmlink::TraxelStore& ts, const TrackingOptions& options);


// helper function to iterate over tif only
bool contains_substring(std::string str, std::string substr);


// same for boost_path
bool contains_substring_boost_path(const boost::filesystem::directory_entry& p, std::string substr);


// copy_if_own because not using c++11
template <class InputIterator, class OutputIterator, class UnaryPredicate>
OutputIterator copy_if_own (InputIterator first, InputIterator last, OutputIterator result, UnaryPredicate pred);


/* -------------------------------------------------- */
/*                   IMPLEMENTATION                   */
/* -------------------------------------------------- */


template <class InputIterator, class OutputIterator, class UnaryPredicate>
OutputIterator copy_if_own (InputIterator first, InputIterator last, OutputIterator result, UnaryPredicate pred)
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

} // namespace isbi_pipeline

#endif /* PIPELINE_HELPERS_HXX */
