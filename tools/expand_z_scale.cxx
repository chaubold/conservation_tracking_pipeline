// stl
#include <iostream>
#include <vector>
#include <string>

// boost
#include <boost/filesystem.hpp>

// vigra
#include <vigra/multi_array.hxx>
#include <vigra/impex.hxx>
#include <vigra/multi_resize.hxx>

// own
#include "pipeline_helpers.hxx"

namespace isbi = isbi_pipeline;
namespace fs = boost::filesystem;

typedef vigra::MultiArray<3, isbi::DataType> DataVolumeType;
typedef DataVolumeType::difference_type VolumeShapeType;

int main(int argc, char** argv) {
  if (argc < 4) {
    std::cout << "usage: " << argv[0];
    std::cout << " <input folder> <output folder> <z-scale>" << std::endl;
    return 1;
  }
  // get i/o paths
  fs::path i_dir = fs::system_complete(argv[1]);
  fs::path o_dir = fs::system_complete(argv[2]);
  double z_scale = boost::lexical_cast<double>(argv[3]);
  // print config
  std::cout << "scale z-axis with factor " << z_scale << std::endl;
  // get all files
  std::vector<fs::path> i_fn_vec = isbi::get_files(i_dir, ".tif");
  #pragma omp parallel for
  for(size_t i = 0; i < i_fn_vec.size(); i++) {
    const fs::path& i_fn = i_fn_vec[i];
    std::cout << "Processing " << i_fn.string() << std::endl;
    // load
    DataVolumeType volume;
    isbi::read_volume(volume, i_fn.string());
    // get new shape
    VolumeShapeType new_shape = volume.shape();
    new_shape[2] = new_shape[2] * z_scale - 1;
    DataVolumeType new_volume(new_shape);
    vigra::resizeMultiArraySplineInterpolation(volume, new_volume);
    // save
    std::string o_fn = o_dir.string() + "/" + i_fn.filename().string();
    std::cout << "Save as " << o_fn << std::endl;
    vigra::VolumeExportInfo export_info(o_fn.c_str());
    vigra::exportVolume(new_volume, export_info);
  }
  return 0;
}
