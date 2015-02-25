#include <iostream>
#include <set>

// vigra
#include <vigra/multi_math.hxx>
#include <vigra/impex.hxx>
#include <vigra/multi_array.hxx>

template<int N, class DataType>
std::set<DataType> find_unique_labels_in_roi(vigra::MultiArrayView<N, DataType> roi, bool ignore_label_zero = true)
{
  typedef std::set<DataType> SetType;
  SetType labels(roi.begin(), roi.end());

  if(ignore_label_zero)
  {
    typename SetType::iterator it = std::find(labels.begin(), labels.end(), 0);
    if(it != labels.end())
      labels.erase(it);
  }

  return std::move(labels);
}


int main(int argc, char** argv) {
  if(argc != 3)
  {
    std::cout << "Wrong number of arguments, 3 expected" << std::endl;
    std::cout << "USAGE: " << argv[0] << " <segmentation> <mask>" << std::endl;
    return -1;
  }

  // load segmentation image
  std::string segmentation_filename(argv[1]);
  vigra::ImageImportInfo info(segmentation_filename.c_str());
  vigra::Shape2 shape(info.width(), info.height());
  vigra::MultiArray<2, vigra::UInt8> segmentation(shape);
  vigra::importImage(info, segmentation);

  std::set<vigra::UInt8> labels_in_segmentation = find_unique_labels_in_roi<2, vigra::UInt8>(segmentation.view());
  std::cout << labels_in_segmentation.size() << " labels within segmentation: (";
  for(auto a : labels_in_segmentation)
    std::cout << (int)a << ", ";
  std::cout << ")" << std::endl;

  // load mask image
  std::string mask_image_filename(argv[2]);
  vigra::ImageImportInfo mask_info(mask_image_filename.c_str());
  vigra::Shape2 mask_shape(mask_info.width(), mask_info.height());
  vigra::MultiArray<2, vigra::UInt8> mask_image(mask_shape);
  vigra::importImage(mask_info, mask_image);
  std::cout << "Image has shape: " << shape << std::endl;

  {
    using namespace vigra::multi_math;

    // make mask image binary, background has label 0!
    mask_image = signi(mask_image);

    // extract part of segmentation that has been masked
    segmentation = segmentation * mask_image;
  }

  // extract labels inside segmentation
  std::set<vigra::UInt8> labels_in_mask = find_unique_labels_in_roi<2, vigra::UInt8>(segmentation.view());
  std::cout << labels_in_mask.size() << " labels within mask: (";
  for(auto a : labels_in_mask)
    std::cout << (int)a << ", ";
  std::cout << ")" << std::endl;
}
