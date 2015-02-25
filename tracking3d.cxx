// stl
#include <stdexcept>
#include <iostream>

// own
#include "workflow.hxx"

// Aliases for convenience
namespace isbi = isbi_pipeline;

int main(int argc, char** argv) {
  try {
    isbi::Workflow workflow(false);
    workflow.init(argc, argv);
    workflow.run<3>();
    return 0;
  } catch (isbi::ArgumentError& e) {
    std::cout << e.what() << std::endl;
    std::cout << "usage:" << std::endl;
    std::cout << argv[0] << " <image folder> <segmentation folder>"
      << " <result_folder> <config file> <classifier file>"
      << " <region feature file> <division feature file>"
      << std::endl;
    return 0;
  } catch (std::runtime_error& e) {
    std::cout << "Program crashed:\n";
    std::cout << e.what();
    std::cout << std::endl;
    return 0;
  }
}
