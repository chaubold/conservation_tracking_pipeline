CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(isbi_challenge)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_extensions/)


##
# dependencies
##

set(Boost_USE_STATIC_LIBS   ON)
set(HDF5_USE_STATIC_LIBRARIES ON)
find_package(Cplex REQUIRED)
find_package(VIGRA REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C HL)
find_package(Boost REQUIRED COMPONENTS filesystem system python serialization)
find_package(PythonLibs REQUIRED)
find_package(PGMLINK REQUIRED)

include_directories(SYSTEM ${VIGRA_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS} ${PYTHON_INCLUDE_DIR})

# build type and compiler options
##
IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release CACHE STRING
"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

SET(CMAKE_CXX_FLAGS "-Wall -pedantic -pthread ${CMAKE_CXX_FLAGS} -fopenmp")
SET(CMAKE_CXX_FLAGS_DEBUG "-Wno-long-long -g -O0 ${CMAKE_CXX_FLAGS_DEBUG}")
SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wno-long-long ${CMAKE_CXX_FLAGS_RELEASE}")
SET(CMAKE_EXE_LINKER_FLAGS "")

# check for C++ 11 support:
INCLUDE(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
IF(COMPILER_SUPPORTS_CXX11)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
ELSEIF(COMPILER_SUPPORTS_CXX0X)
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
ELSE()
    MESSAGE(SEND_ERROR "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
ENDIF()

INCLUDE_DIRECTORIES(
  "${PROJECT_SOURCE_DIR}/include/"
)

SET(ADDITIONAL_PGMLINK_INCLUDE_DIR "" CACHE PATH "Path to pgmlink/pgmlink_export.h")

INCLUDE_DIRECTORIES(include ${CPLEX_INCLUDE_DIRS} ${PGMLINK_INCLUDE_DIR} ${ADDITIONAL_PGMLINK_INCLUDE_DIR})

SET(WITH_CHECKED_STL "False" CACHE BOOL "Build with checked libg++ for more robust code (may degrade performance).")
IF(WITH_CHECKED_STL)
  ADD_DEFINITIONS(-D_GLIBCXX_DEBUG)
ENDIF()

SET(WITH_PARALLEL_FEATURES "True" CACHE BOOL "Switch to false if you need to restrict RAM usage")
IF(WITH_PARALLEL_FEATURES)
  ADD_DEFINITIONS(-DUSE_PARALLEL_FEATURES)
ENDIF()

SET(WITH_TOOLS "False" CACHE BOOL "Build auxiliary tools for pipeline.")

# CPLEX switch to be compatible with STL
ADD_DEFINITIONS(-DIL_STD)
# add flags for vigra, otherwise we would get name conflicts
ADD_DEFINITIONS(-DWITH_LEMON -DWITH_BOOST_GRAPH)

# opengm defines: do not include LPDef symbols everywhere
ADD_DEFINITIONS(-DOPENGM_LPDEF_NO_SYMBOLS)

SET(PIPELINE_HELPERS_SRC
  src/pipeline_helpers.cxx
  src/segmentation.cxx
  src/traxel_extractor.cxx
  src/lineage.cxx
  src/division_feature_extractor.cxx
  src/workflow.cxx)

ADD_LIBRARY(pipeline_helpers STATIC ${PIPELINE_HELPERS_SRC})

#ADD_LIBRARY(pipeline_helpers_32 STATIC ${PIPELINE_HELPERS_SRC})
#SET_TARGET_PROPERTIES(pipeline_helpers_32 PROPERTIES COMPILE_DEFINITIONS USE_32_BIT_LABELS)

ADD_EXECUTABLE(isbi_pipeline main.cxx)
TARGET_LINK_LIBRARIES(isbi_pipeline pipeline_helpers gomp ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${CPLEX_LIBRARIES})

ADD_EXECUTABLE(tracking tracking.cxx)
TARGET_LINK_LIBRARIES(tracking pipeline_helpers gomp ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${CPLEX_LIBRARIES})

ADD_EXECUTABLE(isbi_pipeline3d main3d.cxx)
TARGET_LINK_LIBRARIES(isbi_pipeline3d pipeline_helpers gomp ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${CPLEX_LIBRARIES})

ADD_EXECUTABLE(tracking3d tracking3d.cxx)
TARGET_LINK_LIBRARIES(tracking3d pipeline_helpers gomp ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${CPLEX_LIBRARIES})

#ADD_EXECUTABLE(isbi_pipeline3d_32 main3d.cxx)
#TARGET_LINK_LIBRARIES(isbi_pipeline3d_32 pipeline_helpers_32 gomp ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${CPLEX_LIBRARIES})
#SET_TARGET_PROPERTIES(isbi_pipeline3d_32 PROPERTIES COMPILE_DEFINITIONS USE_32_BIT_LABELS)

#ADD_EXECUTABLE(tracking3d_32 tracking3d.cxx)
#TARGET_LINK_LIBRARIES(tracking3d_32 pipeline_helpers_32 gomp ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${CPLEX_LIBRARIES})
#SET_TARGET_PROPERTIES(tracking3d_32 PROPERTIES COMPILE_DEFINITIONS USE_32_BIT_LABELS)

ADD_EXECUTABLE(test_feature_calculators test_feature_calculators.cxx)
TARGET_LINK_LIBRARIES(test_feature_calculators pipeline_helpers ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES})

ADD_EXECUTABLE(test_mask_traxels test_mask_traxels.cxx)
TARGET_LINK_LIBRARIES(test_mask_traxels pipeline_helpers ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES})

IF(WITH_TOOLS)
  ADD_EXECUTABLE(expand_z_scale tools/expand_z_scale.cxx)
  TARGET_LINK_LIBRARIES(expand_z_scale pipeline_helpers ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES})
ENDIF(WITH_TOOLS)
