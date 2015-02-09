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
    find_package(HDF5 REQUIRED COMPONENTS C CXX HL)
    find_package(Boost REQUIRED COMPONENTS filesystem system python)
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

    SET(WITH_CHECKED_STL "True" CACHE BOOL "Build with checked libg++ for more robust code (may degrade performance).")
    IF(WITH_CHECKED_STL)
      ADD_DEFINITIONS(-D_GLIBCXX_DEBUG)
    ENDIF()

INCLUDE_DIRECTORIES(
  "${PROJECT_SOURCE_DIR}/include/"
)

FILE(GLOB LIB_SOURCES src/*.cpp)

INCLUDE_DIRECTORIES(include ${CPLEX_INCLUDE_DIRS})

#CPLEX switch to be compatible with STL
ADD_DEFINITIONS(-DIL_STD)

ADD_LIBRARY(pipeline_helpers STATIC src/pipeline_helpers.cxx src/segmentation.cxx src/lineage.cxx)

ADD_EXECUTABLE(isbi_pipeline main.cxx)
TARGET_LINK_LIBRARIES(isbi_pipeline pipeline_helpers gomp ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${CPLEX_LIBRARIES})

ADD_EXECUTABLE(h5_tracking h5_tracking.cxx)
TARGET_LINK_LIBRARIES(h5_tracking pipeline_helpers gomp ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES} ${HDF5_LIBRARIES} ${CPLEX_LIBRARIES})

# ADD_EXECUTABLE(isbi_pipeline3d main3D.cxx)
# TARGET_LINK_LIBRARIES(isbi_pipeline3d pipeline_helpers ${PGMLINK_LIBRARIES} ${VIGRA_IMPEX_LIBRARY} ${Boost_LIBRARIES})
