CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(isbi_challenge)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_extensions/)


##
# dependencies
##

    set(Boost_USE_STATIC_LIBS   ON)
    set(HDF5_USE_STATIC_LIBRARIES ON)       
    find_package(VIGRA REQUIRED)
    find_package(HDF5 REQUIRED)

    include_directories(${VIGRA_INCLUDE_DIR} ${HDF5_INCLUDE_DIRS})
    link_directories( ${VIGRA_IMPEX_LIBRARY_DIR} ${HDF5_LIBRARY_DIRS})


# build type and compiler options
##
    IF(NOT CMAKE_BUILD_TYPE)
      SET(CMAKE_BUILD_TYPE Release CACHE STRING
	"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
    ENDIF(NOT CMAKE_BUILD_TYPE)

    SET(CMAKE_CXX_FLAGS "-Wall -Wextra -Wfatal-errors -std=c++98 -pedantic -pthread")
    SET(CMAKE_CXX_FLAGS_DEBUG "-Wno-long-long -g -O0")
    SET(CMAKE_CXX_FLAGS_RELEASE "-O3 -Werror")
    SET(CMAKE_EXE_LINKER_FLAGS "-static -static-libgcc -static-libstdc++") 

    SET(WITH_CHECKED_STL "True" CACHE BOOL "Build with checked libg++ for more robust code (may degrade performance).")
    IF(WITH_CHECKED_STL)
      ADD_DEFINITIONS(-D_GLIBCXX_DEBUG)
    ENDIF()

INCLUDE_DIRECTORIES(
  "${PROJECT_SOURCE_DIR}/include/" 
)

FILE(GLOB LIB_SOURCES src/*.cpp)

find_package( Boost REQUIRED COMPONENTS filesystem system)
find_package( PGMLINK REQUIRED)

INCLUDE_DIRECTORIES(include)

#CPLEX switch to be compatible with STL
ADD_DEFINITIONS(-DIL_STD)  	    

ADD_LIBRARY(pipeline_helpers STATIC src/pipeline_helpers.cxx)
ADD_EXECUTABLE(isbi_pipeline main.cxx)
ADD_EXECUTABLE(isbi_pipeline3d main3D.cxx)
# TARGET_LINK_LIBRARIES(pipeline_helpers ${Boost_LIBRARIES} ${HDF5_LIBRARIES} hdf5_hl vigraimpex pgmlink)
SET_TARGET_PROPERTIES(isbi_pipeline PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1)
SET_TARGET_PROPERTIES(isbi_pipeline3d PROPERTIES LINK_SEARCH_START_STATIC 1 LINK_SEARCH_END_STATIC 1) 
TARGET_LINK_LIBRARIES(isbi_pipeline pipeline_helpers pgmlink z concert ilocplex cplex iculx icudata icutest icuio icuuc icule icutu icui18n ann boost_serialization-mt boost_filesystem-mt boost_system-mt hdf5_hl hdf5 vigraimpex jpeg tiff png pthread z m)
TARGET_LINK_LIBRARIES(isbi_pipeline3d pipeline_helpers pgmlink z concert ilocplex cplex iculx icudata icutest icuio icuuc icule icutu icui18n ann boost_serialization-mt boost_filesystem-mt boost_system-mt hdf5_hl hdf5 vigraimpex jpeg tiff png pthread z m)
#${Boost_LIBRARIES} ${HDF5_LIBRARIES} hdf5_hl vigraimpex pipeline_helpers pgmlink cplex ilocplex)
