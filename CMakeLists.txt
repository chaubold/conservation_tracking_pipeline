CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(isbi_challenge)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_extensions/)


##
# dependencies
##
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

    SET(WITH_CHECKED_STL "True" CACHE BOOL "Build with checked libg++ for more robust code (may degrade performance).")
    IF(WITH_CHECKED_STL)
      ADD_DEFINITIONS(-D_GLIBCXX_DEBUG)
    ENDIF()

INCLUDE_DIRECTORIES(
  "${PROJECT_SOURCE_DIR}/include/" 
)

FILE(GLOB LIB_SOURCES src/*.cpp)

find_package( Boost REQUIRED COMPONENTS filesystem system)

INCLUDE_DIRECTORIES(include)

ADD_EXECUTABLE(isbi_pipeline main.cxx)
MESSAGE("${HDF5_HL_LIBRARIES}")
TARGET_LINK_LIBRARIES(isbi_pipeline ${Boost_LIBRARIES} ${HDF5_LIBRARIES} hdf5_hl vigraimpex)
