# This module finds an installed Vigra package.
#
# It sets the following variables:
#  PGMLINK_FOUND              - Set to false, or undefined, if lemon isn't found.
#  PGMLINK_INCLUDE_DIR        - Lemon include directory.
#  PGMLINK_LIBRARIES          - Lemon library files

# hacky:
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

FIND_PATH(PGMLINK_INCLUDE_DIR pgmlink/tracking.h PATHS /usr/include /usr/local/include ${CMAKE_INCLUDE_PATH} ${CMAKE_PREFIX_PATH}/include $ENV{PGMLINK_ROOT}/include ENV CPLUS_INCLUDE_PATH)
FIND_LIBRARY(PGMLINK_LIBRARIES pgmlink PATHS $ENV{PGMLINK_ROOT}/src/impex $ENV{PGMLINK_ROOT}/lib ENV LD_LIBRARY_PATH ENV LIBRARY_PATH)

GET_FILENAME_COMPONENT(PGMLINK_LIBRARY_PATH ${PGMLINK_LIBRARIES} PATH)
SET( PGMLINK_LIBRARY_DIR ${PGMLINK_LIBRARY_PATH} CACHE PATH "Path to pgmlink library.")

# handle the QUIETLY and REQUIRED arguments and set PGMLINK_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(PGMLINK DEFAULT_MSG PGMLINK_LIBRARIES PGMLINK_INCLUDE_DIR)
IF(PGMLINK_FOUND)
    IF (NOT PGMLINK_FIND_QUIETLY)
      MESSAGE(STATUS "  > pgmlink includes:    ${PGMLINK_INCLUDE_DIR}")
      MESSAGE(STATUS "  > pgmlink libraries:   ${PGMLINK_LIBRARIES}")
    ENDIF()
ENDIF()

MARK_AS_ADVANCED( PGMLINK_INCLUDE_DIR PGMLINK_LIBRARIRES)
