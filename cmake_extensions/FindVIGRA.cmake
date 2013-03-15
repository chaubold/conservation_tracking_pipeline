# This module finds an installed Vigra package.
#
# It sets the following variables:
#  VIGRA_FOUND              - Set to false, or undefined, if vigra isn't found.
#  VIGRA_INCLUDE_DIR        - Vigra include directory.
#  VIGRA_IMPEX_LIBRARY      - Vigra's impex library
#  VIGRA_IMPEX_LIBRARY_DIR  - path to Vigra impex library
#  VIGRA_NUMPY_CORE_LIBRARY - Vigra's vigranumpycore library

# hacky:
# set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")

# configVersion.hxx only present, after build of Vigra
FIND_PATH(VIGRA_INCLUDE_DIR vigra/configVersion.hxx PATHS $ENV{VIGRA_ROOT}/include ENV CPLUS_INCLUDE_PATH)
FIND_LIBRARY(VIGRA_IMPEX_LIBRARY vigraimpex PATHS $ENV{VIGRA_ROOT}/src/impex $ENV{VIGRA_ROOT}/lib ENV LD_LIBRARY_PATH ENV LIBRARY_PATH)
GET_FILENAME_COMPONENT(VIGRA_IMPEX_LIBRARY_PATH ${VIGRA_IMPEX_LIBRARY} PATH)
SET( VIGRA_IMPEX_LIBRARY_DIR ${VIGRA_IMPEX_LIBRARY_PATH} CACHE PATH "Path to Vigra impex library.")

# EXECUTE_PROCESS ( COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" OUTPUT_VARIABLE PYTHON_SITE_PACKAGES OUTPUT_STRIP_TRAILING_WHITESPACE)
# FIND_FILE(VIGRA_NUMPY_CORE_LIBRARY vigranumpycore.so PATHS ${PYTHON_SITE_PACKAGES} ENV PYTHONPATH PATH_SUFFIXES vigra)


# handle the QUIETLY and REQUIRED arguments and set VIGRA_FOUND to TRUE if 
# all listed variables are TRUE
INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(VIGRA DEFAULT_MSG VIGRA_IMPEX_LIBRARY VIGRA_INCLUDE_DIR)
IF(VIGRA_FOUND)
    IF (NOT Vigra_FIND_QUIETLY)
      MESSAGE(STATUS "  > vigra includes:      ${VIGRA_INCLUDE_DIR}")
      MESSAGE(STATUS "  > impex library:       ${VIGRA_IMPEX_LIBRARY}")
    ENDIF()
ENDIF()


MARK_AS_ADVANCED( VIGRA_INCLUDE_DIR VIGRA_IMPEX_LIBRARY VIGRA_IMPEX_LIBRARY_DIR)
