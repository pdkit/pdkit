cmake_minimum_required(VERSION 3.12)
project(libeemd_vendor LANGUAGES C)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Collect sources from flat or src/ layout
file(GLOB_RECURSE LIBEEMD_SOURCES
  "${CMAKE_CURRENT_LIST_DIR}/*.c"
  "${CMAKE_CURRENT_LIST_DIR}/src/*.c"
)
if(NOT LIBEEMD_SOURCES)
  message(FATAL_ERROR "No C sources found for libeemd in ${CMAKE_CURRENT_LIST_DIR}")
endif()

add_library(eemd ${LIBEEMD_SOURCES})

target_include_directories(eemd PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}/src
)

if (MSVC)
  target_compile_options(eemd PRIVATE /O2)
else()
  target_compile_options(eemd PRIVATE -O3)
endif()

set_target_properties(eemd PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# ---- GSL dependency ----
# Try pkg-config first (Homebrew/apt provide .pc files)
find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
  pkg_check_modules(GSL QUIET IMPORTED_TARGET gsl)
endif()

if (TARGET PkgConfig::GSL)
  target_link_libraries(eemd PRIVATE PkgConfig::GSL)
else()
  # Fallback manual find (Windows or when pkg-config missing)
  find_path(GSL_INCLUDE_DIR gsl/gsl_statistics_double.h
            HINTS ENV GSL_DIR
            PATH_SUFFIXES include)
  find_library(GSL_LIBRARY NAMES gsl libgsl
               HINTS ENV GSL_DIR
               PATH_SUFFIXES lib lib64)
  # gslcblas (or cblas/blas)
  find_library(GSL_CBLAS_LIBRARY NAMES gslcblas cblas blas
               HINTS ENV GSL_DIR
               PATH_SUFFIXES lib lib64)

  if (GSL_INCLUDE_DIR AND GSL_LIBRARY)
    target_include_directories(eemd PRIVATE ${GSL_INCLUDE_DIR})
    if (GSL_CBLAS_LIBRARY)
      target_link_libraries(eemd PRIVATE ${GSL_LIBRARY} ${GSL_CBLAS_LIBRARY})
    else()
      target_link_libraries(eemd PRIVATE ${GSL_LIBRARY})
    endif()
  else()
    message(FATAL_ERROR "GNU Scientific Library (GSL) not found. \
Install it (mac: 'brew install gsl'; Ubuntu: 'sudo apt-get install libgsl-dev').")
  endif()
endif()

# Link libm on Linux if needed
if (UNIX AND NOT APPLE)
  target_link_libraries(eemd PRIVATE m)
endif()
