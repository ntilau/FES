cmake_minimum_required(VERSION 3.16)
project(fes LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-register")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Include paths — project headers in include/, all deps in ../dep/include
set(DEP_DIR "${CMAKE_SOURCE_DIR}/../dep")
include_directories(
  include
  ${DEP_DIR}/include
)

# Defines
add_compile_definitions(TETLIBRARY TRILIBRARY)

# Sources — all .cpp in src/
file(GLOB SOURCES src/*.cpp)

add_executable(fes ${SOURCES})

# Link against dep/ libraries
target_link_libraries(fes PRIVATE
  ${DEP_DIR}/lib/libzmumps.a
  ${DEP_DIR}/lib/libcmumps.a
  ${DEP_DIR}/lib/libsmumps.a
  ${DEP_DIR}/lib/libdmumps.a
  ${DEP_DIR}/lib/libmumps_common.a
  ${DEP_DIR}/lib/libmpiseq.a
  ${DEP_DIR}/lib/libmpiseq_c.a
  ${DEP_DIR}/lib/libmpiseq_fortran.a
  ${DEP_DIR}/lib/libpord.a
  ${DEP_DIR}/lib/libtetgen.a
  ${DEP_DIR}/lib/libtriangle.a
  ${DEP_DIR}/lib/libarmadillo.a
  ${DEP_DIR}/lib/libarpack.a
  ${DEP_DIR}/lib/libopenblas.a
)
# Platform-specific gfortran linking
if(APPLE)
  target_link_options(fes PRIVATE
    -L/opt/homebrew/opt/gcc/lib/gcc/current
    -lgfortran
  )
elseif(UNIX)
  target_link_libraries(fes PRIVATE gfortran)
endif()
