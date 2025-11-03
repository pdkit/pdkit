cmake_minimum_required(VERSION 3.23)
project(libclose_ret LANGUAGES C)

set(BUILD_SHARED_LIBS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Build the close_ret library
add_library(close_ret close_ret.c)

# Standard math library (needed for some systems)
if(UNIX AND NOT APPLE)
    target_link_libraries(close_ret PRIVATE m)
endif()

# Set optimization flags
if (MSVC)
  target_compile_options(close_ret PRIVATE /O2)
else()
  target_compile_options(close_ret PRIVATE -O3)
endif()

# Export symbols on Windows
set_target_properties(close_ret PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)

# Make sure the library is named consistently across platforms
set_target_properties(close_ret PROPERTIES
    OUTPUT_NAME "close_ret"
    PREFIX "lib"
)

# Copy library to source tree after build for editable installs
# This allows the Python hook to find it in development mode
add_custom_command(TARGET close_ret POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy 
            $<TARGET_FILE:close_ret>
            ${CMAKE_CURRENT_SOURCE_DIR}/
    COMMENT "Copying libclose_ret to source tree for editable installs"
)
