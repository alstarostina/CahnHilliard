cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

project("CHProject")

# find deal.ii package
find_package(deal.II 9.1.0 REQUIRED
        HINTS ${DEAL_II_DIR} ../../dealii/install $ENV{DEAL_II_DIR})
deal_ii_initialize_cached_variables()

# add our executable
add_executable(CHnew src/CH.cpp)
deal_ii_setup_target(CHnew)


# create output folder
add_custom_command(
        TARGET CHnew POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory $<TARGET_FILE_DIR:CHnew>/output)