include_directories(modules/ruckig/include)
add_executable(Benchmarker Benchmarker.cpp)
target_link_libraries(Benchmarker
					  midi
					  ruckig
					  ${OpenCV_LIBS}
					  yaml-cpp
                      Boost::program_options)
include_directories(${dna_SOURCE_DIR}/include)