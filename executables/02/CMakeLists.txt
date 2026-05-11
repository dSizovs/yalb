add_executable(milestone02 main.cpp)
target_link_libraries(milestone02 PRIVATE lib)

configure_file(plot.py plot.py COPYONLY)
