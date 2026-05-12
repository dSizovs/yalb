add_executable(milestone04 main.cpp)
target_link_libraries(milestone04 PRIVATE lib)

configure_file(plot.py plot.py COPYONLY)
configure_file(fit_viscosity.py fit_viscosity.py COPYONLY)
