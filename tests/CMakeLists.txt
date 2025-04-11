set(TEST_SRC_FILES
    test_smiles.cpp
    test_descriptors.cpp
    test_fingerprints.cpp
    test_conformers.cpp
    test_filters.cpp
    test_data.cpp
)

# Common source files needed by all tests
set(COMMON_SRC_FILES
    ../src/data.cpp
    ../src/smiles.cpp
    ../src/descriptors.cpp
    ../src/fingerprints.cpp
    ../src/conformers.cpp
    ../src/filters.cpp
    ../src/visualization.cpp
)

# Create a library with all the common source files
add_library(rdkit_cli_lib STATIC ${COMMON_SRC_FILES})
target_link_libraries(rdkit_cli_lib 
    ${RDKIT_LIBS}
    ${Boost_LIBRARIES}
)

# Ensure the library can see the include directory
target_include_directories(rdkit_cli_lib PRIVATE "../include")

if(HAVE_OPENMP)
    target_compile_options(rdkit_cli_lib PRIVATE ${OpenMP_CXX_FLAGS})
    target_include_directories(rdkit_cli_lib PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(rdkit_cli_lib ${OpenMP_CXX_LIBRARIES})
else()
    target_compile_definitions(rdkit_cli_lib PRIVATE -DNO_OPENMP)
endif()

foreach(test_src ${TEST_SRC_FILES})
    get_filename_component(test_name ${test_src} NAME_WE)
    add_executable(${test_name} ${test_src})
    target_link_libraries(${test_name} 
        rdkit_cli_lib
        ${RDKIT_LIBS}
        ${Boost_LIBRARIES}
    )
    
    # Ensure tests can see the include directory
    target_include_directories(${test_name} PRIVATE "../include")
    
    if(HAVE_OPENMP)
        target_compile_options(${test_name} PRIVATE ${OpenMP_CXX_FLAGS})
        target_include_directories(${test_name} PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
        target_link_libraries(${test_name} ${OpenMP_CXX_LIBRARIES})
    else()
        target_compile_definitions(${test_name} PRIVATE -DNO_OPENMP)
    endif()
    
    add_test(NAME ${test_name} COMMAND ${test_name})
endforeach() 