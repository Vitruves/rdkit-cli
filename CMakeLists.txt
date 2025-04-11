cmake_minimum_required(VERSION 3.12)
project(rdkit-cli)
cmake_policy(SET CMP0167 OLD)

# Enable CTest
include(CTest)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_SCOPED_ENUMS -DRD_USING_BOOST_SHARED_PTR_FOR_STD=0")

# Set output directory for executables
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Define options
option(BUILD_TESTING "Build the testing tree" ON)

# Find dependencies
find_package(RDKit REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options filesystem system)

# Handle OpenMP support
if(APPLE)
    # Check if libomp is installed via homebrew
    if(EXISTS "/opt/homebrew/opt/libomp")
        # On macOS with Apple Clang, we need to set the OpenMP flags manually
        set(HAVE_OPENMP TRUE)
        set(OpenMP_CXX_FLAGS "-Xpreprocessor;-fopenmp")
        set(OpenMP_CXX_INCLUDE_DIRS "/opt/homebrew/opt/libomp/include") 
        set(OpenMP_CXX_LIBRARIES "/opt/homebrew/opt/libomp/lib/libomp.dylib")
    else()
        message(STATUS "OpenMP not found - parallel execution will be disabled")
        set(HAVE_OPENMP FALSE)
    endif()
else()
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        set(HAVE_OPENMP TRUE)
    else()
        message(STATUS "OpenMP not found - parallel execution will be disabled")
        set(HAVE_OPENMP FALSE)
    endif()
endif()

# Handle RDKit headers
# Treat RDKit includes as system includes to avoid multiple include problems
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -isystem ${RDKit_INCLUDE_DIRS}")
endif()

# Include directories
include_directories(${RDKit_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})
include_directories(include)

# Add Boost include explicitly - resolves missing boost header issue
if(APPLE)
    # Add direct include path to the specific Boost installation found
    if(EXISTS "/opt/homebrew/Cellar/boost/1.87.0_1/include")
        include_directories(SYSTEM "/opt/homebrew/Cellar/boost/1.87.0_1/include")
        message(STATUS "Adding Boost include: /opt/homebrew/Cellar/boost/1.87.0_1/include")
    endif()
    
    # Also try other common locations as a fallback
    set(POSSIBLE_BOOST_PATHS
        "/opt/homebrew/Cellar/boost/${Boost_VERSION}"
        "/opt/homebrew/Cellar/boost/${Boost_VERSION}_1"
        "/opt/homebrew/Cellar/boost/include"
        "/opt/homebrew/include/boost"
        "/usr/local/include/boost"
    )
    
    foreach(BOOST_PATH ${POSSIBLE_BOOST_PATHS})
        if(EXISTS "${BOOST_PATH}")
            include_directories(SYSTEM ${BOOST_PATH})
            message(STATUS "Adding additional Boost include: ${BOOST_PATH}")
            break()
        endif()
    endforeach()
endif()

# Define RDKit component libraries needed
set(RDKIT_LIBS
    RDKitGraphMol
    RDKitSmilesParse
    RDKitFileParsers
    RDKitSubstructMatch
    RDKitDescriptors
    RDKitForceFieldHelpers
    RDKitDistGeomHelpers
    RDKitMolAlign
    RDKitO3AAlign
    RDKitOptimizer
    RDKitMolTransforms
    RDKitDepictor
    RDKitMolDraw2D
    RDKitRDGeometryLib
    RDKitMolStandardize
    RDKitChemTransforms
    RDKitFingerprints
    RDKitDataStructs
    RDKitMolHash
    RDKitCatalogs
    RDKitForceField
    RDKitMolInterchange
    RDKitPartialCharges
    RDKitRDGeneral
    RDKitRDInchiLib
    RDKitAlignment
)

# Print configuration summary
message(STATUS "RDKit CLI configured with:")
message(STATUS "  RDKit includes: ${RDKit_INCLUDE_DIRS}")
message(STATUS "  Boost includes: ${Boost_INCLUDE_DIRS}")
message(STATUS "  Source files: src/")
message(STATUS "  Include files: include/")
message(STATUS "  Output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
if(HAVE_OPENMP)
    message(STATUS "  OpenMP: Enabled")
else()
    message(STATUS "  OpenMP: Disabled")
endif()
message(STATUS "  Tests: ${BUILD_TESTING}")

# Add tests first if enabled, so the library gets built
if(BUILD_TESTING)
    # Check if test directory exists AND contains a CMakeLists.txt file
    if(EXISTS "${CMAKE_SOURCE_DIR}/test/CMakeLists.txt")
        enable_testing()
        add_subdirectory(test)
        message(STATUS "Tests: Enabled")
        # Add custom targets for running tests
        add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure)
        add_custom_target(test-all COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure || echo "Some tests failed, but continuing")
    else()
        message(STATUS "Tests directory doesn't contain CMakeLists.txt, disabling tests")
        set(BUILD_TESTING OFF)
    endif()
else()
    message(STATUS "Tests: Disabled")
endif()

# Define source files
set(SOURCE_FILES
    src/main.cpp
)

# Define executable - Link against the library if testing is enabled, otherwise build from source
add_executable(rdkit-cli ${SOURCE_FILES})

if(BUILD_TESTING AND TARGET rdkit_cli_lib)
    target_link_libraries(rdkit-cli
        rdkit_cli_lib
        ${RDKIT_LIBS}
        ${Boost_LIBRARIES}
    )
else()
    # If tests are not enabled or the library doesn't exist, build from source
    set(MAIN_SOURCES
        src/data.cpp
        src/descriptors.cpp
        src/filters.cpp
        src/fingerprints.cpp
        src/smiles.cpp
        src/conformers.cpp
        src/visualization.cpp
        src/progress.cpp
    )
    target_sources(rdkit-cli PRIVATE ${MAIN_SOURCES})
    
    target_link_libraries(rdkit-cli
        ${RDKIT_LIBS}
        ${Boost_LIBRARIES}
    )
endif()

# Add OpenMP if available
if(HAVE_OPENMP)
    target_compile_options(rdkit-cli PRIVATE ${OpenMP_CXX_FLAGS})
    target_include_directories(rdkit-cli PRIVATE ${OpenMP_CXX_INCLUDE_DIRS})
    target_link_libraries(rdkit-cli ${OpenMP_CXX_LIBRARIES})
else()
    # Define NO_OPENMP to disable OpenMP code in source
    target_compile_definitions(rdkit-cli PRIVATE -DNO_OPENMP)
endif()

# Platform-specific optimizations
if(APPLE)
    target_compile_options(rdkit-cli PRIVATE -faligned-allocation)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
        target_compile_options(rdkit-cli PRIVATE -mcpu=apple-m1)
        message(STATUS "Optimizing for Apple M-series chips")
    endif()
endif()

# Installation
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
install(TARGETS rdkit-cli DESTINATION ${CMAKE_SOURCE_DIR}/bin)