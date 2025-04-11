cmake_minimum_required(VERSION 3.12)
project(rdkit-cli)
cmake_policy(SET CMP0167 OLD)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DBOOST_NO_CXX11_SCOPED_ENUMS -DRD_USING_BOOST_SHARED_PTR_FOR_STD=0")

# Define option for testing
option(BUILD_TESTING "Build the testing tree" ON)

# Find dependencies
find_package(RDKit REQUIRED)
find_package(Boost REQUIRED COMPONENTS program_options filesystem)

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

# Define source files
set(SOURCE_FILES
    src/main.cpp
    src/data.cpp
    src/descriptors.cpp
    src/fingerprints.cpp
    src/smiles.cpp
    src/conformers.cpp
    src/filters.cpp
    src/visualization.cpp
)

# Define executable
add_executable(rdkit-cli ${SOURCE_FILES})

# Link libraries
target_link_libraries(rdkit-cli
    ${RDKIT_LIBS}
    ${Boost_LIBRARIES}
)

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
install(TARGETS rdkit-cli DESTINATION "$ENV{HOME}/.local/bin")

# Add tests if enabled
if(BUILD_TESTING)
    enable_testing()
    add_subdirectory(tests)
endif()

# Print configuration summary
message(STATUS "RDKit CLI configured with:")
message(STATUS "  RDKit includes: ${RDKit_INCLUDE_DIRS}")
message(STATUS "  Boost includes: ${Boost_INCLUDE_DIRS}")
message(STATUS "  Source files: src/")
message(STATUS "  Include files: include/")
if(HAVE_OPENMP)
    message(STATUS "  OpenMP: Enabled")
else()
    message(STATUS "  OpenMP: Disabled")
endif()
message(STATUS "  Tests: ${BUILD_TESTING}")