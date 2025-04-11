#include <iostream>
#include <fstream>
#include <thread>

// Include our local headers
#include "../include/visualization.h"
#include "../include/progress.h"

// OpenMP support if available
#ifndef NO_OPENMP
#include <omp.h>
#endif

void VisualizationOptions::addOptions(po::options_description& desc) {
    desc.add_options()
        ("highlight-substructure", po::value<std::vector<std::string>>()->multitoken(), "Highlight substructure (smarts output_dir)")
        ("export-svg", po::value<std::vector<std::string>>()->multitoken(), "Export as SVG (output_dir width height)")
        ("export-png", po::value<std::vector<std::string>>()->multitoken(), "Export as PNG (output_dir width height)");
}

bool VisualizationHandler::shouldProcess(const po::variables_map& vm) {
    return vm.count("highlight-substructure") ||
           vm.count("export-svg") ||
           vm.count("export-png");
}

void VisualizationHandler::process(MoleculeDataset& dataset, const po::variables_map& vm) {
    std::cout << "-- Processing visualization operations" << std::endl;
    
    // Determine number of workers to use
    int numWorkers = std::thread::hardware_concurrency() - 2;
    numWorkers = std::max(1, numWorkers);
    
    if (vm.count("mpu")) {
        numWorkers = vm["mpu"].as<int>();
    } else if (vm.count("workers")) {
        numWorkers = vm["workers"].as<int>();
    } else if (vm.count("parallels")) {
        numWorkers = vm["parallels"].as<int>();
    } else if (vm.count("multiprocessing")) {
        numWorkers = vm["multiprocessing"].as<int>();
    }
    
    // Set OpenMP threads if available
#ifndef NO_OPENMP
    omp_set_num_threads(numWorkers);
#endif
    
    if (vm.count("highlight-substructure")) {
        auto args = vm["highlight-substructure"].as<std::vector<std::string>>();
        if (args.size() >= 2) {
            std::cout << "-- Highlighting substructure using " << numWorkers << " worker threads" << std::endl;
            highlightSubstructure(dataset, args[0], args[1]);
            std::cout << "-- Substructure highlighting - done" << std::endl;
        } else {
            std::cerr << "-- ERROR: highlight-substructure requires SMARTS and output_dir arguments" << std::endl;
        }
    }
    
    if (vm.count("export-svg")) {
        auto args = vm["export-svg"].as<std::vector<std::string>>();
        if (args.size() >= 3) {
            std::cout << "-- Exporting SVGs using " << numWorkers << " worker threads" << std::endl;
            exportSVG(dataset, args[0], std::stoi(args[1]), std::stoi(args[2]));
            std::cout << "-- SVG export - done" << std::endl;
        } else {
            std::cerr << "-- ERROR: export-svg requires output_dir, width, and height arguments" << std::endl;
        }
    }
    
    if (vm.count("export-png")) {
        auto args = vm["export-png"].as<std::vector<std::string>>();
        if (args.size() >= 3) {
            std::cout << "-- Exporting PNGs using " << numWorkers << " worker threads" << std::endl;
            exportPNG(dataset, args[0], std::stoi(args[1]), std::stoi(args[2]));
            std::cout << "-- PNG export - done" << std::endl;
        } else {
            std::cerr << "-- ERROR: export-png requires output_dir, width, and height arguments" << std::endl;
        }
    }
}

void VisualizationHandler::highlightSubstructure(MoleculeDataset& dataset, const std::string& smarts, const std::string& outputDir) {
    fs::create_directories(outputDir);
    
    RDKit::ROMol* pattern = nullptr;
    try {
        pattern = RDKit::SmartsToMol(smarts);
        if (!pattern) {
            std::cerr << "-- ERROR: Invalid SMARTS pattern: " << smarts << std::endl;
            return;
        }
    } catch (const std::exception& e) {
        std::cerr << "-- ERROR: Failed to parse SMARTS pattern: " << e.what() << std::endl;
        return;
    }
    
    // Count valid molecules that match the pattern
    std::vector<size_t> matchingIndices;
    for (size_t i = 0; i < dataset.size(); i++) {
        if (dataset[i].mol) {
            std::vector<RDKit::MatchVectType> matches;
            RDKit::SubstructMatch(*dataset[i].mol, *pattern, matches);
            if (!matches.empty()) {
                matchingIndices.push_back(i);
            }
        }
    }
    
    std::string operationName = "Highlighting substructure matches";
    ProgressTracker progress(operationName, matchingIndices.size());

    parallelProcessWithProgress(operationName, matchingIndices.size(), omp_get_max_threads(), false,
        [&](size_t idx) {
            size_t i = matchingIndices[idx];
            std::vector<RDKit::MatchVectType> matches;
            RDKit::SubstructMatch(*dataset[i].mol, *pattern, matches);
            
            std::string molName = "molecule_" + std::to_string(i);
            if (dataset[i].properties.find("Name") != dataset[i].properties.end()) {
                molName = dataset[i].properties["Name"];
            } else if (dataset[i].properties.find("ID") != dataset[i].properties.end()) {
                molName = dataset[i].properties["ID"];
            }
            
            std::string filename = outputDir + "/" + molName + ".svg";
            
            std::vector<int> highlightAtoms;
            for (const auto& match : matches) {
                for (const auto& pair : match) {
                    highlightAtoms.push_back(pair.second);
                }
            }
            
            RDKit::MolDraw2DSVG drawer(300, 300);
            RDKit::MolDraw2DUtils::prepareAndDrawMolecule(drawer, *dataset[i].mol, "", &highlightAtoms);
            drawer.finishDrawing();
            
            #pragma omp critical
            {
                std::ofstream outFile(filename);
                outFile << drawer.getDrawingText();
                outFile.close();
            }
        }
    );
    
    delete pattern;
}

void VisualizationHandler::exportSVG(MoleculeDataset& dataset, const std::string& outputDir, int width, int height) {
    fs::create_directories(outputDir);
    
    std::string operationName = "Exporting SVG files";
    
    // Filter out invalid molecules first
    std::vector<size_t> validIndices;
    for (size_t i = 0; i < dataset.size(); i++) {
        if (dataset[i].mol) {
            validIndices.push_back(i);
        }
    }
    
    parallelProcessWithProgress(operationName, validIndices.size(), omp_get_max_threads(), false,
        [&](size_t idx) {
            size_t i = validIndices[idx];
            
            std::string molName = "molecule_" + std::to_string(i);
            if (dataset[i].properties.find("Name") != dataset[i].properties.end()) {
                molName = dataset[i].properties["Name"];
            } else if (dataset[i].properties.find("ID") != dataset[i].properties.end()) {
                molName = dataset[i].properties["ID"];
            }
            
            std::string filename = outputDir + "/" + molName + ".svg";
            
            RDKit::MolDraw2DSVG drawer(width, height);
            RDKit::MolDraw2DUtils::prepareAndDrawMolecule(drawer, *dataset[i].mol);
            drawer.finishDrawing();
            
            #pragma omp critical
            {
                std::ofstream outFile(filename);
                outFile << drawer.getDrawingText();
                outFile.close();
            }
        }
    );
}

void VisualizationHandler::exportPNG(MoleculeDataset& dataset, const std::string& outputDir, int width, int height) {
    fs::create_directories(outputDir);
    
    std::cerr << "-- WARNING: PNG export is not currently supported in this build. Using SVG instead." << std::endl;
    exportSVG(dataset, outputDir, width, height);
}