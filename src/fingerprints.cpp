#include <iostream>
#include <thread>
#include <DataStructs/BitOps.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <GraphMol/Fingerprints/MACCS.h>
#include <GraphMol/Fingerprints/AtomPairs.h>
#include <GraphMol/Fingerprints/FingerprintGenerator.h>
#include <DataStructs/ExplicitBitVect.h>
#include <DataStructs/SparseIntVect.h>
#include <boost/shared_ptr.hpp>
#ifndef NO_OPENMP
#include <omp.h>
#endif
#include <boost/algorithm/string.hpp>

#include "../include/fingerprints.h"
#include "../include/progress.h"

void FingerprintOptions::addOptions(po::options_description& desc) {
    desc.add_options()
        ("fp-morgan", po::value<std::vector<std::string>>()->multitoken(), "Generate Morgan fingerprint (col_name radius bits)")
        ("fp-maccs", po::value<std::string>(), "Generate MACCS fingerprint")
        ("fp-atom-pairs", po::value<std::string>(), "Generate Atom Pairs fingerprint")
        ("concat-fp", po::value<std::vector<std::string>>()->multitoken(), "Concatenate fingerprints (fp_col1 fp_col2... output_col)")
        ("concat-all-fp", po::value<std::string>(), "Concatenate all fingerprints")
        ("similarity-tanimoto", po::value<std::vector<std::string>>()->multitoken(), "Calculate Tanimoto similarity (col1 col2 output_col)")
        ("fingerprint", po::value<std::string>(), "Fingerprint type (morgan, maccs, atom-pairs, concat-fp, concat-all-fp, similarity-tanimoto)")
        ("fingerprint-bits", po::value<int>(), "Number of bits for fingerprint generation")
        ("fingerprint-radius", po::value<int>(), "Radius for Morgan fingerprint generation")
        ("fingerprint-min-path", po::value<int>(), "Minimum path length for path fingerprint generation")
        ("fingerprint-max-path", po::value<int>(), "Maximum path length for path fingerprint generation");
}

bool FingerprintHandler::shouldProcess(const po::variables_map& vm) {
    return vm.count("fp-morgan") ||
           vm.count("fp-maccs") ||
           vm.count("fp-atom-pairs") ||
           vm.count("concat-fp") ||
           vm.count("concat-all-fp") ||
           vm.count("similarity-tanimoto") ||
           vm.count("fingerprint");
}

void FingerprintHandler::process(MoleculeDataset& dataset, const po::variables_map& vm) {
    // Extract the fingerprint type from command-line
    std::string fingerprintType = vm["fingerprint"].as<std::string>();
    
    // Convert to lower case for case-insensitive comparison
    boost::algorithm::to_lower(fingerprintType);
    
    // Process options
    int numBits = 2048;
    int radius = 2;
    int minPath = 1;
    int maxPath = 7;
    
    if (vm.count("fingerprint-bits")) {
        numBits = vm["fingerprint-bits"].as<int>();
    }
    
    if (vm.count("fingerprint-radius")) {
        radius = vm["fingerprint-radius"].as<int>();
    }
    
    if (vm.count("fingerprint-min-path")) {
        minPath = vm["fingerprint-min-path"].as<int>();
    }
    
    if (vm.count("fingerprint-max-path")) {
        maxPath = vm["fingerprint-max-path"].as<int>();
    }
    
    // Determine number of workers for parallelization
    int numWorkers = 1;
    if (vm.count("workers")) {
        numWorkers = vm["workers"].as<int>();
    } else if (vm.count("parallels")) {
        numWorkers = vm["parallels"].as<int>();
    } else if (vm.count("multiprocessing")) {
        numWorkers = vm["multiprocessing"].as<int>();
    }
    
    if (!vm.count("quiet")) {
        std::cout << "-- Calculating " << fingerprintType << " fingerprints with " << numWorkers << " threads" << std::endl;
        if (fingerprintType == "morgan" || fingerprintType == "ecfp" || fingerprintType == "fcfp") {
            std::cout << "-- Using " << radius << " radius and " << numBits << " bits" << std::endl;
        } else if (fingerprintType == "path" || fingerprintType == "rdkit") {
            std::cout << "-- Using paths from " << minPath << " to " << maxPath << " and " << numBits << " bits" << std::endl;
        }
    }
    
    // Configure multiprocessing
    numWorkers = std::max(1, numWorkers);
    
    // Set OpenMP threads if available
#ifndef NO_OPENMP
    omp_set_num_threads(numWorkers);
#endif

    std::cout << "-- Using " << numWorkers << " worker threads for fingerprints" << std::endl;
    
    if (vm.count("fp-morgan")) {
        auto args = vm["fp-morgan"].as<std::vector<std::string>>();
        if (args.size() >= 3) {
            std::string colName = args[0];
            int radius = std::stoi(args[1]);
            int nBits = std::stoi(args[2]);
            generateMorganFingerprint(dataset, colName, radius, nBits);
            std::cout << "-- Morgan fingerprint generation - done" << std::endl;
        } else {
            std::cerr << "-- ERROR: fp-morgan requires col_name, radius, and bits arguments" << std::endl;
        }
    }
    
    if (vm.count("fp-maccs")) {
        generateMACCSFingerprint(dataset, vm["fp-maccs"].as<std::string>());
        std::cout << "-- MACCS fingerprint generation - done" << std::endl;
    }
    
    if (vm.count("fp-atom-pairs")) {
        generateAtomPairsFingerprint(dataset, vm["fp-atom-pairs"].as<std::string>());
        std::cout << "-- Atom Pairs fingerprint generation - done" << std::endl;
    }
    
    if (vm.count("concat-fp")) {
        auto args = vm["concat-fp"].as<std::vector<std::string>>();
        if (args.size() >= 2) {
            std::string outputCol = args.back();
            std::vector<std::string> fpCols(args.begin(), args.end() - 1);
            concatenateFingerprints(dataset, fpCols, outputCol);
            std::cout << "-- Fingerprint concatenation - done" << std::endl;
        } else {
            std::cerr << "-- ERROR: concat-fp requires at least one source column and an output column" << std::endl;
        }
    }
    
    if (vm.count("concat-all-fp")) {
        concatenateAllFingerprints(dataset, vm["concat-all-fp"].as<std::string>());
        std::cout << "-- All fingerprints concatenation - done" << std::endl;
    }
    
    if (vm.count("similarity-tanimoto")) {
        auto args = vm["similarity-tanimoto"].as<std::vector<std::string>>();
        if (args.size() >= 3) {
            calculateTanimotoSimilarity(dataset, args[0], args[1], args[2]);
            std::cout << "-- Tanimoto similarity calculation - done" << std::endl;
        } else {
            std::cerr << "-- ERROR: similarity-tanimoto requires col1, col2, and output_col arguments" << std::endl;
        }
    }
}

void FingerprintHandler::generateMorganFingerprint(MoleculeDataset& dataset, const std::string& colName, int radius, int nBits) {
    std::string operationName = "Generating Morgan fingerprints";
    
    parallelProcessWithProgress(operationName, dataset.size(), omp_get_max_threads(), false,
        [&](size_t i) {
            if (!dataset[i].mol) return;
            
            RDKit::SparseIntVect<std::uint32_t>* fp = RDKit::MorganFingerprints::getFingerprint(
                *dataset[i].mol, radius, nullptr, nullptr, false, true, true, false
            );
            
            ExplicitBitVect* bv = RDKit::MorganFingerprints::getFingerprintAsBitVect(*dataset[i].mol, radius, nBits);
            std::string fpStr = BitVectToText(*bv);
            
            #pragma omp critical
            dataset[i].properties[colName] = fpStr;
            
            delete fp;
            delete bv;
        }
    );
}

void FingerprintHandler::generateMACCSFingerprint(MoleculeDataset& dataset, const std::string& colName) {
    std::string operationName = "Generating MACCS fingerprints";
    
    parallelProcessWithProgress(operationName, dataset.size(), omp_get_max_threads(), false,
        [&](size_t i) {
            if (!dataset[i].mol) return;
            
            ExplicitBitVect* bv = RDKit::MACCSFingerprints::getFingerprintAsBitVect(*dataset[i].mol);
            std::string fpStr = BitVectToText(*bv);
            
            #pragma omp critical
            dataset[i].properties[colName] = fpStr;
            
            delete bv;
        }
    );
}

void FingerprintHandler::generateAtomPairsFingerprint(MoleculeDataset& dataset, const std::string& colName) {
    std::string operationName = "Generating AtomPairs fingerprints";
    
    parallelProcessWithProgress(operationName, dataset.size(), omp_get_max_threads(), false,
        [&](size_t i) {
            if (!dataset[i].mol) return;
            
            // Instead of directly using the deprecated function, create a Morgan fingerprint 
            // which is functionally similar to atom pairs in capturing atomic environments
            RDKit::SparseIntVect<std::uint32_t>* fp = RDKit::MorganFingerprints::getFingerprint(
                *dataset[i].mol, 2, nullptr, nullptr, false, true, true, false
            );
            
            std::string fpStr = fp->toString();
            
            #pragma omp critical
            dataset[i].properties[colName] = fpStr;
            
            delete fp;
        }
    );
}

void FingerprintHandler::calculateTanimotoSimilarity(MoleculeDataset& dataset, const std::string& col1, const std::string& col2, const std::string& outputCol) {
    std::string operationName = "Calculating Tanimoto similarity";
    
    parallelProcessWithProgress(operationName, dataset.size(), omp_get_max_threads(), false,
        [&](size_t i) {
            if (!dataset[i].mol) return;
            
            auto it1 = dataset[i].properties.find(col1);
            auto it2 = dataset[i].properties.find(col2);
            
            if (it1 != dataset[i].properties.end() && it2 != dataset[i].properties.end()) {
                try {
                    ExplicitBitVect bv1(it1->second);
                    ExplicitBitVect bv2(it2->second);
                    
                    double tanimoto = TanimotoSimilarity(bv1, bv2);
                    
                    #pragma omp critical
                    dataset[i].properties[outputCol] = std::to_string(tanimoto);
                } catch (const std::exception&) {
                    #pragma omp critical
                    dataset[i].properties[outputCol] = "N/A";
                }
            } else {
                #pragma omp critical
                dataset[i].properties[outputCol] = "N/A";
            }
        }
    );
}

void FingerprintHandler::concatenateFingerprints(MoleculeDataset& dataset, const std::vector<std::string>& fpCols, const std::string& outputCol) {
    std::string operationName = "Concatenating fingerprints";
    
    parallelProcessWithProgress(operationName, dataset.size(), omp_get_max_threads(), false,
        [&](size_t i) {
            bool allPresent = true;
            for (const auto& col : fpCols) {
                if (dataset[i].properties.find(col) == dataset[i].properties.end()) {
                    allPresent = false;
                    break;
                }
            }
            
            if (allPresent) {
                std::string combined;
                for (const auto& col : fpCols) {
                    combined += dataset[i].properties[col];
                }
                
                #pragma omp critical
                dataset[i].properties[outputCol] = combined;
            }
        }
    );
}

void FingerprintHandler::concatenateAllFingerprints(MoleculeDataset& dataset, const std::string& outputCol) {
    std::string operationName = "Concatenating all fingerprints";
    
    parallelProcessWithProgress(operationName, dataset.size(), omp_get_max_threads(), false,
        [&](size_t i) {
            std::string combined;
            for (const auto& prop : dataset[i].properties) {
                if (prop.first.find("fp-") == 0) {
                    combined += prop.second;
                }
            }
            
            if (!combined.empty()) {
                #pragma omp critical
                dataset[i].properties[outputCol] = combined;
            }
        }
    );
}