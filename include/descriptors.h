#pragma once

#include <vector>
#include <string>
#include <memory>
#include <map>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/Descriptors/Property.h>
#include <boost/program_options.hpp>
#include "data.h"

namespace po = boost::program_options;

class DescriptorOptions {
public:
    static void addOptions(po::options_description& desc) {
        desc.add_options()
            ("descriptors", po::value<std::string>()->default_value(""), 
                "Calculate descriptors: 2d (all 2D), 3d (all 3D), all (both 2D/3D) or a comma-separated list of descriptors")
            ("list-available-descriptors", "List all available descriptors and exit")
            ("compute-inchikey", "Compute InChIKey for molecules");
    }
};

class DescriptorHandler {
public:
    static bool shouldProcess(const po::variables_map& vm);
    static void process(MoleculeDataset& dataset, const po::variables_map& vm);
    
    // New descriptor batch processing
    static void process2DDescriptors(MoleculeDataset& dataset, int numWorkers, const po::variables_map& vm);
    static void process3DDescriptors(MoleculeDataset& dataset, int numWorkers, const po::variables_map& vm);
    static void processAllDescriptors(MoleculeDataset& dataset, int numWorkers, const po::variables_map& vm);
    static void processCustomDescriptors(MoleculeDataset& dataset, const std::string& descriptorList, int numWorkers, const po::variables_map& vm);
    
    // Helper methods
    static void listAvailableDescriptors();
    static std::map<std::string, std::string> getAvailable2DDescriptors();
    static std::map<std::string, std::string> getAvailable3DDescriptors();
    static int getDefaultNumWorkers();
    
    // Expose for testing
    static void calculateDescriptor(MoleculeDataset& dataset, const std::string& descriptorName, int numWorkers, const po::variables_map& vm);
    
    static void computeInChIKey(MoleculeDataset& dataset, int numWorkers, const po::variables_map& vm);
    
private:
    static void calculateAllDescriptors(MoleculeDataset& dataset, const std::vector<std::string>& descriptorNames, int numWorkers, const po::variables_map& vm);
};