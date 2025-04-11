#pragma once

// System/standard includes
#include <string>
#include <vector>
#include <map>
#include <memory>

// Third-party libraries
#include <boost/program_options.hpp>
#include <boost/shared_ptr.hpp>

// Forward declarations for RDKit components
namespace RDKit {
    class ROMol;
}

namespace po = boost::program_options;

struct MoleculeRecord {
    std::shared_ptr<RDKit::ROMol> mol;
    std::map<std::string, std::string> properties;
};

using MoleculeDataset = std::vector<MoleculeRecord>;

class DataOptions {
public:
    static void addOptions(po::options_description& desc);
};

class DataHandler {
public:
    static MoleculeDataset loadFile(const po::variables_map& vm);
    static MoleculeDataset loadSmiles(const po::variables_map& vm);
    static void saveData(MoleculeDataset& dataset, const po::variables_map& vm);
    static std::string getFileExtension(const std::string& filename);
    
private:
    static MoleculeDataset loadSDF(const std::string& filePath);
    static MoleculeDataset loadSMILES(const std::string& filePath);
    static MoleculeDataset loadCSV(const std::string& filePath, char delimiter, const po::variables_map& vm);
};