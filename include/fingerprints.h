#pragma once

#include <vector>
#include <string>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/Fingerprints/Fingerprints.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <GraphMol/Fingerprints/AtomPairs.h>
#include <DataStructs/BitVects.h>
#include <boost/program_options.hpp>
#include "data.h"

namespace po = boost::program_options;

class FingerprintOptions {
public:
    static void addOptions(po::options_description& desc);
};

class FingerprintHandler {
public:
    static bool shouldProcess(const po::variables_map& vm);
    static void process(MoleculeDataset& dataset, const po::variables_map& vm);
    static void generateMorganFingerprint(MoleculeDataset& dataset, const std::string& colName, int radius, int nBits);
    static void generateMACCSFingerprint(MoleculeDataset& dataset, const std::string& colName);
    static void generateAtomPairsFingerprint(MoleculeDataset& dataset, const std::string& colName);
    static void calculateTanimotoSimilarity(MoleculeDataset& dataset, const std::string& col1, const std::string& col2, const std::string& outputCol);
    static void concatenateFingerprints(MoleculeDataset& dataset, const std::vector<std::string>& fpCols, const std::string& outputCol);
    static void concatenateAllFingerprints(MoleculeDataset& dataset, const std::string& outputCol);
};