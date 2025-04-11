#pragma once

#include <vector>
#include <string>
#include <memory>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/Descriptors/Lipinski.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <boost/program_options.hpp>
#include "data.h"

namespace po = boost::program_options;

class FilterOptions {
public:
    static void addOptions(po::options_description& desc);
};

class FilterHandler {
public:
    static bool shouldProcess(const po::variables_map& vm);
    static void process(MoleculeDataset& dataset, const po::variables_map& vm);
    static void lipinskiFilter(MoleculeDataset& dataset, const std::string& outputCol);
    static void veberFilter(MoleculeDataset& dataset, const std::string& outputCol);
    static void ghoseFilter(MoleculeDataset& dataset, const std::string& outputCol);
    static void filterByProperty(MoleculeDataset& dataset, const std::string& property, double min, double max);
    static void sortByProperty(MoleculeDataset& dataset, const std::string& property, bool ascending);
};