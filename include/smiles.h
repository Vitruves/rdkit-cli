#pragma once

#include <vector>
#include <string>
#include <memory>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/ChemTransforms/ChemTransforms.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <boost/program_options.hpp>
#include "data.h"

namespace po = boost::program_options;

class SmilesOptions {
public:
    static void addOptions(po::options_description& desc);
};

class SmilesHandler {
public:
    static bool shouldProcess(const po::variables_map& vm);
    static void process(MoleculeDataset& dataset, const po::variables_map& vm);
    static void canonicalize(MoleculeDataset& dataset);
    static void deduplicate(MoleculeDataset& dataset);
    static void generateSynonyms(MoleculeDataset& dataset, int count, const std::string& method);
    static void fragmentMolecules(MoleculeDataset& dataset, int count, const std::string& method, const po::variables_map& vm);
    static void desalt(MoleculeDataset& dataset);
    static void keepLargestFragment(MoleculeDataset& dataset);
    static void generateRandomSmiles(MoleculeDataset& dataset, int count);
    static void tautomerize(MoleculeDataset& dataset);
    static void removeInvalid(MoleculeDataset& dataset);
    static void neutralize(MoleculeDataset& dataset);
    static void addHydrogens(MoleculeDataset& dataset);
    static void generateStereoisomers(MoleculeDataset& dataset, int count);
    static void generateMurckoScaffold(MoleculeDataset& dataset, const std::string& colName);
    static void standardize(MoleculeDataset& dataset);
    static void removeStereochemistry(MoleculeDataset& dataset);
    static void substructureMatch(MoleculeDataset& dataset, const std::string& smarts, const std::string& colName);
};