#pragma once

#include <vector>
#include <string>
#include <memory>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/Conformer.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/MolAlign/AlignMolecules.h>
#include <boost/program_options.hpp>
#include "data.h"

namespace po = boost::program_options;

class ConformerOptions {
public:
    static void addOptions(po::options_description& desc);
};

class ConformerHandler {
public:
    static bool shouldProcess(const po::variables_map& vm);
    static void process(MoleculeDataset& dataset, const po::variables_map& vm);
    static void generate2DCoords(MoleculeDataset& dataset);
    static void generate3DCoords(MoleculeDataset& dataset);
    static void generateConformers(MoleculeDataset& dataset, int count);
    static void minimizeEnergy(MoleculeDataset& dataset, const std::string& forcefield);
    static void alignMolecules(MoleculeDataset& dataset, const std::string& referenceSmiles);
    static void calculateRMSDMatrix(MoleculeDataset& dataset, const std::string& outputFile);
};