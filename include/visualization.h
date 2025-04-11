#pragma once

// System/standard includes
#include <string>

#include <boost/program_options.hpp>

#include <GraphMol/RDKitBase.h>
#include <GraphMol/MolDraw2D/MolDraw2DSVG.h>
#include <GraphMol/MolDraw2D/MolDraw2DUtils.h>
#include <GraphMol/Depictor/RDDepictor.h>
#include <GraphMol/Substruct/SubstructMatch.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

// Local includes
#include "data.h"

namespace po = boost::program_options;
namespace fs = std::filesystem;

class VisualizationOptions {
public:
    static void addOptions(po::options_description& desc);
};

class VisualizationHandler {
public:
    static bool shouldProcess(const po::variables_map& vm);
    static void process(MoleculeDataset& dataset, const po::variables_map& vm);
    static void highlightSubstructure(MoleculeDataset& dataset, const std::string& smarts, const std::string& outputDir);
    static void exportSVG(MoleculeDataset& dataset, const std::string& outputDir, int width, int height);
    static void exportPNG(MoleculeDataset& dataset, const std::string& outputDir, int width, int height);
};