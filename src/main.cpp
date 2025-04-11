#include <iostream>
#include <string>
#include <vector>
#include <boost/program_options.hpp>
#include <RDGeneral/versions.h>
#include "../include/data.h"
#include "../include/descriptors.h"
#include "../include/fingerprints.h"
#include "../include/smiles.h"
#include "../include/conformers.h"
#include "../include/filters.h"
#include "../include/visualization.h"

namespace po = boost::program_options;

void printVersion() {
    std::cout << "-- RDKit CLI - Command Line Interface for RDKit" << '\n';
    std::cout << "-- RDKit version: " << RDKit::rdkitVersion << '\n';
    std::cout << "-- Boost version: " << BOOST_VERSION / 100000 << "." << BOOST_VERSION / 100 % 1000 << "." << BOOST_VERSION % 100 << '\n';
}

int main(int argc, char* argv[]) {
    try {
        // Main options
        po::options_description general("General Options");
        general.add_options()
            ("help", "Print help message")
            ("version", "Print version information")
            ("verbose", "Enable verbose output")
            ("quiet", "Suppress warnings but keep normal logs and monitoring")
            ("mpu", po::value<int>(), "Number of CPU cores to use for processing")
            ("workers", po::value<int>(), "Alias for --mpu")
            ("parallels", po::value<int>(), "Alias for --mpu")
            ("multiprocessing", po::value<int>(), "Alias for --mpu");
            
        // Input/Output options
        po::options_description io("Input/Output Options");
        io.add_options()
            ("file", po::value<std::string>(), "Input file path")
            ("format", po::value<std::string>(), "Input file format (sdf, smi, csv, tsv)")
            ("smiles", po::value<std::string>(), "Input SMILES string")
            ("smiles-col", po::value<std::vector<std::string>>()->multitoken(), "SMILES column name(s) in CSV/TSV file")
            ("output", po::value<std::string>(), "Output file path")
            ("output-format", po::value<std::string>(), "Output file format (sdf, smi, csv, tsv)")
            ("keep-original-data", "Keep original data in output file");

        // Create category-specific options
        po::options_description descriptor("Descriptor Options");
        po::options_description fingerprint("Fingerprint Options");
        po::options_description smiles("SMILES Processing Options");
        po::options_description conformer("3D Conformer Options");
        po::options_description filter("Filtering Options");
        po::options_description visualization("Visualization Options");
        po::options_description data("Data Processing Options");
        
        // Add options from handler classes to their respective categories
        DescriptorOptions::addOptions(descriptor);
        FingerprintOptions::addOptions(fingerprint);
        SmilesOptions::addOptions(smiles);
        ConformerOptions::addOptions(conformer);
        FilterOptions::addOptions(filter);
        VisualizationOptions::addOptions(visualization);
        DataOptions::addOptions(data);
        
        // Combine all option categories for parsing
        po::options_description all_options;
        all_options.add(general).add(io).add(descriptor).add(fingerprint)
                    .add(smiles).add(conformer).add(filter).add(visualization).add(data);
        
        // User-friendly descriptions for --help display
        po::options_description visible("RDKit CLI Options");
        visible.add(general).add(io).add(descriptor).add(fingerprint)
               .add(smiles).add(conformer).add(filter).add(visualization).add(data);

        po::variables_map vm;
        po::store(po::parse_command_line(argc, argv, all_options), vm);
        po::notify(vm);

        if (vm.count("help") != 0u) {
            std::cout << "-- RDKit CLI - Command Line Interface for RDKit" << '\n';
            std::cout << "-- Usage: rdkit-cli [options]" << '\n';
            std::cout << visible << '\n';
            return 0;
        }

        if (vm.count("version")) {
            printVersion();
            return 0;
        }
        
        // Handle special commands that don't require input files
        if (vm.count("list-available-descriptors")) {
            DescriptorHandler::listAvailableDescriptors();
            return 0;
        }

        MoleculeDataset dataset;

        // Load data
        if (vm.count("file")) {
            dataset = DataHandler::loadFile(vm);
        } else if (vm.count("smiles")) {
            dataset = DataHandler::loadSmiles(vm);
        } else {
            std::cerr << "-- ERROR: No input specified. Use --file or --smiles" << '\n';
            std::cout << visible << '\n';
            return 1;
        }

        // Process SMILES operations
        if (SmilesHandler::shouldProcess(vm)) {
            SmilesHandler::process(dataset, vm);
        }

        // Process conformer operations
        if (ConformerHandler::shouldProcess(vm)) {
            ConformerHandler::process(dataset, vm);
        }

        // Process descriptor operations
        if (DescriptorHandler::shouldProcess(vm)) {
            DescriptorHandler::process(dataset, vm);
        }

        // Process fingerprint operations
        if (FingerprintHandler::shouldProcess(vm)) {
            FingerprintHandler::process(dataset, vm);
        }

        // Process filter operations
        if (FilterHandler::shouldProcess(vm)) {
            FilterHandler::process(dataset, vm);
        }

        // Process visualization operations
        if (VisualizationHandler::shouldProcess(vm)) {
            VisualizationHandler::process(dataset, vm);
        }

        // Save data
        if (vm.count("output")) {
            DataHandler::saveData(dataset, vm);
        }

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "-- ERROR: " << e.what() << '\n';
        return 1;
    }
}