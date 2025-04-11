#include <iostream>
#include <cassert>
#include <fstream>
#include <filesystem>
#include <sstream>
#include "../include/data.h"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <boost/program_options.hpp>

namespace fs = std::filesystem;
namespace po = boost::program_options;

// Helper function to create a test file with SMILES
void create_test_smiles_file(const std::string& filename) {
    std::ofstream file(filename);
    file << "CC(=O)OC1=CC=CC=C1C(=O)O aspirin" << std::endl;
    file << "c1ccccc1 benzene" << std::endl;
    file << "C1CCCCC1 cyclohexane" << std::endl;
    file.close();
}

// Helper function to create a test CSV file
void create_test_csv_file(const std::string& filename) {
    std::ofstream file(filename);
    file << "ID,SMILES,Name,LogP" << std::endl;
    file << "1,CC(=O)OC1=CC=CC=C1C(=O)O,aspirin,1.43" << std::endl;
    file << "2,c1ccccc1,benzene,2.13" << std::endl;
    file << "3,C1CCCCC1,cyclohexane,3.44" << std::endl;
    file.close();
}

// Helper function to clean up test files
void cleanup_test_files(const std::vector<std::string>& filenames) {
    for (const auto& filename : filenames) {
        if (fs::exists(filename)) {
            fs::remove(filename);
        }
    }
}

// Helper function to display a separator line
void printSeparator() {
    std::cout << "-- ----------------------------------------" << std::endl;
}

bool test_dataset_operations() {
    std::cout << "-- Starting dataset operations test" << std::endl;
    
    // Create a dataset with a few molecules
    MoleculeDataset dataset;
    
    MoleculeRecord record1;
    record1.mol.reset(RDKit::SmilesToMol("CC(=O)OC1=CC=CC=C1C(=O)O"));
    record1.properties["SMILES"] = "CC(=O)OC1=CC=CC=C1C(=O)O";
    record1.properties["Name"] = "aspirin";
    
    MoleculeRecord record2;
    record2.mol.reset(RDKit::SmilesToMol("c1ccccc1"));
    record2.properties["SMILES"] = "c1ccccc1";
    record2.properties["Name"] = "benzene";
    
    dataset.push_back(record1);
    dataset.push_back(record2);
    
    // Test dataset size
    std::cout << "-- Dataset size: " << dataset.size() << " (expected: 2)" << std::endl;
    assert(dataset.size() == 2);
    
    // Test access to molecule properties
    std::cout << "-- Testing property access" << std::endl;
    std::cout << "-- First molecule name: " << dataset[0].properties["Name"] << std::endl;
    std::cout << "-- Second molecule name: " << dataset[1].properties["Name"] << std::endl;
    assert(dataset[0].properties["Name"] == "aspirin");
    assert(dataset[1].properties["Name"] == "benzene");
    
    // Verify molecules are valid
    std::cout << "-- Testing molecule pointers" << std::endl;
    assert(dataset[0].mol);
    assert(dataset[1].mol);
    
    // Test molecule atom counts
    std::cout << "-- First molecule atom count: " << dataset[0].mol->getNumAtoms() << std::endl;
    std::cout << "-- Second molecule atom count: " << dataset[1].mol->getNumAtoms() << std::endl;
    assert(dataset[0].mol->getNumAtoms() > 0);
    assert(dataset[1].mol->getNumAtoms() == 6);  // Benzene has 6 carbon atoms
    
    std::cout << "-- Dataset operations test completed" << std::endl;
    return true;
}

bool test_file_extension() {
    std::cout << "-- Starting file extension test" << std::endl;
    
    // Test file extension extraction
    std::cout << "-- Testing extension extraction for various file types" << std::endl;
    std::cout << "-- .smi: " << DataHandler::getFileExtension("test.smi") << std::endl;
    std::cout << "-- .mol: " << DataHandler::getFileExtension("test.mol") << std::endl;
    std::cout << "-- .sdf: " << DataHandler::getFileExtension("test.sdf") << std::endl;
    std::cout << "-- .csv: " << DataHandler::getFileExtension("test.csv") << std::endl;
    
    assert(DataHandler::getFileExtension("test.smi") == "smi");
    assert(DataHandler::getFileExtension("test.mol") == "mol");
    assert(DataHandler::getFileExtension("test.sdf") == "sdf");
    assert(DataHandler::getFileExtension("test.csv") == "csv");
    assert(DataHandler::getFileExtension("test.txt") == "txt");
    assert(DataHandler::getFileExtension("test") == "");
    assert(DataHandler::getFileExtension("/path/to/test.smi") == "smi");
    
    std::cout << "-- File extension test completed" << std::endl;
    return true;
}

bool test_basic_file_operations() {
    std::cout << "-- Starting basic file operations test" << std::endl;
    
    // Create test files
    std::string smiles_file = "test_basic.smi";
    std::vector<std::string> test_files = {smiles_file};
    
    // Clean up any existing test files
    cleanup_test_files(test_files);
    
    // Create a test SMILES file
    std::cout << "-- Creating test SMILES file: " << smiles_file << std::endl;
    create_test_smiles_file(smiles_file);
    
    // Simple check if file exists
    std::cout << "-- Checking if file exists: " << (fs::exists(smiles_file) ? "yes" : "no") << std::endl;
    assert(fs::exists(smiles_file));
    
    // Create a dataset from scratch
    MoleculeDataset dataset;
    
    // Add a few molecules
    std::cout << "-- Adding a molecule to dataset" << std::endl;
    MoleculeRecord record1;
    record1.mol.reset(RDKit::SmilesToMol("CC(=O)OC1=CC=CC=C1C(=O)O"));
    record1.properties["SMILES"] = "CC(=O)OC1=CC=CC=C1C(=O)O";
    record1.properties["Name"] = "aspirin";
    dataset.push_back(record1);
    
    std::cout << "-- Dataset size: " << dataset.size() << std::endl;
    
    // Clean up test files
    std::cout << "-- Cleaning up test files" << std::endl;
    cleanup_test_files(test_files);
    
    std::cout << "-- Basic file operations test completed" << std::endl;
    return true;
}

bool test_csv_file_loading() {
    std::cout << "-- Starting CSV file loading test (VERBOSE)" << std::endl;
    printSeparator();
    
    // Create test files
    std::string csv_file = "test_molecules.csv";
    std::string output_file = "test_output.csv";
    std::vector<std::string> test_files = {csv_file, output_file};
    
    // Clean up any existing test files
    cleanup_test_files(test_files);
    
    // Create a test CSV file
    std::cout << "-- Creating test CSV file: " << csv_file << std::endl;
    create_test_csv_file(csv_file);
    
    // Simple check if file exists
    std::cout << "-- Checking if CSV file exists: " << (fs::exists(csv_file) ? "yes" : "no") << std::endl;
    assert(fs::exists(csv_file));
    
    // Print file contents for debugging
    std::cout << "-- CSV file content:" << std::endl;
    std::ifstream file(csv_file);
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        std::cout << "-- Line " << ++line_count << ": " << line << std::endl;
    }
    file.close();
    printSeparator();
    
    // Now try to directly parse the CSV file with manual parsing
    std::cout << "-- Manual CSV parsing test:" << std::endl;
    file.open(csv_file);
    std::getline(file, line); // Read header
    
    std::vector<std::string> headers;
    std::stringstream ss(line);
    std::string field;
    while (std::getline(ss, field, ',')) {
        headers.push_back(field);
    }
    
    // Find SMILES column
    int smiles_col = -1;
    for (size_t i = 0; i < headers.size(); i++) {
        std::cout << "-- Header[" << i << "]: " << headers[i] << std::endl;
        if (headers[i] == "SMILES") {
            smiles_col = i;
            break;
        }
    }
    
    if (smiles_col == -1) {
        std::cout << "-- ERROR: SMILES column not found in headers!" << std::endl;
    } else {
        std::cout << "-- Found SMILES column at index: " << smiles_col << std::endl;
    }
    
    // Try to parse each line and build molecules
    int valid_mols = 0;
    int invalid_mols = 0;
    
    while (std::getline(file, line)) {
        std::vector<std::string> fields;
        std::stringstream line_ss(line);
        std::string value;
        
        while (std::getline(line_ss, value, ',')) {
            fields.push_back(value);
        }
        
        if (smiles_col < static_cast<int>(fields.size())) {
            std::string smiles = fields[smiles_col];
            std::cout << "-- Trying to parse SMILES: " << smiles << std::endl;
            
            try {
                RDKit::RWMol* mol = RDKit::SmilesToMol(smiles);
                if (mol) {
                    valid_mols++;
                    std::cout << "-- Successfully parsed molecule with " << mol->getNumAtoms() << " atoms" << std::endl;
                    delete mol;
                } else {
                    invalid_mols++;
                    std::cout << "-- Failed to parse molecule (returned nullptr)" << std::endl;
                }
            } catch (const std::exception& e) {
                invalid_mols++;
                std::cout << "-- Exception while parsing SMILES: " << e.what() << std::endl;
            }
        } else {
            std::cout << "-- WARNING: Line does not have enough fields!" << std::endl;
        }
    }
    file.close();
    
    std::cout << "-- Manual parsing results: " << valid_mols << " valid molecules, " 
              << invalid_mols << " invalid molecules" << std::endl;
    printSeparator();
    
    // Now try using the DataHandler to load the file
    std::cout << "-- Attempting to load CSV file with DataHandler" << std::endl;
    
    // Setup command line args
    po::variables_map vm;
    po::options_description desc("Options");
    DataOptions::addOptions(desc);
    
    // Set input file and SMILES column
    vm.insert(std::make_pair("file", po::variable_value(csv_file, false)));
    vm.insert(std::make_pair("smiles-col", po::variable_value(std::string("SMILES"), false)));
    vm.insert(std::make_pair("format", po::variable_value(std::string("csv"), false)));
    
    try {
        std::cout << "-- Calling DataHandler::loadFile()" << std::endl;
        MoleculeDataset dataset = DataHandler::loadFile(vm);
        std::cout << "-- DataHandler loaded " << dataset.size() << " molecules" << std::endl;
        
        for (size_t i = 0; i < dataset.size() && i < 5; i++) {
            std::cout << "-- Molecule " << i << " properties:" << std::endl;
            for (const auto& prop : dataset[i].properties) {
                std::cout << "--   " << prop.first << ": " << prop.second << std::endl;
            }
            std::cout << "--   Atom count: " << dataset[i].mol->getNumAtoms() << std::endl;
        }
        
        assert(dataset.size() > 0);
    } catch (const std::exception& e) {
        std::cout << "-- EXCEPTION in DataHandler::loadFile: " << e.what() << std::endl;
    }
    
    // Clean up test files
    std::cout << "-- Cleaning up test files" << std::endl;
    cleanup_test_files(test_files);
    
    std::cout << "-- CSV file loading test completed" << std::endl;
    return true;
}

int main() {
    std::cout << "-- Running data tests with verbose output" << std::endl;
    printSeparator();
    
    bool test1 = test_dataset_operations();
    std::cout << "-- Dataset operations test - " << (test1 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test2 = test_file_extension();
    std::cout << "-- File extension test - " << (test2 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test3 = test_basic_file_operations();
    std::cout << "-- Basic file operations test - " << (test3 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test4 = test_csv_file_loading();
    std::cout << "-- CSV file loading test - " << (test4 ? "success" : "failed") << std::endl;
    printSeparator();
    
    return (test1 && test2 && test3 && test4) ? 0 : 1;
} 