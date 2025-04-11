#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <memory>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/Descriptors/Property.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <boost/program_options.hpp>
#include "../include/data.h"
#include "../include/descriptors.h"

namespace po = boost::program_options;

// Helper to print separator line
void printSeparator() {
    std::cout << "-- ----------------------------------------" << std::endl;
}

// Basic test for descriptor calculation
bool test_descriptor_calculation() {
    std::cout << "-- Starting basic descriptor calculation test" << std::endl;
    
    // Create a small test dataset
    MoleculeDataset dataset;
    
    {
        MoleculeRecord record;
        record.mol = std::shared_ptr<RDKit::ROMol>(RDKit::SmilesToMol("CC(=O)OC1=CC=CC=C1C(=O)O")); // Aspirin
        dataset.push_back(record);
    }
    
    {
        MoleculeRecord record;
        record.mol = std::shared_ptr<RDKit::ROMol>(RDKit::SmilesToMol("c1ccccc1")); // Benzene
        dataset.push_back(record);
    }
    
    std::cout << "-- Dataset created with " << dataset.size() << " molecules" << std::endl;
    
    // Create a variables map for testing
    po::variables_map vm;
    
    // Calculate LogP
    std::cout << "-- Calculating LogP" << std::endl;
    int numWorkers = 1; // Single thread for testing
    DescriptorHandler::calculateDescriptor(dataset, "LogP", numWorkers, vm);
    
    // Check that LogP values were calculated correctly
    std::cout << "-- Checking LogP values" << std::endl;
    double aspirin_logp = std::stod(dataset[0].properties["LogP"]);
    double benzene_logp = std::stod(dataset[1].properties["LogP"]);
    
    std::cout << "--   Aspirin LogP: " << aspirin_logp << std::endl;
    std::cout << "--   Benzene LogP: " << benzene_logp << std::endl;
    
    // Typical values: aspirin ~1.3, benzene ~1.7
    if (aspirin_logp < 1.0 || aspirin_logp > 1.5 || benzene_logp < 1.5 || benzene_logp > 2.0) {
        std::cerr << "-- ERROR: LogP values outside expected range" << std::endl;
        return false;
    }
    
    // Calculate TPSA
    std::cout << "-- Calculating TPSA" << std::endl;
    DescriptorHandler::calculateDescriptor(dataset, "TPSA", numWorkers, vm);
    
    // Check that TPSA values were calculated correctly
    std::cout << "-- Checking TPSA values" << std::endl;
    double aspirin_tpsa = std::stod(dataset[0].properties["TPSA"]);
    double benzene_tpsa = std::stod(dataset[1].properties["TPSA"]);
    
    std::cout << "--   Aspirin TPSA: " << aspirin_tpsa << std::endl;
    std::cout << "--   Benzene TPSA: " << benzene_tpsa << std::endl;
    
    // Typical values: aspirin ~63.6, benzene ~0
    if (aspirin_tpsa < 60.0 || aspirin_tpsa > 65.0 || benzene_tpsa != 0.0) {
        std::cerr << "-- ERROR: TPSA values outside expected range" << std::endl;
        return false;
    }
    
    // Calculate molecular weight
    std::cout << "-- Calculating molecular weight" << std::endl;
    DescriptorHandler::calculateDescriptor(dataset, "MolWt", numWorkers, vm);
    
    // Check that MW values were calculated correctly
    std::cout << "-- Checking molecular weight values" << std::endl;
    double aspirin_mw = std::stod(dataset[0].properties["MolWt"]);
    double benzene_mw = std::stod(dataset[1].properties["MolWt"]);
    
    std::cout << "--   Aspirin MolWt: " << aspirin_mw << std::endl;
    std::cout << "--   Benzene MolWt: " << benzene_mw << std::endl;
    
    // Typical values: aspirin ~180.16, benzene ~78.11
    if (aspirin_mw < 180.0 || aspirin_mw > 181.0 || benzene_mw < 78.0 || benzene_mw > 79.0) {
        std::cerr << "-- ERROR: Molecular weight values outside expected range" << std::endl;
        return false;
    }
    
    // Compare values calculated with direct RDKit calls
    std::cout << "-- Comparing molecular weights:" << std::endl;
    double rdkit_aspirin_mw = RDKit::Descriptors::calcExactMW(*dataset[0].mol);
    double rdkit_benzene_mw = RDKit::Descriptors::calcExactMW(*dataset[1].mol);
    
    std::cout << "--   Aspirin: " << rdkit_aspirin_mw << " g/mol" << std::endl;
    std::cout << "--   Benzene: " << rdkit_benzene_mw << " g/mol" << std::endl;
    
    // The values should match exactly
    if (std::abs(aspirin_mw - rdkit_aspirin_mw) > 0.001 || std::abs(benzene_mw - rdkit_benzene_mw) > 0.001) {
        std::cerr << "-- ERROR: Calculated values don't match direct RDKit calls" << std::endl;
        return false;
    }
    
    std::cout << "-- Basic descriptor calculation test completed successfully" << std::endl;
    return true;
}

// Test with more complex molecules
bool test_complex_molecules() {
    std::cout << "-- Starting complex molecule descriptor test" << std::endl;
    
    // Create a small test dataset with complex molecules
    MoleculeDataset dataset;
    
    // Create molecules
    std::cout << "-- Creating complex molecules:" << std::endl;
    std::string cholesterol_smiles = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C";
    std::string caffeine_smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C";
    
    std::cout << "--   Cholesterol: " << cholesterol_smiles << std::endl;
    std::cout << "--   Caffeine: " << caffeine_smiles << std::endl;
    
    {
        MoleculeRecord record;
        record.mol = std::shared_ptr<RDKit::ROMol>(RDKit::SmilesToMol(cholesterol_smiles));
        dataset.push_back(record);
    }
    
    {
        MoleculeRecord record;
        record.mol = std::shared_ptr<RDKit::ROMol>(RDKit::SmilesToMol(caffeine_smiles));
        dataset.push_back(record);
    }
    
    // Create a variables map for testing
    po::variables_map vm;
    
    // Calculate multiple descriptors
    std::cout << "-- Calculating multiple descriptors for complex molecules" << std::endl;
    int numWorkers = 1; // Single thread for testing
    
    // Use custom list of descriptors
    std::vector<std::string> descriptors = {"LogP", "TPSA", "MolWt", "NumHAcceptors", "NumHDonors", "NumRings"};
    for (const auto& desc : descriptors) {
        DescriptorHandler::calculateDescriptor(dataset, desc, numWorkers, vm);
    }
    
    // Print results
    std::cout << "-- Descriptor results for complex molecules:" << std::endl;
    std::cout << "-- Cholesterol:" << std::endl;
    for (const auto& desc : descriptors) {
        std::cout << "--   " << desc << ": " << dataset[0].properties[desc] << std::endl;
    }
    
    std::cout << "-- Caffeine:" << std::endl;
    for (const auto& desc : descriptors) {
        std::cout << "--   " << desc << ": " << dataset[1].properties[desc] << std::endl;
    }
    
    // Compare LogP values (cholesterol should be more lipophilic than caffeine)
    std::cout << "-- Comparing LogP values:" << std::endl;
    double cholesterol_logp = std::stod(dataset[0].properties["LogP"]);
    double caffeine_logp = std::stod(dataset[1].properties["LogP"]);
    
    std::cout << "--   Cholesterol: " << cholesterol_logp << std::endl;
    std::cout << "--   Caffeine: " << caffeine_logp << std::endl;
    
    if (cholesterol_logp <= caffeine_logp) {
        std::cerr << "-- ERROR: Cholesterol should have higher LogP than caffeine" << std::endl;
        return false;
    }
    
    std::cout << "-- Complex molecule descriptor test completed successfully" << std::endl;
    return true;
}

// Test edge cases and error handling
bool test_edge_cases() {
    std::cout << "-- Starting descriptor edge case test" << std::endl;
    
    // Test with empty dataset
    MoleculeDataset empty_dataset;
    std::cout << "-- Testing with empty dataset" << std::endl;
    
    int numWorkers = 1; // Single thread for testing
    
    // Create a variables map for testing
    po::variables_map vm;
    
    try {
        DescriptorHandler::calculateDescriptor(empty_dataset, "LogP", numWorkers, vm);
        std::cout << "--   Successfully handled empty dataset for LogP" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "-- ERROR: Failed to handle empty dataset for LogP: " << e.what() << std::endl;
        return false;
    }
    
    try {
        DescriptorHandler::calculateDescriptor(empty_dataset, "TPSA", numWorkers, vm);
        std::cout << "--   Successfully handled empty dataset for TPSA" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "-- ERROR: Failed to handle empty dataset for TPSA: " << e.what() << std::endl;
        return false;
    }
    
    // Test with invalid molecule (null pointer)
    MoleculeDataset invalid_dataset;
    {
        MoleculeRecord record;
        record.mol = nullptr; // Invalid molecule
        invalid_dataset.push_back(record);
    }
    
    std::cout << "-- Testing with invalid molecule (null pointer)" << std::endl;
    
    try {
        DescriptorHandler::calculateDescriptor(invalid_dataset, "LogP", numWorkers, vm);
        std::cout << "--   Handler attempted to process null molecule for LogP" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "-- ERROR: Handler failed to handle null molecule for LogP: " << e.what() << std::endl;
        return false;
    }
    
    // Test with unusual molecules (single atom, disconnected fragments)
    std::cout << "-- Testing with unusual molecules" << std::endl;
    std::string single_atom_smiles = "[Cu]";
    std::string disconnected_smiles = "C.C.C.C";
    
    std::cout << "--   Single atom: " << single_atom_smiles << std::endl;
    std::cout << "--   Disconnected: " << disconnected_smiles << std::endl;
    
    MoleculeDataset unusual_dataset;
    
    try {
        MoleculeRecord record1;
        record1.mol = std::shared_ptr<RDKit::ROMol>(RDKit::SmilesToMol(single_atom_smiles));
        unusual_dataset.push_back(record1);
        std::cout << "--   Successfully created single atom molecule" << std::endl;
        
        MoleculeRecord record2;
        record2.mol = std::shared_ptr<RDKit::ROMol>(RDKit::SmilesToMol(disconnected_smiles));
        unusual_dataset.push_back(record2);
        std::cout << "--   Successfully created disconnected molecule" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "-- ERROR: Failed to create unusual molecules: " << e.what() << std::endl;
        return false;
    }
    
    // Calculate descriptors for unusual molecules
    std::cout << "-- Calculating descriptors for unusual molecules" << std::endl;
    
    try {
        DescriptorHandler::calculateDescriptor(unusual_dataset, "LogP", numWorkers, vm);
        DescriptorHandler::calculateDescriptor(unusual_dataset, "TPSA", numWorkers, vm);
        DescriptorHandler::calculateDescriptor(unusual_dataset, "MolWt", numWorkers, vm);
        
        // Print results
        std::cout << "-- Single atom:" << std::endl;
        std::cout << "--   LogP: " << unusual_dataset[0].properties["LogP"] << std::endl;
        std::cout << "--   TPSA: " << unusual_dataset[0].properties["TPSA"] << std::endl;
        std::cout << "--   MolWt: " << unusual_dataset[0].properties["MolWt"] << std::endl;
        
        std::cout << "-- Disconnected:" << std::endl;
        std::cout << "--   LogP: " << unusual_dataset[1].properties["LogP"] << std::endl;
        std::cout << "--   TPSA: " << unusual_dataset[1].properties["TPSA"] << std::endl;
        std::cout << "--   MolWt: " << unusual_dataset[1].properties["MolWt"] << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "-- ERROR: Failed to calculate descriptors for unusual molecules: " << e.what() << std::endl;
        return false;
    }
    
    std::cout << "-- Edge case descriptor test completed" << std::endl;
    return true;
}

// Test custom descriptors
bool test_custom_descriptors() {
    std::cout << "-- Starting custom descriptor test" << std::endl;
    
    // Create a dataset with diverse structures
    MoleculeDataset dataset;
    
    std::cout << "-- Testing with diverse structures:" << std::endl;
    std::string aspirin_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O";
    std::string cholesterol_smiles = "CC(C)CCCC(C)C1CCC2C1(CCC3C2CC=C4C3(CCC(C4)O)C)C";
    
    std::cout << "--   Aspirin: " << aspirin_smiles << std::endl;
    std::cout << "--   Cholesterol: " << cholesterol_smiles << std::endl;
    
    {
        MoleculeRecord record;
        record.mol = std::shared_ptr<RDKit::ROMol>(RDKit::SmilesToMol(aspirin_smiles));
        dataset.push_back(record);
    }
    
    {
        MoleculeRecord record;
        record.mol = std::shared_ptr<RDKit::ROMol>(RDKit::SmilesToMol(cholesterol_smiles));
        dataset.push_back(record);
    }
    
    // Create a variables map for testing
    po::variables_map vm;
    
    // Calculate custom descriptors
    int numWorkers = 1; // Single thread for testing
    
    std::cout << "-- Calculating molecular weight" << std::endl;
    DescriptorHandler::calculateDescriptor(dataset, "MolWt", numWorkers, vm);
    
    std::cout << "-- Calculating number of rings" << std::endl;
    DescriptorHandler::calculateDescriptor(dataset, "NumRings", numWorkers, vm);
    
    std::cout << "-- Calculating fraction of sp3 carbons" << std::endl;
    DescriptorHandler::calculateDescriptor(dataset, "FractionCSP3", numWorkers, vm);
    
    std::cout << "-- Calculating heavy atom count" << std::endl;
    DescriptorHandler::calculateDescriptor(dataset, "HeavyAtomCount", numWorkers, vm);
    
    // Print results
    std::cout << "-- Custom descriptor results:" << std::endl;
    for (size_t i = 0; i < dataset.size(); i++) {
        std::cout << "-- Molecule " << i << ":" << std::endl;
        std::cout << "--   MolWt: " << dataset[i].properties["MolWt"] << std::endl;
        std::cout << "--   NumRings: " << dataset[i].properties["NumRings"] << std::endl;
        std::cout << "--   FractionCSP3: " << dataset[i].properties["FractionCSP3"] << std::endl;
        std::cout << "--   NumHeavyAtoms: " << dataset[i].properties["HeavyAtomCount"] << std::endl;
    }
    
    // Verify descriptors
    std::cout << "-- Verifying descriptors:" << std::endl;
    
    int aspirin_rings = std::stoi(dataset[0].properties["NumRings"]);
    int cholesterol_rings = std::stoi(dataset[1].properties["NumRings"]);
    
    std::cout << "--   Aspirin rings: " << aspirin_rings << " (expected ~1)" << std::endl;
    std::cout << "--   Cholesterol rings: " << cholesterol_rings << " (expected ~4)" << std::endl;
    
    double aspirin_frac_sp3 = std::stod(dataset[0].properties["FractionCSP3"]);
    double cholesterol_frac_sp3 = std::stod(dataset[1].properties["FractionCSP3"]);
    
    std::cout << "--   Aspirin fraction sp3: " << aspirin_frac_sp3 << " (expected low)" << std::endl;
    std::cout << "--   Cholesterol fraction sp3: " << cholesterol_frac_sp3 << " (expected high)" << std::endl;
    
    // Cholesterol should have significantly higher fraction of sp3 carbons than aspirin
    if (cholesterol_frac_sp3 <= aspirin_frac_sp3) {
        std::cerr << "-- ERROR: Cholesterol should have higher fraction of sp3 carbons than aspirin" << std::endl;
        return false;
    }
    
    std::cout << "-- Custom descriptor test completed successfully" << std::endl;
    return true;
}

int main() {
    std::cout << "-- Running descriptor tests with verbose output" << std::endl;
    printSeparator();
    
    bool test1 = test_descriptor_calculation();
    std::cout << "-- Basic descriptor calculation test - " << (test1 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test2 = test_complex_molecules();
    std::cout << "-- Complex molecule descriptor test - " << (test2 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test3 = test_edge_cases();
    std::cout << "-- Edge case descriptor test - " << (test3 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test4 = test_custom_descriptors();
    std::cout << "-- Custom descriptor test - " << (test4 ? "success" : "failed") << std::endl;
    printSeparator();
    
    return (test1 && test2 && test3 && test4) ? 0 : 1;
} 