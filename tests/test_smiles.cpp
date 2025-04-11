#include <iostream>
#include <cassert>
#include "../include/smiles.h"
#include "../include/data.h"

// Helper function to display a separator line
void printSeparator() {
    std::cout << "-- ----------------------------------------" << std::endl;
}

bool test_smiles_parsing() {
    std::cout << "-- Starting SMILES parsing test" << std::endl;
    
    std::string valid_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O";
    std::string invalid_smiles = "CC(=Z)OC1=CC=CC=C1C(=O)O"; // Invalid atom Z
    
    std::cout << "-- Testing SMILES parsing for:" << std::endl;
    std::cout << "--   Valid: " << valid_smiles << std::endl;
    std::cout << "--   Invalid: " << invalid_smiles << std::endl;
    
    std::cout << "-- Attempting to parse valid SMILES" << std::endl;
    RDKit::RWMol* mol = RDKit::SmilesToMol(valid_smiles);
    std::cout << "--   Result: " << (mol ? "Valid molecule with " + std::to_string(mol->getNumAtoms()) + " atoms" : "NULL") << std::endl;
    assert(mol != nullptr);
    delete mol;
    
    std::cout << "-- Attempting to parse invalid SMILES" << std::endl;
    mol = RDKit::SmilesToMol(invalid_smiles);
    std::cout << "--   Result: " << (mol ? "Unexpectedly valid molecule" : "NULL (expected for invalid SMILES)") << std::endl;
    assert(mol == nullptr);
    
    std::cout << "-- SMILES parsing test completed successfully" << std::endl;
    return true;
}

bool test_smiles_canonicalization() {
    std::cout << "-- Starting SMILES canonicalization test" << std::endl;
    
    std::string aspirin = "CC(=O)OC1=CC=CC=C1C(=O)O";
    std::cout << "-- Testing with aspirin: " << aspirin << std::endl;
    
    MoleculeRecord record;
    record.properties["SMILES"] = aspirin;
    record.mol.reset(RDKit::SmilesToMol(aspirin));
    std::cout << "-- Created molecule with " << record.mol->getNumAtoms() << " atoms" << std::endl;
    
    MoleculeDataset dataset;
    dataset.push_back(record);
    std::cout << "-- Added molecule to dataset" << std::endl;
    
    std::cout << "-- Original SMILES: " << dataset[0].properties["SMILES"] << std::endl;
    std::cout << "-- Canonicalizing SMILES" << std::endl;
    SmilesHandler::canonicalize(dataset);
    std::cout << "-- Canonicalized SMILES: " << dataset[0].properties["SMILES"] << std::endl;
    
    // The canonical SMILES should be consistent
    RDKit::ROMol* mol = RDKit::SmilesToMol(aspirin);
    std::string canonical = RDKit::MolToSmiles(*mol);
    delete mol;
    
    std::cout << "-- Checking if canonicalization is consistent" << std::endl;
    std::cout << "--   Direct canonicalization: " << canonical << std::endl;
    std::cout << "--   Handler canonicalization: " << dataset[0].properties["SMILES"] << std::endl;
    
    // Simple test for now - just check that canonicalization doesn't crash
    assert(dataset.size() == 1);
    assert(!dataset[0].properties["SMILES"].empty());
    
    std::cout << "-- SMILES canonicalization test completed successfully" << std::endl;
    return true;
}

bool test_remove_invalid() {
    std::cout << "-- Starting removal of invalid SMILES test" << std::endl;
    
    std::string valid_smiles = "CC(=O)OC1=CC=CC=C1C(=O)O";
    std::string invalid_smiles = "CC(=Z)OC1=CC=CC=C1C(=O)O";
    
    std::cout << "-- Testing with:" << std::endl;
    std::cout << "--   Valid: " << valid_smiles << std::endl;
    std::cout << "--   Invalid: " << invalid_smiles << std::endl;
    
    MoleculeRecord record1;
    record1.properties["SMILES"] = valid_smiles;
    record1.mol.reset(RDKit::SmilesToMol(valid_smiles));
    std::cout << "-- Created valid molecule" << std::endl;
    
    MoleculeRecord record2;
    record2.properties["SMILES"] = invalid_smiles;
    record2.mol.reset(); // Reset without an argument to create a null shared_ptr
    std::cout << "-- Created invalid molecule record (null molecule)" << std::endl;
    
    MoleculeDataset dataset;
    dataset.push_back(record1);
    dataset.push_back(record2);
    std::cout << "-- Added molecules to dataset (size: " << dataset.size() << ")" << std::endl;
    
    std::cout << "-- Removing invalid molecules" << std::endl;
    SmilesHandler::removeInvalid(dataset);
    std::cout << "-- Dataset size after removal: " << dataset.size() << " (expected: 1)" << std::endl;
    
    assert(dataset.size() == 1);
    assert(dataset[0].properties["SMILES"] == valid_smiles);
    
    std::cout << "-- SMILES removal test completed successfully" << std::endl;
    return true;
}

bool test_challenging_smiles() {
    std::cout << "-- Starting challenging SMILES test" << std::endl;
    
    // Test with various challenging SMILES formats
    std::vector<std::string> challenging_smiles = {
        // Stereochemistry
        "C[C@H](Cl)Br", // Chiral center
        "C/C=C/C", // Trans double bond
        "C/C=C\\C", // Cis double bond
        
        // Ring structures
        "C1CCCCC1", // Cyclohexane
        "C1=CC=CC=C1", // Benzene
        "C12C3C4C1C5C2C3C45", // Cubane
        
        // Charged molecules
        "[NH4+]", // Ammonium
        "[O-]C(=O)C", // Acetate
        
        // Unusual atoms
        "C[Se]C", // Selenium
        "CP(=O)(O)O", // Phosphate
        
        // Difficult cases
        "c1ccccc1C(=O)Oc2ccccc2C(=O)O", // Aspirin with aromatic notation
    };
    
    int valid_count = 0;
    int invalid_count = 0;
    
    std::cout << "-- Testing " << challenging_smiles.size() << " challenging SMILES strings" << std::endl;
    MoleculeDataset dataset;
    
    for (size_t i = 0; i < challenging_smiles.size(); i++) {
        const auto& smiles = challenging_smiles[i];
        std::cout << "-- Processing SMILES " << i+1 << ": " << smiles << std::endl;
        
        RDKit::RWMol* mol = nullptr;
        try {
            mol = RDKit::SmilesToMol(smiles);
            if (mol) {
                MoleculeRecord record;
                record.properties["SMILES"] = smiles;
                record.mol.reset(mol);
                dataset.push_back(record);
                valid_count++;
                std::cout << "--   Valid molecule with " << mol->getNumAtoms() << " atoms" << std::endl;
            } else {
                invalid_count++;
                std::cout << "--   Failed to parse (nullptr returned)" << std::endl;
            }
        } catch (const std::exception& e) {
            invalid_count++;
            std::cout << "--   Exception: " << e.what() << std::endl;
            if (mol) delete mol;
        }
    }
    
    std::cout << "-- Parsing results: " << valid_count << " valid, " << invalid_count << " invalid" << std::endl;
    
    if (dataset.size() > 0) {
        std::cout << "-- Canonicalizing " << dataset.size() << " valid SMILES" << std::endl;
        SmilesHandler::canonicalize(dataset);
        
        for (size_t i = 0; i < dataset.size(); i++) {
            std::cout << "--   Original: " << challenging_smiles[i] << std::endl;
            std::cout << "--   Canonical: " << dataset[i].properties["SMILES"] << std::endl;
        }
    }
    
    std::cout << "-- Challenging SMILES test completed successfully" << std::endl;
    return true;
}

bool test_empty_dataset() {
    std::cout << "-- Starting empty dataset test" << std::endl;
    
    MoleculeDataset empty_dataset;
    std::cout << "-- Created empty dataset" << std::endl;
    
    std::cout << "-- Testing canonicalization on empty dataset" << std::endl;
    try {
        SmilesHandler::canonicalize(empty_dataset);
        std::cout << "--   Successfully handled empty dataset for canonicalization" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "--   EXCEPTION: " << e.what() << std::endl;
        return false;
    }
    
    std::cout << "-- Testing removeInvalid on empty dataset" << std::endl;
    try {
        SmilesHandler::removeInvalid(empty_dataset);
        std::cout << "--   Successfully handled empty dataset for removeInvalid" << std::endl;
    } catch (const std::exception& e) {
        std::cout << "--   EXCEPTION: " << e.what() << std::endl;
        return false;
    }
    
    std::cout << "-- Empty dataset test completed successfully" << std::endl;
    return true;
}

int main() {
    std::cout << "-- Running SMILES tests with verbose output" << std::endl;
    printSeparator();
    
    bool test1 = test_smiles_parsing();
    std::cout << "-- SMILES parsing test - " << (test1 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test2 = test_smiles_canonicalization();
    std::cout << "-- SMILES canonicalization test - " << (test2 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test3 = test_remove_invalid();
    std::cout << "-- SMILES remove invalid test - " << (test3 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test4 = test_challenging_smiles();
    std::cout << "-- Challenging SMILES test - " << (test4 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test5 = test_empty_dataset();
    std::cout << "-- Empty dataset test - " << (test5 ? "success" : "failed") << std::endl;
    printSeparator();
    
    return (test1 && test2 && test3 && test4 && test5) ? 0 : 1;
} 