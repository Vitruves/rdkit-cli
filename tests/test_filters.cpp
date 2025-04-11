#include <iostream>
#include <cassert>
#include "../include/filters.h"
#include "../include/data.h"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/Descriptors/Crippen.h>
#include <GraphMol/Descriptors/Lipinski.h>
#include <GraphMol/Substruct/SubstructMatch.h>

// Helper function to display a separator line
void printSeparator() {
    std::cout << "-- ----------------------------------------" << std::endl;
}

bool test_filter_operations() {
    std::cout << "-- Starting basic filter operations test" << std::endl;
    
    // Create a dataset with a few molecules
    MoleculeDataset dataset;
    
    MoleculeRecord record1;
    record1.mol.reset(RDKit::SmilesToMol("CC(=O)OC1=CC=CC=C1C(=O)O"));
    record1.properties["SMILES"] = "CC(=O)OC1=CC=CC=C1C(=O)O";
    record1.properties["Name"] = "Aspirin";
    
    MoleculeRecord record2;
    record2.mol.reset(RDKit::SmilesToMol("c1ccccc1"));
    record2.properties["SMILES"] = "c1ccccc1";
    record2.properties["Name"] = "Benzene";
    
    MoleculeRecord record3;
    record3.mol.reset(RDKit::SmilesToMol("CC1=CC=C(C=C1)O"));
    record3.properties["SMILES"] = "CC1=CC=C(C=C1)O";
    record3.properties["Name"] = "p-Cresol";
    
    dataset.push_back(record1);
    dataset.push_back(record2);
    dataset.push_back(record3);
    
    std::cout << "-- Initial dataset size: " << dataset.size() << std::endl;
    
    // Initialize RDKit descriptors for all molecules
    for (auto& record : dataset) {
        double logp = 0.0, mr = 0.0;
        RDKit::Descriptors::calcCrippenDescriptors(*(record.mol), logp, mr);
        record.properties["LogP"] = std::to_string(logp);
        record.properties["MR"] = std::to_string(mr);
        
        double mw = RDKit::Descriptors::calcAMW(*(record.mol));
        record.properties["MW"] = std::to_string(mw);
    }
    
    // Filter by LogP between 1.0 and 4.0
    std::cout << "-- Filtering by LogP between 1.0 and 4.0" << std::endl;
    MoleculeDataset filtered;
    // Create a copy of the dataset to filter
    filtered = dataset;
    // Now apply the filter to the copied dataset
    FilterHandler::filterByProperty(filtered, "LogP", 1.0, 4.0);
    std::cout << "-- Filtered dataset size: " << filtered.size() << std::endl;
    
    // Print filtered molecules
    std::cout << "-- Molecules passing LogP filter:" << std::endl;
    for (const auto& record : filtered) {
        std::cout << "--   " << record.properties.at("Name") << ": LogP = " 
                 << record.properties.at("LogP") << std::endl;
    }
    
    // Test substructure filter with a phenol pattern
    std::cout << "-- Filtering by substructure (phenol)" << std::endl;
    std::string smarts_pattern = "c1ccccc1O";
    std::unique_ptr<RDKit::ROMol> query(RDKit::SmartsToMol(smarts_pattern));
    
    if (query) {
        MoleculeDataset substructure_filtered;
        
        for (const auto& record : dataset) {
            if (record.mol && RDKit::SubstructMatch(*(record.mol), *query).size() > 0) {
                substructure_filtered.push_back(record);
            }
        }
        
        std::cout << "-- Substructure filtered dataset size: " << substructure_filtered.size() << std::endl;
        std::cout << "-- Molecules containing phenol substructure:" << std::endl;
        for (const auto& record : substructure_filtered) {
            std::cout << "--   " << record.properties.at("Name") << std::endl;
        }
        
        // Verify only p-cresol matches the phenol pattern
        assert(substructure_filtered.size() == 1);
        assert(substructure_filtered[0].properties.at("Name") == "p-Cresol");
    } else {
        std::cout << "-- ERROR: Failed to create SMARTS query pattern" << std::endl;
        return false;
    }
    
    // Test boolean property filter
    std::cout << "-- Testing boolean property filter" << std::endl;
    
    // Add a boolean property - molecules with MW > 100 get "IsHeavy = 1"
    for (auto& record : dataset) {
        double mw = std::stod(record.properties.at("MW"));
        record.properties["IsHeavy"] = (mw > 100) ? "1" : "0";
        std::cout << "--   " << record.properties.at("Name") << ": MW = " 
                 << record.properties.at("MW") << ", IsHeavy = " 
                 << record.properties.at("IsHeavy") << std::endl;
    }
    
    // Filter for heavy molecules (MW > 100)
    MoleculeDataset heavy_filtered;
    for (const auto& record : dataset) {
        if (record.properties.at("IsHeavy") == "1") {
            heavy_filtered.push_back(record);
        }
    }
    
    std::cout << "-- Heavy filtered dataset size: " << heavy_filtered.size() << std::endl;
    std::cout << "-- Heavy molecules (MW > 100):" << std::endl;
    for (const auto& record : heavy_filtered) {
        std::cout << "--   " << record.properties.at("Name") << ": MW = " 
                 << record.properties.at("MW") << std::endl;
    }
    
    // Verify aspirin is the only heavy molecule
    assert(heavy_filtered.size() == 1);
    assert(heavy_filtered[0].properties.at("Name") == "Aspirin");
    
    std::cout << "-- Basic filter operations test completed successfully" << std::endl;
    return true;
}

bool test_complex_filters() {
    std::cout << "-- Starting complex filter test" << std::endl;
    
    // Create a dataset with diverse structures
    MoleculeDataset dataset;
    
    std::vector<std::string> smiles = {
        "CC(=O)OC1=CC=CC=C1C(=O)O",           // aspirin
        "c1ccccc1",                           // benzene
        "CC1=CC=C(C=C1)O",                    // p-cresol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",       // caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",      // ibuprofen
        "COC1=CC=CC=C1OC(=O)C",               // methyl salicylate
        "CNC(=O)C1=CC=CC=C1O",                // salicylamide
        "CC(C)(C)NCC(O)COC1=CC=CC2=C1C=CC=C2" // propranolol
    };
    
    std::vector<std::string> names = {
        "Aspirin", "Benzene", "p-Cresol", "Caffeine", 
        "Ibuprofen", "Methyl salicylate", "Salicylamide", "Propranolol"
    };
    
    for (size_t i = 0; i < smiles.size(); i++) {
        MoleculeRecord record;
        record.mol.reset(RDKit::SmilesToMol(smiles[i]));
        record.properties["SMILES"] = smiles[i];
        record.properties["Name"] = names[i];
        
        // Calculate descriptors
        double logp = 0.0, mr = 0.0;
        RDKit::Descriptors::calcCrippenDescriptors(*(record.mol), logp, mr);
        record.properties["LogP"] = std::to_string(logp);
        
        double mw = RDKit::Descriptors::calcAMW(*(record.mol));
        record.properties["MW"] = std::to_string(mw);
        
        int hba = RDKit::Descriptors::calcLipinskiHBA(*(record.mol));
        record.properties["HBA"] = std::to_string(hba);
        
        int hbd = RDKit::Descriptors::calcLipinskiHBD(*(record.mol));
        record.properties["HBD"] = std::to_string(hbd);
        
        int rotBonds = RDKit::Descriptors::calcNumRotatableBonds(*(record.mol));
        record.properties["RotBonds"] = std::to_string(rotBonds);
        
        if (record.mol->getNumAtoms() > 0) {
            dataset.push_back(record);
        }
    }
    
    std::cout << "-- Created dataset with " << dataset.size() << " molecules" << std::endl;
    
    // Test combined filters (Lipinski Rule of 5)
    std::cout << "-- Testing Lipinski Rule of 5 filters" << std::endl;
    
    MoleculeDataset lipinski_compliant;
    for (const auto& record : dataset) {
        double mw = std::stod(record.properties.at("MW"));
        double logp = std::stod(record.properties.at("LogP"));
        int hba = std::stoi(record.properties.at("HBA"));
        int hbd = std::stoi(record.properties.at("HBD"));
        
        bool passes = (mw <= 500) && (logp <= 5) && (hba <= 10) && (hbd <= 5);
        
        if (passes) {
            lipinski_compliant.push_back(record);
        }
    }
    
    std::cout << "-- Lipinski compliant molecules: " << lipinski_compliant.size() << std::endl;
    for (const auto& record : lipinski_compliant) {
        std::cout << "--   " << record.properties.at("Name") << ": MW=" 
                 << record.properties.at("MW") << ", LogP=" 
                 << record.properties.at("LogP") << ", HBA=" 
                 << record.properties.at("HBA") << ", HBD=" 
                 << record.properties.at("HBD") << std::endl;
    }
    
    // Test aromatic filter
    std::cout << "-- Testing aromatic filter" << std::endl;
    
    // Add "IsAromatic" property
    for (auto& record : dataset) {
        // Check if molecule contains any aromatic atoms
        bool hasAromatic = false;
        for (const auto atom : record.mol->atoms()) {
            if (atom->getIsAromatic()) {
                hasAromatic = true;
                break;
            }
        }
        record.properties["IsAromatic"] = hasAromatic ? "1" : "0";
    }
    
    // Filter for aromatic molecules
    MoleculeDataset aromatic_filtered;
    for (const auto& record : dataset) {
        if (record.properties.at("IsAromatic") == "1") {
            aromatic_filtered.push_back(record);
        }
    }
    
    std::cout << "-- Aromatic filtered dataset size: " << aromatic_filtered.size() << std::endl;
    std::cout << "-- Aromatic molecules:" << std::endl;
    for (const auto& record : aromatic_filtered) {
        std::cout << "--   " << record.properties.at("Name") << std::endl;
    }
    
    // Verify benzene is aromation
    bool benzene_is_aromatic = false;
    for (const auto& record : aromatic_filtered) {
        if (record.properties.at("Name") == "Benzene") {
            benzene_is_aromatic = true;
            break;
        }
    }
    assert(benzene_is_aromatic);
    
    std::cout << "-- Complex filter test completed successfully" << std::endl;
    return true;
}

int main() {
    std::cout << "-- Running filter tests with verbose output" << std::endl;
    printSeparator();
    
    bool test1 = test_filter_operations();
    std::cout << "-- Basic filter operations test - " << (test1 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test2 = test_complex_filters();
    std::cout << "-- Complex filter test - " << (test2 ? "success" : "failed") << std::endl;
    printSeparator();
    
    return (test1 && test2) ? 0 : 1;
} 