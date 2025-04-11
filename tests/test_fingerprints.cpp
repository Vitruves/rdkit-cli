#include <iostream>
#include <cassert>
#include "../include/fingerprints.h"
#include "../include/data.h"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <string>
#include <vector>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/Fingerprints/MorganFingerprints.h>
#include <DataStructs/BitVects.h>
#include <DataStructs/BitOps.h>
#include <GraphMol/Fingerprints/MACCS.h>

// Helper function to display a separator line
void printSeparator() {
    std::cout << "-- ----------------------------------------" << std::endl;
}

bool test_fingerprint_generation() {
    std::cout << "-- Starting fingerprint generation test" << std::endl;
    
    // Create a dataset with a few molecules
    MoleculeDataset dataset;
    
    MoleculeRecord record1;
    record1.mol.reset(RDKit::SmilesToMol("CC(=O)OC1=CC=CC=C1C(=O)O")); // Aspirin
    record1.properties["SMILES"] = "CC(=O)OC1=CC=CC=C1C(=O)O";
    record1.properties["Name"] = "Aspirin";
    
    MoleculeRecord record2;
    record2.mol.reset(RDKit::SmilesToMol("c1ccccc1")); // Benzene
    record2.properties["SMILES"] = "c1ccccc1";
    record2.properties["Name"] = "Benzene";
    
    MoleculeRecord record3;
    record3.mol.reset(RDKit::SmilesToMol("CC1=CC=C(C=C1)O")); // p-Cresol
    record3.properties["SMILES"] = "CC1=CC=C(C=C1)O";
    record3.properties["Name"] = "p-Cresol";
    
    dataset.push_back(record1);
    dataset.push_back(record2);
    dataset.push_back(record3);
    
    std::cout << "-- Dataset created with " << dataset.size() << " molecules" << std::endl;
    
    // Test Morgan fingerprint generation
    std::cout << "-- Generating Morgan fingerprints" << std::endl;
    FingerprintHandler::generateMorganFingerprint(dataset, "Morgan", 2, 2048);
    
    // Check that fingerprints were generated as properties
    for (size_t i = 0; i < dataset.size(); i++) {
        assert(dataset[i].properties.find("Morgan") != dataset[i].properties.end());
        std::cout << "-- Morgan fingerprint generated for " << dataset[i].properties["Name"] << std::endl;
    }
    
    // Test MACCS fingerprint generation
    std::cout << "-- Generating MACCS fingerprints" << std::endl;
    FingerprintHandler::generateMACCSFingerprint(dataset, "MACCS");
    
    // Check that fingerprints were generated
    for (size_t i = 0; i < dataset.size(); i++) {
        assert(dataset[i].properties.find("MACCS") != dataset[i].properties.end());
        std::cout << "-- MACCS fingerprint generated for " << dataset[i].properties["Name"] << std::endl;
    }
    
    std::cout << "-- All fingerprints generated successfully" << std::endl;
    return true;
}

bool test_tanimoto_similarity() {
    std::cout << "-- Starting Tanimoto similarity test" << std::endl;
    
    // Create a dataset with similar molecules
    MoleculeDataset dataset;
    
    MoleculeRecord record1;
    record1.mol.reset(RDKit::SmilesToMol("c1ccccc1")); // Benzene
    record1.properties["SMILES"] = "c1ccccc1";
    record1.properties["Name"] = "Benzene";
    
    MoleculeRecord record2;
    record2.mol.reset(RDKit::SmilesToMol("Cc1ccccc1")); // Toluene (methylbenzene)
    record2.properties["SMILES"] = "Cc1ccccc1";
    record2.properties["Name"] = "Toluene";
    
    MoleculeRecord record3;
    record3.mol.reset(RDKit::SmilesToMol("CCO")); // Ethanol (very different)
    record3.properties["SMILES"] = "CCO";
    record3.properties["Name"] = "Ethanol";
    
    dataset.push_back(record1);
    dataset.push_back(record2);
    dataset.push_back(record3);
    
    std::cout << "-- Dataset created with " << dataset.size() << " molecules" << std::endl;
    
    // Generate Morgan fingerprints directly for this test
    std::cout << "-- Generating Morgan fingerprints for similarity test" << std::endl;
    FingerprintHandler::generateMorganFingerprint(dataset, "Morgan", 2, 2048);
    
    // Calculate and store Tanimoto similarities manually
    std::cout << "-- Calculating Tanimoto similarities" << std::endl;
    
    // Convert fingerprint strings to ExplicitBitVect objects
    std::vector<ExplicitBitVect> bitVects;
    for (size_t i = 0; i < dataset.size(); i++) {
        ExplicitBitVect bv(dataset[i].properties["Morgan"]);
        bitVects.push_back(bv);
    }
    
    // Benzene vs Toluene (should be high similarity)
    double sim_benzene_toluene = TanimotoSimilarity(bitVects[0], bitVects[1]);
    std::cout << "-- Tanimoto similarity between Benzene and Toluene: " 
              << sim_benzene_toluene << std::endl;
    
    // Benzene vs Ethanol (should be low similarity)
    double sim_benzene_ethanol = TanimotoSimilarity(bitVects[0], bitVects[2]);
    std::cout << "-- Tanimoto similarity between Benzene and Ethanol: " 
              << sim_benzene_ethanol << std::endl;
    
    // Verify similarity expectations
    assert(sim_benzene_toluene > 0.5); // Benzene and toluene should be similar
    assert(sim_benzene_ethanol < 0.5); // Benzene and ethanol should be dissimilar
    assert(sim_benzene_toluene > sim_benzene_ethanol); // Toluene more similar to benzene than ethanol
    
    std::cout << "-- Tanimoto similarity test completed successfully" << std::endl;
    return true;
}

bool test_fp_search() {
    std::cout << "-- Starting fingerprint search test" << std::endl;
    
    // Create a dataset of compounds
    MoleculeDataset dataset;
    std::vector<std::string> smiles = {
        "CC(=O)OC1=CC=CC=C1C(=O)O",           // aspirin
        "c1ccccc1",                           // benzene
        "Cc1ccccc1",                          // toluene
        "CC1=CC=C(C=C1)O",                    // p-cresol
        "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",       // caffeine
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",      // ibuprofen
        "COC1=CC=CC=C1OC(=O)C",               // methyl salicylate
        "CNC(=O)C1=CC=CC=C1O",                // salicylamide
        "CC(C)(C)NCC(O)COC1=CC=CC2=C1C=CC=C2" // propranolol
    };
    
    std::vector<std::string> names = {
        "Aspirin", "Benzene", "Toluene", "p-Cresol", "Caffeine", 
        "Ibuprofen", "Methyl salicylate", "Salicylamide", "Propranolol"
    };
    
    for (size_t i = 0; i < smiles.size(); i++) {
        MoleculeRecord record;
        record.mol.reset(RDKit::SmilesToMol(smiles[i]));
        if (record.mol) {
            record.properties["SMILES"] = smiles[i];
            record.properties["Name"] = names[i];
            dataset.push_back(record);
        }
    }
    
    std::cout << "-- Dataset created with " << dataset.size() << " molecules" << std::endl;
    
    // Generate fingerprints
    std::cout << "-- Generating Morgan fingerprints for search" << std::endl;
    FingerprintHandler::generateMorganFingerprint(dataset, "Morgan", 2, 2048);
    
    // Select a query molecule (aspirin)
    std::cout << "-- Using Aspirin as query molecule" << std::endl;
    const MoleculeRecord& query = dataset[0]; // Aspirin
    
    // Convert fingerprint strings to ExplicitBitVect objects
    std::vector<ExplicitBitVect> bitVects;
    for (size_t i = 0; i < dataset.size(); i++) {
        ExplicitBitVect bv(dataset[i].properties["Morgan"]);
        bitVects.push_back(bv);
    }
    
    // Perform similarity search
    std::vector<std::pair<size_t, double>> results;
    for (size_t i = 1; i < dataset.size(); i++) { // Skip the query itself
        double similarity = TanimotoSimilarity(bitVects[0], bitVects[i]);
        results.push_back(std::make_pair(i, similarity));
    }
    
    // Sort by decreasing similarity
    std::sort(results.begin(), results.end(), 
        [](const std::pair<size_t, double>& a, const std::pair<size_t, double>& b) {
            return a.second > b.second;
        });
    
    // Print similarity results
    std::cout << "-- Compounds most similar to " << query.properties.at("Name") << ":" << std::endl;
    for (const auto& result : results) {
        std::cout << "--   " << dataset[result.first].properties.at("Name") 
                 << ": " << result.second << std::endl;
    }
    
    // Similarity to salicylates should be high
    bool found_salicylate = false;
    for (size_t i = 0; i < 3 && i < results.size(); i++) {
        std::string name = dataset[results[i].first].properties.at("Name");
        if (name == "Methyl salicylate" || name == "Salicylamide") {
            found_salicylate = true;
            std::cout << "-- Found salicylate derivative in top 3 similar compounds" << std::endl;
            break;
        }
    }
    
    assert(found_salicylate);
    
    std::cout << "-- Fingerprint search test completed successfully" << std::endl;
    return true;
}

int main() {
    std::cout << "-- Running fingerprint tests with verbose output" << std::endl;
    printSeparator();
    
    bool test1 = test_fingerprint_generation();
    std::cout << "-- Fingerprint generation test - " << (test1 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test2 = test_tanimoto_similarity();
    std::cout << "-- Tanimoto similarity test - " << (test2 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test3 = test_fp_search();
    std::cout << "-- Fingerprint search test - " << (test3 ? "success" : "failed") << std::endl;
    printSeparator();
    
    return (test1 && test2 && test3) ? 0 : 1;
} 