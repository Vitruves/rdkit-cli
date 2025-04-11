#include <iostream>
#include <cassert>
#include "../include/conformers.h"
#include "../include/data.h"
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>

// Helper function to display a separator line
void printSeparator() {
    std::cout << "-- ----------------------------------------" << std::endl;
}

bool test_conformer_generation() {
    std::cout << "-- Starting basic conformer generation test" << std::endl;
    
    std::string ethanol = "CCO";
    std::string cyclohexane = "C1CCCCC1";
    
    std::cout << "-- Testing with ethanol (CCO) and cyclohexane (C1CCCCC1)" << std::endl;
    
    MoleculeRecord record1;
    record1.mol.reset(RDKit::SmilesToMol(ethanol));
    std::cout << "-- Created ethanol molecule" << std::endl;
    
    MoleculeRecord record2;
    record2.mol.reset(RDKit::SmilesToMol(cyclohexane));
    std::cout << "-- Created cyclohexane molecule" << std::endl;
    
    MoleculeDataset dataset;
    dataset.push_back(record1);
    dataset.push_back(record2);
    std::cout << "-- Added molecules to dataset" << std::endl;
    
    // Generate 2D coordinates
    std::cout << "-- Generating 2D coordinates" << std::endl;
    ConformerHandler::generate2DCoords(dataset);
    
    for (size_t i = 0; i < dataset.size(); i++) {
        const auto& record = dataset[i];
        std::cout << "-- Molecule " << i << " has " << record.mol->getNumConformers() 
                 << " conformers" << std::endl;
        assert(record.mol->getNumConformers() > 0);
        
        const RDKit::Conformer& conf = record.mol->getConformer();
        std::cout << "-- Molecule " << i << " is " << (conf.is3D() ? "3D" : "2D") << std::endl;
        assert(conf.is3D() == false);
    }
    
    // Generate 3D coordinates
    std::cout << "-- Generating 3D coordinates" << std::endl;
    ConformerHandler::generate3DCoords(dataset);
    
    for (size_t i = 0; i < dataset.size(); i++) {
        const auto& record = dataset[i];
        std::cout << "-- Molecule " << i << " has " << record.mol->getNumConformers() 
                 << " conformers" << std::endl;
        assert(record.mol->getNumConformers() > 0);
        
        const RDKit::Conformer& conf = record.mol->getConformer();
        std::cout << "-- Molecule " << i << " is " << (conf.is3D() ? "3D" : "2D") << std::endl;
        assert(conf.is3D() == true);
    }
    
    std::cout << "-- Basic conformer generation test completed successfully" << std::endl;
    return true;
}

bool test_complex_molecules() {
    std::cout << "-- Starting complex molecule conformer test" << std::endl;
    
    // Test a more complex molecule - a steroid
    std::string testosterone = "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C";
    std::cout << "-- Testing with testosterone " << testosterone << std::endl;
    
    MoleculeRecord record1;
    record1.mol.reset(RDKit::SmilesToMol(testosterone));
    if (!record1.mol) {
        std::cout << "-- WARNING: Failed to create testosterone molecule" << std::endl;
        return false;
    }
    std::cout << "-- Created testosterone molecule with " << record1.mol->getNumAtoms() << " atoms" << std::endl;
    
    // Test a large peptide-like molecule
    std::string peptide = "CC(C)C[C@H](NC(=O)[C@H](Cc1ccccc1)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](CC(=O)O)NC(=O)CNC(=O)[C@@H]1CCCN1C(=O)[C@H](CC(=O)O)NC(=O)[C@H](Cc1c[nH]c2ccccc12)NC(=O)[C@H](CO)NC(=O)[C@H](Cc1ccc(O)cc1)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(C)C)NC(=O)[C@H](C)NC(=O)CNC(=O)[C@H](CC(C)C)NC(=O)[C@H](CCSC)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CC(N)=O)NC(=O)[C@H](C)NC(=O)[C@H](CCC(N)=O)NC(=O)[C@H](C)NC(=O)[C@H](CCCCN)NC(=O)[C@H](CCCNC(=N)N)NC(=O)[C@H](CC(C)C)N)C(=O)O";
    std::cout << "-- Testing with large peptide molecule" << std::endl;
    
    MoleculeRecord record2;
    record2.mol.reset(RDKit::SmilesToMol(peptide));
    if (!record2.mol) {
        std::cout << "-- WARNING: Failed to create peptide molecule" << std::endl;
        return false;
    }
    std::cout << "-- Created peptide molecule with " << record2.mol->getNumAtoms() << " atoms" << std::endl;
    
    MoleculeDataset dataset;
    dataset.push_back(record1);
    if (record2.mol) {
        dataset.push_back(record2);
    }
    
    try {
        // Generate 2D coordinates
        std::cout << "-- Generating 2D coordinates for complex molecules" << std::endl;
        ConformerHandler::generate2DCoords(dataset);
        
        for (size_t i = 0; i < dataset.size(); i++) {
            const auto& record = dataset[i];
            std::cout << "-- Complex molecule " << i << " has " << record.mol->getNumConformers() 
                     << " conformers (2D)" << std::endl;
            assert(record.mol->getNumConformers() > 0);
        }
        
        // Generate 3D coordinates
        std::cout << "-- Generating 3D coordinates for complex molecules" << std::endl;
        ConformerHandler::generate3DCoords(dataset);
        
        for (size_t i = 0; i < dataset.size(); i++) {
            const auto& record = dataset[i];
            std::cout << "-- Complex molecule " << i << " has " << record.mol->getNumConformers() 
                     << " conformers (3D)" << std::endl;
            assert(record.mol->getNumConformers() > 0);
            
            // Output some coordinates to verify 3D structure
            const RDKit::Conformer& conf = record.mol->getConformer();
            std::cout << "-- First 3 atom coordinates:" << std::endl;
            for (unsigned int j = 0; j < 3 && j < record.mol->getNumAtoms(); j++) {
                const RDGeom::Point3D& pos = conf.getAtomPos(j);
                std::cout << "-- Atom " << j << ": (" << pos.x << ", " << pos.y << ", " << pos.z << ")" << std::endl;
            }
        }
        
        std::cout << "-- Complex molecule conformer test completed successfully" << std::endl;
        return true;
    }
    catch (const std::exception& e) {
        std::cout << "-- ERROR in complex molecule test: " << e.what() << std::endl;
        return false;
    }
}

bool test_edge_cases() {
    std::cout << "-- Starting edge case conformer test" << std::endl;
    
    // Empty molecule
    std::cout << "-- Testing with empty molecule dataset" << std::endl;
    MoleculeDataset emptyDataset;
    
    try {
        ConformerHandler::generate2DCoords(emptyDataset);
        std::cout << "-- Successfully handled empty dataset (2D)" << std::endl;
        
        ConformerHandler::generate3DCoords(emptyDataset);
        std::cout << "-- Successfully handled empty dataset (3D)" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "-- ERROR: Failed to handle empty dataset: " << e.what() << std::endl;
        return false;
    }
    
    // Invalid molecule
    std::cout << "-- Testing with invalid molecule" << std::endl;
    MoleculeRecord invalidRecord;
    invalidRecord.mol.reset(); // null molecule
    
    MoleculeDataset invalidDataset;
    invalidDataset.push_back(invalidRecord);
    
    try {
        ConformerHandler::generate2DCoords(invalidDataset);
        std::cout << "-- Handler attempted to process dataset with null molecule (2D)" << std::endl;
        
        ConformerHandler::generate3DCoords(invalidDataset);
        std::cout << "-- Handler attempted to process dataset with null molecule (3D)" << std::endl;
    }
    catch (const std::exception& e) {
        std::cout << "-- EXCEPTION with null molecule: " << e.what() << std::endl;
    }
    
    std::cout << "-- Edge case conformer test completed" << std::endl;
    return true;
}

int main() {
    std::cout << "-- Running conformer tests with verbose output" << std::endl;
    printSeparator();
    
    bool test1 = test_conformer_generation();
    std::cout << "-- Basic conformer generation test - " << (test1 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test2 = test_complex_molecules();
    std::cout << "-- Complex molecule conformer test - " << (test2 ? "success" : "failed") << std::endl;
    printSeparator();
    
    bool test3 = test_edge_cases();
    std::cout << "-- Edge case conformer test - " << (test3 ? "success" : "failed") << std::endl;
    printSeparator();
    
    return (test1 && test2 && test3) ? 0 : 1;
} 