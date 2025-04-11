#include "../include/conformers.h"

#include <GraphMol/Depictor/RDDepictor.h>
#include <GraphMol/DistGeomHelpers/Embedder.h>
#include <GraphMol/ForceFieldHelpers/MMFF/MMFF.h>
#include <GraphMol/ForceFieldHelpers/UFF/UFF.h>
#include <GraphMol/MolAlign/AlignMolecules.h>
#include <GraphMol/MolAlign/O3AAlignMolecules.h>
#include <GraphMol/MolDraw2D/MolDraw2DSVG.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MolTransforms/MolTransforms.h>
#include <GraphMol/RDKitBase.h>
#include <GraphMol/SmilesParse/SmilesParse.h>

#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <thread>

#include "../include/progress.h"

#ifndef NO_OPENMP
#include <omp.h>
#endif

// Define these param structs if they don't exist in the RDKit version
// These are compatibility fixes for different RDKit versions
namespace RDKit {
  namespace MMFF {
    #ifndef MMFFOPTMIZEMOLECULEPARAMS
    #define MMFFOPTMIZEMOLECULEPARAMS
    struct MMFFOptimizeMoleculeParams {
      int maxIters = 1000;
      double nonBondedThresh = 100.0;
    };
    #endif
  }
  
  namespace UFF {
    #ifndef UFFOPTMIZEMOLECULEPARAMS
    #define UFFOPTMIZEMOLECULEPARAMS
    struct UFFOptimizeMoleculeParams {
      int maxIters = 1000;
      double vdwThresh = 100.0;
    };
    #endif
  }
}

void ConformerOptions::addOptions(po::options_description& desc) {
  desc.add_options()("generate-2d-coords",
                     "Generate 2D coordinates for the molecules")(
      "generate-3d-coords", "Generate 3D coordinates for the molecules")(
      "generate-conformers", po::value<int>(), "Generate conformers (count)")(
      "minimize-energy", po::value<std::string>(),
      "Minimize energy using forcefield (MMFF94|UFF)")(
      "align-molecules", po::value<std::string>(),
      "Align molecules to a reference SMILES")(
      "rmsd-matrix", po::value<std::string>(),
      "Calculate RMSD matrix and write to file");
}

bool ConformerHandler::shouldProcess(const po::variables_map& vm) {
  return (vm.count("generate-2d-coords") != 0u) ||
         (vm.count("generate-3d-coords") != 0u) ||
         vm.count("generate-conformers") || vm.count("minimize-energy") ||
         vm.count("align-molecules") || vm.count("rmsd-matrix");
}

void ConformerHandler::process(MoleculeDataset& dataset,
                               const po::variables_map& vm) {
  std::cout << "-- Processing conformer operations" << std::endl;

  // Configure multiprocessing
  int numWorkers = std::thread::hardware_concurrency() - 2;
  numWorkers = std::max(1, numWorkers);

  if (vm.count("mpu")) {
    numWorkers = vm["mpu"].as<int>();
  } else if (vm.count("workers")) {
    numWorkers = vm["workers"].as<int>();
  } else if (vm.count("parallels")) {
    numWorkers = vm["parallels"].as<int>();
  } else if (vm.count("multiprocessing")) {
    numWorkers = vm["multiprocessing"].as<int>();
  }

  // Set OpenMP threads if available
#ifndef NO_OPENMP
  omp_set_num_threads(numWorkers);
#endif

  bool quiet = vm.count("quiet");

  if (!quiet) {
    std::cout << "-- Using " << numWorkers
              << " worker threads for conformer operations" << std::endl;
  }

  if (vm.count("generate-2d-coords")) {
    generate2DCoords(dataset);
    std::cout << "-- 2D coordinate generation - done" << std::endl;
  }

  if (vm.count("generate-3d-coords")) {
    generate3DCoords(dataset);
    std::cout << "-- 3D coordinate generation - done" << std::endl;
  }

  if (vm.count("generate-conformers")) {
    int count = vm["generate-conformers"].as<int>();
    generateConformers(dataset, count);
    std::cout << "-- Conformer generation - done" << std::endl;
  }

  if (vm.count("minimize-energy")) {
    std::string forcefield = vm["minimize-energy"].as<std::string>();
    minimizeEnergy(dataset, forcefield);
    std::cout << "-- Energy minimization - done" << std::endl;
  }

  if (vm.count("align-molecules")) {
    std::string referenceSmiles = vm["align-molecules"].as<std::string>();
    alignMolecules(dataset, referenceSmiles);
    std::cout << "-- Molecule alignment - done" << std::endl;
  }

  if (vm.count("rmsd-matrix")) {
    std::string outputFile = vm["rmsd-matrix"].as<std::string>();
    calculateRMSDMatrix(dataset, outputFile);
    std::cout << "-- RMSD matrix calculation - done" << std::endl;
  }
}

void ConformerHandler::generate2DCoords(MoleculeDataset& dataset) {
  std::string operationName = "Generating 2D coordinates";

  parallelProcessWithProgress(operationName, dataset.size(),
                              omp_get_max_threads(), false, [&](size_t i) {
                                if (!dataset[i].mol) return;

                                try {
                                  RDKit::RWMol rwmol(*dataset[i].mol);
                                  RDDepict::compute2DCoords(rwmol);
                                  
                                  // Create a clean new copy with the generated coordinates
                                  std::shared_ptr<RDKit::ROMol> mol_with_coords(new RDKit::ROMol(rwmol));

                                  #pragma omp critical
                                  {
                                    dataset[i].mol = mol_with_coords;
                                  }
                                } catch (const std::exception& e) {
                                  #pragma omp critical
                                  std::cerr << "-- WARNING: Failed to generate 2D coordinates for molecule " 
                                            << i << ": " << e.what() << std::endl;
                                }
                              });
}

void ConformerHandler::generate3DCoords(MoleculeDataset& dataset) {
  std::string operationName = "Generating 3D coordinates";

  parallelProcessWithProgress(operationName, dataset.size(),
                              omp_get_max_threads(), false, [&](size_t i) {
                                if (!dataset[i].mol) return;

                                try {
                                  RDKit::RWMol rwmol(*dataset[i].mol);
                                  
                                  // Use appropriate parameters based on molecule size
                                  const unsigned int atomCount = rwmol.getNumAtoms();
                                  
                                  RDKit::DGeomHelpers::EmbedParameters params = 
                                    RDKit::DGeomHelpers::ETKDG;
                                  
                                  // For large molecules, use different parameters
                                  if (atomCount > 100) {
                                    params.useRandomCoords = true;
                                    params.maxIterations = 5000;  // More iterations for complex molecules
                                    params.optimizerForceTol = 0.001;  // More lenient tolerance
                                    params.numThreads = 0;  // Auto-detect threads
                                  }
                                  
                                  int confId = RDKit::DGeomHelpers::EmbedMolecule(rwmol, params);
                                  
                                  if (confId < 0) {
                                    #pragma omp critical
                                    std::cerr << "-- WARNING: 3D embedding failed for molecule " 
                                              << i << " with " << atomCount << " atoms" << std::endl;
                                    return;
                                  }
                                  
                                  // Create a clean new copy with the generated coordinates
                                  std::shared_ptr<RDKit::ROMol> mol_with_coords(new RDKit::ROMol(rwmol));

                                  #pragma omp critical
                                  {
                                    dataset[i].mol = mol_with_coords;
                                  }
                                } catch (const std::exception& e) {
                                  #pragma omp critical
                                  std::cerr << "-- WARNING: Exception during 3D generation for molecule " 
                                            << i << ": " << e.what() << std::endl;
                                }
                              });
}

void ConformerHandler::generateConformers(MoleculeDataset& dataset, int count) {
  std::string operationName = "Generating conformers";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;
        
        try {
          RDKit::RWMol rwmol(*dataset[i].mol);
          
          // Use a more robust embedding method for large molecules
          const unsigned int atomCount = rwmol.getNumAtoms();
          
          // For large molecules, use different parameters
          RDKit::DGeomHelpers::EmbedParameters params = 
            RDKit::DGeomHelpers::ETKDG;
          
          if (atomCount > 100) {
            params.useRandomCoords = true;
            params.maxIterations = 5000;  // More iterations for complex molecules
            params.optimizerForceTol = 0.001;  // More lenient tolerance
            params.numThreads = 0;  // Auto-detect threads
            
            // For very large molecules, generate fewer conformers to avoid memory issues
            int adjustedCount = (atomCount > 150) ? std::min(count, 3) : count;
            
            std::vector<int> confIds;
            RDKit::DGeomHelpers::EmbedMultipleConfs(rwmol, confIds, adjustedCount);
            
            if (confIds.empty()) {
              #pragma omp critical
              std::cerr << "-- WARNING: Failed to generate conformers for large molecule " 
                        << i << std::endl;
            }
          } else {
            // For regular molecules, use standard method
            std::vector<int> confIds;
            RDKit::DGeomHelpers::EmbedMultipleConfs(rwmol, confIds, count);
          }
          
          // Create a clean new copy with the generated conformers
          std::shared_ptr<RDKit::ROMol> mol_with_confs(new RDKit::ROMol(rwmol));

          #pragma omp critical
          {
            dataset[i].mol = mol_with_confs;
          }
        } catch (const std::exception& e) {
          #pragma omp critical
          std::cerr << "-- WARNING: Exception during conformer generation for molecule " 
                    << i << ": " << e.what() << std::endl;
        }
      });
}

void ConformerHandler::minimizeEnergy(MoleculeDataset& dataset, const std::string& ff) {
  std::string forcefield = ff;
  std::string operationName = "Minimizing energy";
  std::cout << "-- " << operationName << " using " << forcefield << " forcefield" << std::endl;
  
  // Process molecules sequentially to avoid thread safety issues
  for (size_t i = 0; i < dataset.size(); i++) {
    if (!dataset[i].mol) {
      continue;
    }
    
    // Update progress
    double progress = (static_cast<double>(i) / dataset.size()) * 100.0;
    std::cout << "\r-- " << operationName << " [" 
              << std::fixed << std::setprecision(2) << std::setw(6) << progress 
              << "%]" << std::flush;
    
    try {
      RDKit::RWMol rwmol(*dataset[i].mol);
      
      // Make sure the molecule has at least one conformer
      if (rwmol.getNumConformers() == 0) {
        try {
          // Generate a conformer with very basic settings to avoid issues
          RDKit::DGeomHelpers::EmbedParameters params;
          params.useRandomCoords = true;
          params.clearConfs = true;
          params.numThreads = 1;
          
          int confId = RDKit::DGeomHelpers::EmbedMolecule(rwmol, params);
          if (confId < 0) {
            // Failed to generate 3D coords, skip this molecule
            continue;
          }
        } catch (...) {
          // Failed to generate 3D coords, skip this molecule
          continue;
        }
      }
      
      // Very simple minimization approach - just try to minimize the first conformer
      int confId = 0;
      if (rwmol.getNumConformers() > 0) {
        confId = rwmol.getConformer(0).getId();
      }
      
      bool success = false;
      
      try {
        // Try UFF minimization with simplified approach
        if (forcefield.find("UFF") != std::string::npos) {
          try {
            ForceFields::ForceField* uff = nullptr;
            uff = RDKit::UFF::constructForceField(rwmol, confId);
            
            if (uff) {
              double initialEnergy = uff->calcEnergy();
              uff->minimize(50);  // Just do 50 iterations maximum
              double finalEnergy = uff->calcEnergy();
              
              std::cout << "\n-- Molecule " << i 
                        << " minimized: " << initialEnergy << " -> " << finalEnergy << std::endl;
              
              delete uff;
              success = true;
            }
          } catch (...) {
            // UFF failed, but we'll continue
          }
        }
        
        // Try MMFF as fallback or if requested
        if (!success && (forcefield.find("MMFF") != std::string::npos || !success)) {
          try {
            RDKit::MMFF::MMFFMolProperties mmffMolProperties(rwmol);
            ForceFields::ForceField* mmff = RDKit::MMFF::constructForceField(rwmol, &mmffMolProperties, confId);
            
            if (mmff) {
              double initialEnergy = mmff->calcEnergy();
              mmff->minimize(50);  // Just do 50 iterations maximum
              double finalEnergy = mmff->calcEnergy();
              
              std::cout << "\n-- Molecule " << i 
                        << " minimized with MMFF: " << initialEnergy << " -> " << finalEnergy << std::endl;
              
              delete mmff;
              success = true;
            }
          } catch (...) {
            // MMFF failed, but we tried
          }
        }
      } catch (...) {
        // Catch all exceptions to avoid crashing
      }
      
      // Keep the molecule even if minimization failed
      dataset[i].mol = std::make_shared<RDKit::ROMol>(rwmol);
      
    } catch (...) {
      // Skip any molecule that causes problems
    }
  }
  
  // Final progress update
  std::cout << "\r-- " << operationName << " [100.00%] - Completed" << std::endl;
}

void ConformerHandler::alignMolecules(MoleculeDataset& dataset,
                                      const std::string& referenceSmiles) {
  RDKit::ROMol* refMol_raw = RDKit::SmilesToMol(referenceSmiles);
  if (!refMol_raw) {
    std::cerr << "-- ERROR: Invalid reference SMILES: " << referenceSmiles
              << std::endl;
    return;
  }
  std::shared_ptr<RDKit::ROMol> refMol(refMol_raw);
  RDKit::DGeomHelpers::EmbedMolecule(*refMol);

  std::string operationName = "Aligning molecules to reference";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;

        RDKit::ROMol mol(*dataset[i].mol);

        if (mol.getNumConformers() == 0) {
          RDKit::DGeomHelpers::EmbedMolecule(mol);
        }

        try {
          RDKit::MMFF::MMFFMolProperties ref_props(*refMol);
          RDKit::MMFF::MMFFMolProperties prb_props(mol);
          RDKit::MolAlign::O3A o3a(mol, *refMol, &prb_props, &ref_props,
                                   RDKit::MolAlign::O3A::MMFF94);
          o3a.align();
        } catch (const std::exception& e) {
          std::cerr << "-- WARNING: O3A alignment failed for molecule " << i
                    << ": " << e.what() << std::endl;
        }

#pragma omp critical
        {
            dataset[i].mol = std::shared_ptr<RDKit::ROMol>(new RDKit::ROMol(mol));
        }
      });
}

void ConformerHandler::calculateRMSDMatrix(MoleculeDataset& dataset, const std::string& outputFile) {
  if (dataset.empty()) {
    std::cerr << "-- ERROR: Empty dataset for RMSD calculation" << std::endl;
    return;
  }

  std::cout << "-- Calculating RMSD matrix..." << std::endl;
  std::ofstream outfile(outputFile);
  if (!outfile.is_open()) {
    throw std::runtime_error("Failed to open RMSD output file: " + outputFile);
  }

  // Get the first molecule with conformers
  std::shared_ptr<RDKit::ROMol> molPtr = nullptr;
  for (auto& molData : dataset) {
    if (molData.mol && molData.mol->getNumConformers() > 0) {
      molPtr = molData.mol;
      break;
    }
  }

  if (!molPtr) {
    outfile.close();
    throw std::runtime_error("No molecules with conformers found in dataset");
  }

  const RDKit::ROMol& mol = *molPtr;
  unsigned int numConformers = mol.getNumConformers();
  
  if (numConformers <= 1) {
    std::cout << "-- WARNING: Only " << numConformers << " conformer(s) available, RMSD calculation may be limited" << std::endl;
  }

  // Write header with conformer IDs
  outfile << "ConformerID";
  for (unsigned int i = 0; i < numConformers; ++i) {
    outfile << ",Conf" << mol.getConformer(i).getId();
  }
  outfile << std::endl;

  // Calculate and write RMSD matrix
  std::vector<std::vector<double>> rmsdMatrix(numConformers, std::vector<double>(numConformers, 0.0));

  // Calculate pairwise RMSD values
  for (unsigned int i = 0; i < numConformers; ++i) {
    for (unsigned int j = i; j < numConformers; ++j) {
      double rmsd = 0.0;
      
      try {
        if (i == j) {
          rmsd = 0.0; // Same conformer
        } else {
          // Make a copy of the molecule for alignment to avoid const issues
          RDKit::ROMol molCopy(mol);
          // Try to align and calculate RMSD
          rmsd = RDKit::MolAlign::alignMol(molCopy, molCopy, i, j);
        }
      } catch (const std::exception& e) {
        std::cerr << "-- WARNING: Failed to calculate RMSD between conformers " 
                  << i << " and " << j << ": " << e.what() << std::endl;
        rmsd = -1.0; // Mark as failed calculation
      }
      
      rmsdMatrix[i][j] = rmsd;
      rmsdMatrix[j][i] = rmsd; // Matrix is symmetric
    }
  }

  // Write the RMSD matrix to file
  for (unsigned int i = 0; i < numConformers; ++i) {
    outfile << "Conf" << mol.getConformer(i).getId();
    for (unsigned int j = 0; j < numConformers; ++j) {
      if (rmsdMatrix[i][j] < 0) {
        outfile << ",N/A"; // Calculation failed
      } else {
        outfile << "," << std::fixed << std::setprecision(3) << rmsdMatrix[i][j];
      }
    }
    outfile << std::endl;
  }

  outfile.close();
  std::cout << "-- RMSD matrix saved to " << outputFile << std::endl;
}