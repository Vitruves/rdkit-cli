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

                                RDKit::ROMol* mol =
                                    new RDKit::ROMol(*dataset[i].mol);
                                RDDepict::compute2DCoords(*mol);

#pragma omp critical
                                {
                                    dataset[i].mol = std::shared_ptr<RDKit::ROMol>(mol);
                                }
                              });
}

void ConformerHandler::generate3DCoords(MoleculeDataset& dataset) {
  std::string operationName = "Generating 3D coordinates";

  parallelProcessWithProgress(operationName, dataset.size(),
                              omp_get_max_threads(), false, [&](size_t i) {
                                if (!dataset[i].mol) return;

                                RDKit::ROMol* mol =
                                    new RDKit::ROMol(*dataset[i].mol);
                                RDKit::DGeomHelpers::EmbedMolecule(*mol);

#pragma omp critical
                                {
                                    dataset[i].mol = std::shared_ptr<RDKit::ROMol>(mol);
                                }
                              });
}

void ConformerHandler::generateConformers(MoleculeDataset& dataset, int count) {
  std::string operationName = "Generating conformers";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;

        RDKit::ROMol* mol = new RDKit::ROMol(*dataset[i].mol);
        RDKit::DGeomHelpers::EmbedMultipleConfs(*mol, count);

#pragma omp critical
        {
            dataset[i].mol = std::shared_ptr<RDKit::ROMol>(mol);
        }
      });
}

void ConformerHandler::minimizeEnergy(MoleculeDataset& dataset,
                                      const std::string& forcefield) {
  std::string operationName = "Minimizing energy using " + forcefield;

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;

        RDKit::ROMol mol(*dataset[i].mol);

        if (mol.getNumConformers() == 0) {
          RDKit::DGeomHelpers::EmbedMolecule(mol);
        }

        for (unsigned int confId = 0; confId < mol.getNumConformers();
             ++confId) {
          if (forcefield == "MMFF94") {
            RDKit::MMFF::MMFFOptimizeMolecule(mol, confId);
          } else if (forcefield == "UFF") {
            RDKit::UFF::UFFOptimizeMolecule(mol, confId);
          }
        }

#pragma omp critical
        {
            dataset[i].mol = std::shared_ptr<RDKit::ROMol>(new RDKit::ROMol(mol));
        }
      });
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

void ConformerHandler::calculateRMSDMatrix(MoleculeDataset& dataset,
                                           const std::string& outputFile) {
  std::ofstream outFile(outputFile);
  if (!outFile.is_open()) {
    std::cerr << "-- ERROR: Could not open output file: " << outputFile
              << std::endl;
    return;
  }

  size_t n = dataset.size();
  std::cout << "-- Calculating RMSD matrix for " << n << " molecules"
            << std::endl;

  // First, ensure all molecules have conformers
  bool needsEmbedding = false;
  for (size_t i = 0; i < n; i++) {
    if (dataset[i].mol && dataset[i].mol->getNumConformers() == 0) {
      needsEmbedding = true;
      break;
    }
  }

  if (needsEmbedding) {
    std::cout << "-- Generating 3D coordinates for molecules without conformers"
              << std::endl;

    parallelProcessWithProgress("Generating conformers for RMSD calculation", n,
                                omp_get_max_threads(), false, [&](size_t i) {
                                  if (!dataset[i].mol) return;

                                  if (dataset[i].mol->getNumConformers() == 0) {
                                    RDKit::ROMol* mol =
                                        new RDKit::ROMol(*dataset[i].mol);
                                    RDKit::DGeomHelpers::EmbedMolecule(*mol);

#pragma omp critical
                                    {
                                        dataset[i].mol = std::shared_ptr<RDKit::ROMol>(mol);
                                    }
                                  }
                                });
  }

  // Now calculate RMSD matrix
  // We can't parallelize this entirely as we're writing to the file
  std::vector<std::vector<double>> rmsdMatrix(n, std::vector<double>(n, 0.0));

  // Calculate RMSD for each pair in parallel
  std::string operationName = "Calculating RMSD values";
  size_t totalPairs = n * (n - 1) / 2;  // Number of unique pairs

  std::vector<std::tuple<size_t, size_t, double>> results;
  std::mutex resultsMutex;

  parallelProcessWithProgress(operationName, totalPairs, omp_get_max_threads(),
                              false, [&](size_t pairIdx) {
                                // Convert pairIdx to i, j coordinates
                                size_t i = 0, j = 0;
                                size_t remaining = pairIdx;
                                for (i = 0; i < n; i++) {
                                  if (remaining < (n - i - 1)) {
                                    j = i + 1 + remaining;
                                    break;
                                  }
                                  remaining -= (n - i - 1);
                                }

                                if (i >= n || j >= n) return;  // Invalid index

                                if (!dataset[i].mol || !dataset[j].mol) return;

                                double rmsd =
                                    RDKit::MolAlign::alignMol(*dataset[j].mol, *dataset[i].mol);

                                // Store result
                                std::lock_guard<std::mutex> lock(resultsMutex);
                                results.emplace_back(i, j, rmsd);
                              });

  // Fill matrix from results
  for (const auto& result : results) {
    size_t i, j;
    double rmsd;
    std::tie(i, j, rmsd) = result;
    rmsdMatrix[i][j] = rmsd;
    rmsdMatrix[j][i] = rmsd;  // Matrix is symmetric
  }

  // Write matrix to file
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      outFile << std::fixed << std::setprecision(3) << rmsdMatrix[i][j] << " ";
    }
    outFile << std::endl;
  }

  outFile.close();
  std::cout << "-- RMSD matrix written to " << outputFile << std::endl;
}