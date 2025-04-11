#include "../include/smiles.h"

#include <GraphMol/ChemTransforms/ChemTransforms.h>
#include <GraphMol/ChemTransforms/MolFragmenter.h>
#include <GraphMol/Chirality.h>
#include <GraphMol/Descriptors/MolDescriptors.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/MolEnumerator/MolEnumerator.h>
#include <GraphMol/MolOps.h>
#include <GraphMol/MolStandardize/Charge.h>
#include <GraphMol/MolStandardize/Fragment.h>
#include <GraphMol/MolStandardize/MolStandardize.h>
#include <GraphMol/MolStandardize/Normalize.h>
#include <GraphMol/MolStandardize/Tautomer.h>
#include <GraphMol/ScaffoldNetwork/ScaffoldNetwork.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>
#include <GraphMol/StereoGroup.h>
#include <GraphMol/Substruct/SubstructMatch.h>

#include <algorithm>
#include <boost/make_shared.hpp>
#include <boost/shared_ptr.hpp>
#include <iostream>
#include <memory>
#include <random>
#include <set>
#include <thread>
#include <unordered_set>

#include "../include/progress.h"
#ifndef NO_OPENMP
#include <omp.h>
#endif

void SmilesOptions::addOptions(po::options_description& desc) {
  desc.add_options()("canonicalize", "Canonicalize SMILES")(
      "deduplicate", "Remove duplicates based on canonical SMILES")(
      "synonyms", po::value<int>()->default_value(0),
      "Generate N synonyms using random SMILES")(
      "fragment", po::value<std::string>(),
      "Fragment molecules (method: brics, recap)")(
      "fragment-count", po::value<int>()->default_value(0),
      "Maximum number of fragments per molecule")(
      "desalt",
      "Remove salt/solvent molecules, keeping only the largest fragment")(
      "tautomerize", "Canonicalize tautomers")(
      "remove-invalid", "Remove molecules that cannot be sanitized")(
      "neutralize", "Neutralize charged molecules")(
      "add-h", "Add hydrogens to molecules")(
      "stereoisomers", po::value<int>()->default_value(0),
      "Generate N stereoisomers per molecule")(
      "scaffold", po::value<std::string>(),
      "Generate Murcko scaffolds (name of output column)")(
      "standardize", "Standardize molecules using RDKit's standardizer")(
      "remove-stereo", "Remove stereochemistry information from molecules")(
      "match", po::value<std::string>(), "Substructure match (SMARTS pattern)")(
      "match-column", po::value<std::string>()->default_value("Match"),
      "Output column name for match results");
}

bool SmilesHandler::shouldProcess(const po::variables_map& vm) {
  return (vm.count("canonicalize") != 0u) || (vm.count("deduplicate") != 0u) ||
         (vm.count("synonyms") != 0u) || (vm.count("fragment") != 0u) ||
         (vm.count("desalt") != 0u) || (vm.count("tautomerize") != 0u) ||
         (vm.count("remove-invalid") != 0u) || (vm.count("neutralize") != 0u) ||
         (vm.count("add-h") != 0u) || (vm.count("stereoisomers") != 0u) ||
         (vm.count("scaffold") != 0u) || (vm.count("standardize") != 0u) ||
         (vm.count("remove-stereo") != 0u) || (vm.count("match") != 0u);
}

void SmilesHandler::process(MoleculeDataset& dataset,
                            const po::variables_map& vm) {
  // Configure multiprocessing
  int numWorkers = std::thread::hardware_concurrency() - 2;
  numWorkers = std::max(1, numWorkers);

  if (vm.count("mpu") != 0u) {
    numWorkers = vm["mpu"].as<int>();
  } else if (vm.count("workers") != 0u) {
    numWorkers = vm["workers"].as<int>();
  } else if (vm.count("parallels") != 0u) {
    numWorkers = vm["parallels"].as<int>();
  } else if (vm.count("multiprocessing") != 0u) {
    numWorkers = vm["multiprocessing"].as<int>();
  }

  // Set OpenMP threads if available
#ifndef NO_OPENMP
  omp_set_num_threads(numWorkers);
#endif

  bool quiet = vm.count("quiet") != 0u;

  if (!quiet) {
    std::cout << "-- Using " << numWorkers
              << " worker threads for SMILES processing" << '\n';
  }

  if (vm.count("canonicalize") != 0u) {
    std::cout << "-- Canonicalizing SMILES" << '\n';
    canonicalize(dataset);
  }

  if (vm.count("deduplicate") != 0u) {
    std::cout << "-- Removing duplicates" << '\n';
    deduplicate(dataset);
  }

  if ((vm.count("synonyms") != 0u) && vm["synonyms"].as<int>() > 0) {
    std::string method = "random";
    int count = vm["synonyms"].as<int>();
    std::cout << "-- Generating " << count << " synonym(s) per molecule using "
              << method << '\n';
    generateSynonyms(dataset, count, method);
  }

  if (vm.count("fragment") != 0u) {
    std::string method = vm["fragment"].as<std::string>();
    int count = vm["fragment-count"].as<int>();
    std::cout << "-- Fragmenting molecules using " << method << '\n';
    fragmentMolecules(dataset, count, method, vm);
  }

  if (vm.count("desalt") != 0u) {
    std::cout << "-- Removing salts/solvents" << '\n';
    desalt(dataset);
  }

  if (vm.count("tautomerize") != 0u) {
    std::cout << "-- Canonicalizing tautomers" << '\n';
    tautomerize(dataset);
  }

  if (vm.count("remove-invalid") != 0u) {
    std::cout << "-- Removing invalid molecules" << '\n';
    removeInvalid(dataset);
  }

  if (vm.count("neutralize") != 0u) {
    std::cout << "-- Neutralizing charged molecules" << '\n';
    neutralize(dataset);
  }

  if (vm.count("add-h") != 0u) {
    std::cout << "-- Adding hydrogens" << '\n';
    addHydrogens(dataset);
  }

  if ((vm.count("stereoisomers") != 0u) && vm["stereoisomers"].as<int>() > 0) {
    int count = vm["stereoisomers"].as<int>();
    std::cout << "-- Generating " << count << " stereoisomer(s) per molecule"
              << '\n';
    generateStereoisomers(dataset, count);
  }

  if (vm.count("scaffold") != 0u) {
    std::string colName = vm["scaffold"].as<std::string>();
    std::cout << "-- Generating Murcko scaffolds (column: " << colName << ")"
              << '\n';
    generateMurckoScaffold(dataset, colName);
  }

  if (vm.count("standardize") != 0u) {
    std::cout << "-- Standardizing molecules" << '\n';
    standardize(dataset);
  }

  if (vm.count("remove-stereo") != 0u) {
    std::cout << "-- Removing stereochemistry" << '\n';
    removeStereochemistry(dataset);
  }

  if (vm.count("match") != 0u) {
    std::string smarts = vm["match"].as<std::string>();
    std::string colName = vm["match-column"].as<std::string>();
    std::cout << "-- Finding substructure matches for " << smarts
              << " (column: " << colName << ")" << '\n';
    substructureMatch(dataset, smarts, colName);
  }
}

void SmilesHandler::canonicalize(MoleculeDataset& dataset) {
  std::string operationName = "Canonicalizing SMILES";

  parallelProcessWithProgress(operationName, dataset.size(),
                              omp_get_max_threads(), false, [&](size_t i) {
                                if (!dataset[i].mol) {
                                  return;
                                }

                                try {
                                  std::string smiles =
                                      RDKit::MolToSmiles(*dataset[i].mol);

#pragma omp critical
                                  dataset[i].properties["SMILES"] = smiles;
                                } catch (const std::exception& e) {
                                  // Skip molecules that fail
                                }
                              });
}

void SmilesHandler::deduplicate(MoleculeDataset& dataset) {
  // Set temporary progress bar
  std::string operationName = "Deduplicating molecules";

  // First pass: canonicalize all SMILES and count occurrences
  std::unordered_map<std::string, size_t> firstOccurrence;
  std::vector<std::string> canonSmiles(dataset.size());
  std::vector<bool> keepMolecule(dataset.size(), false);

  // First pass with progress monitoring
  parallelProcessWithProgress(
      operationName + " - Pass 1: Canonicalizing", 
      dataset.size(), 
      omp_get_max_threads(), 
      false,
      [&](size_t i) {
        if (dataset[i].mol) {
          try {
            std::string smiles = RDKit::MolToSmiles(*dataset[i].mol);
            canonSmiles[i] = smiles;
          } catch (...) {
            canonSmiles[i] = "";  // Skip molecules that fail
          }
        }
      }
  );

  // Process in chunks to identify duplicates
  ProgressTracker progress(operationName + " - Pass 2: Identifying duplicates", dataset.size());
  
  for (size_t i = 0; i < dataset.size(); i++) {
    if (!canonSmiles[i].empty()) {
      auto it = firstOccurrence.find(canonSmiles[i]);
      if (it == firstOccurrence.end()) {
        firstOccurrence[canonSmiles[i]] = i;
        keepMolecule[i] = true;
      }
    }
    progress.update();
  }

  // Calculate how many molecules we're keeping
  size_t uniqueCount = 0;
  for (bool keep : keepMolecule) {
    if (keep) {
      uniqueCount++;
    }
  }

  std::cout << "-- Found " << uniqueCount << " unique molecules from "
            << dataset.size() << " total" << '\n';

  // Create a new dataset with only unique molecules
  MoleculeDataset uniqueDataset;
  uniqueDataset.reserve(uniqueCount);

  // Second pass: copy kept molecules to the new dataset
  ProgressTracker finalProgress(operationName + " - Pass 3: Creating unique dataset", dataset.size());
  for (size_t i = 0; i < dataset.size(); i++) {
    if (keepMolecule[i]) {
      uniqueDataset.push_back(std::move(dataset[i]));
    }
    finalProgress.update();
  }

  // Swap the datasets
  dataset.swap(uniqueDataset);

  finalProgress.finish();
}

void SmilesHandler::generateSynonyms(MoleculeDataset& dataset, int count,
                                     const std::string& method) {
  if (method != "random") {
    std::cerr << "-- ERROR: Unsupported synonym generation method: " << method
              << '\n';
    return;
  }

  std::string operationName = "Generating random SMILES synonyms";
  
  // First, identify valid molecules and determine total output size
  std::vector<bool> validMolecule(dataset.size(), false);
  size_t totalOutputSize = 0;
  
  parallelProcessWithProgress(
      operationName + " (identifying valid molecules)", dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        validMolecule[i] = dataset[i].mol && dataset[i].mol->getNumAtoms() > 0;
        
        #pragma omp atomic
        totalOutputSize += validMolecule[i] ? (count + 1) : 0;
      });
  
  // Create output dataset with correct size
  MoleculeDataset newDataset;
  newDataset.reserve(totalOutputSize);
  
  // First, add all original molecules
  for (size_t i = 0; i < dataset.size(); i++) {
    if (validMolecule[i]) {
      newDataset.push_back(dataset[i]);
    }
  }
  
  // Then generate synonyms with progress tracking
  size_t synonymCount = totalOutputSize - dataset.size();
  size_t currentIndex = newDataset.size();
  
  // Pre-allocate space for all synonyms
  newDataset.resize(totalOutputSize);
  
  // Track progress for synonym generation
  ProgressTracker progress(operationName + " (generating synonyms)", synonymCount);
  
  for (size_t i = 0; i < dataset.size(); i++) {
    if (!validMolecule[i]) continue;
    
    for (int j = 0; j < count; j++) {
      MoleculeRecord newRecord = dataset[i];
      std::string newSmiles = RDKit::MolToSmiles(*dataset[i].mol, true, false, -1,
                                               false, false, false, false);
      newRecord.properties["SMILES"] = newSmiles;
      newDataset[currentIndex++] = newRecord;
      progress.update();
    }
  }
  
  progress.finish();
  
  // Replace original dataset with expanded one
  dataset = std::move(newDataset);
}

void SmilesHandler::fragmentMolecules(MoleculeDataset& dataset, int count,
                                      const std::string& method,
                                      const po::variables_map& vm) {
  std::string operationName = "Fragmenting molecules using " + method;
  
  // First pass - count fragments to properly allocate memory
  std::vector<std::vector<std::shared_ptr<RDKit::ROMol>>> allFragments(dataset.size());
  std::atomic<size_t> totalFragments{0};
  
  parallelProcessWithProgress(
      operationName + " (analyzing)", dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;
        
        try {
          std::vector<std::shared_ptr<RDKit::ROMol>> fragments;
          
          if (method == "recap") {
            // Get bonds to fragment on
            std::vector<unsigned int> bondIndices;
            for (auto* const bond : dataset[i].mol->bonds()) {
              if (bond->getBondType() == RDKit::Bond::SINGLE &&
                  bond->getBeginAtom()->getAtomicNum() > 1 &&
                  bond->getEndAtom()->getAtomicNum() > 1) {
                bondIndices.push_back(bond->getIdx());
              }
            }

            // Fragment at those bonds
            std::shared_ptr<RDKit::ROMol> fragmented(
                RDKit::MolFragmenter::fragmentOnBonds(*dataset[i].mol, bondIndices,
                                                      false));
            auto boostFrags = RDKit::MolOps::getMolFrags(*fragmented);
            for (const auto& frag : boostFrags) {
              if (frag && frag->getNumAtoms() > 0) {
                fragments.push_back(std::make_shared<RDKit::ROMol>(*frag));
              }
            }
          } else if (method == "brics") {
            // Use BRICS fragmentation from RDKit
            std::vector<unsigned int> bondIndices;

            // Find BRICS bonds using patterns
            RDKit::ROMOL_SPTR bricsPat(
                RDKit::SmartsToMol("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"));
            std::vector<RDKit::MatchVectType> matches;
            RDKit::SubstructMatch(*dataset[i].mol, *bricsPat, matches);

            for (const auto& match : matches) {
              const RDKit::Atom* a1 = dataset[i].mol->getAtomWithIdx(match[0].second);
              const RDKit::Atom* a2 = dataset[i].mol->getAtomWithIdx(match[1].second);

              // Simple check for carbon-hetero bonds (a rough approximation of
              // BRICS rules)
              if ((a1->getAtomicNum() == 6 && a2->getAtomicNum() != 6) ||
                  (a1->getAtomicNum() != 6 && a2->getAtomicNum() == 6)) {
                const RDKit::Bond* bond = dataset[i].mol->getBondBetweenAtoms(
                    match[0].second, match[1].second);
                if (bond != nullptr) {
                  bondIndices.push_back(bond->getIdx());
                }
              }
            }

            if (!bondIndices.empty()) {
              std::shared_ptr<RDKit::ROMol> fragmented(
                  RDKit::MolFragmenter::fragmentOnBonds(*dataset[i].mol, bondIndices,
                                                        false));
              auto boostFrags = RDKit::MolOps::getMolFrags(*fragmented);
              for (const auto& frag : boostFrags) {
                if (frag && frag->getNumAtoms() > 0) {
                  fragments.push_back(std::make_shared<RDKit::ROMol>(*frag));
                }
              }
            }
          } else {
            #pragma omp critical
            std::cerr << "-- ERROR: Unsupported fragmentation method: " << method
                    << '\n';
            return;
          }
          
          // Limit fragments if count > 0
          if (count > 0 && fragments.size() > static_cast<size_t>(count)) {
            fragments.resize(count);
          }
          
          allFragments[i] = fragments;
          totalFragments += fragments.size();
          
        } catch (const std::exception& e) {
          #pragma omp critical
          std::cerr << "-- WARNING: Fragmentation call failed for molecule "
                  << i << ": " << e.what() << '\n';
        }
      });
  
  // Calculate total size needed for final dataset
  size_t keepOriginals = (vm.count("keep-original-data") != 0u);
  size_t finalSize = totalFragments;
  if (keepOriginals) {
    finalSize += dataset.size();
  }
  
  // Create the new dataset
  MoleculeDataset newDataset;
  newDataset.reserve(finalSize);
  
  // First add originals if keeping them
  if (keepOriginals) {
    newDataset.insert(newDataset.end(), dataset.begin(), dataset.end());
  }
  
  // Second pass - convert fragments to records and add to new dataset
  ProgressTracker progress(operationName + " (building dataset)", totalFragments);
  
  for (size_t i = 0; i < dataset.size(); i++) {
    if (allFragments[i].empty()) continue;
    
    for (const auto& frag : allFragments[i]) {
      MoleculeRecord fragRecord = dataset[i];
      fragRecord.mol = frag;
      try {
        fragRecord.properties["Fragment_Source"] = dataset[i].properties.at("SMILES");
        fragRecord.properties["SMILES"] = RDKit::MolToSmiles(*frag);
        newDataset.push_back(fragRecord);
      } catch (const std::exception& e) {
        std::cerr << "-- WARNING: Could not generate SMILES for fragment from molecule "
                  << i << ": " << e.what() << '\n';
      }
      progress.update();
    }
  }
  
  progress.finish();
  
  // Replace original dataset with new one
  dataset = std::move(newDataset);
}

void SmilesHandler::desalt(MoleculeDataset& dataset) {
  RDKit::MolStandardize::CleanupParameters params;
  RDKit::MolStandardize::LargestFragmentChooser chooser(params);

  std::string operationName = "Removing salts/solvents";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;
        
        try {
          std::shared_ptr<RDKit::ROMol> cleaned(chooser.choose(*dataset[i].mol));
          std::string newSmiles = RDKit::MolToSmiles(*cleaned);

          #pragma omp critical
          {
            dataset[i].mol = cleaned;
            dataset[i].properties["SMILES"] = newSmiles;
          }
        } catch (...) {
          // Skip molecules that fail
        }
      });
}

void SmilesHandler::keepLargestFragment(MoleculeDataset& dataset) {
  std::string operationName = "Keeping largest fragments";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;
        
        RDKit::ROMol* mol = dataset[i].mol.get();
        
        std::vector<std::shared_ptr<RDKit::ROMol>> molFrags;
        auto boostFrags = RDKit::MolOps::getMolFrags(*mol);
        molFrags.reserve(boostFrags.size());
        for (auto& frag : boostFrags) {
          molFrags.push_back(std::make_shared<RDKit::ROMol>(*frag));
        }

        if (molFrags.size() > 1) {
          size_t maxIdx = 0;
          unsigned int maxAtoms = 0;

          for (size_t j = 0; j < molFrags.size(); j++) {
            unsigned int numAtoms = molFrags[j]->getNumAtoms();
            if (numAtoms > maxAtoms) {
              maxAtoms = numAtoms;
              maxIdx = j;
            }
          }

          #pragma omp critical
          {
            dataset[i].mol = molFrags[maxIdx];
            dataset[i].properties["SMILES"] = RDKit::MolToSmiles(*molFrags[maxIdx]);
          }
        }
      });
}

void SmilesHandler::tautomerize(MoleculeDataset& dataset) {
  RDKit::MolStandardize::CleanupParameters params;
  RDKit::MolStandardize::TautomerEnumerator tautomerizer(params);

  std::string operationName = "Canonicalizing tautomers";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;
        
        try {
          std::shared_ptr<RDKit::ROMol> taut(tautomerizer.canonicalize(*dataset[i].mol));
          std::string newSmiles = RDKit::MolToSmiles(*taut);

          #pragma omp critical
          {
            dataset[i].mol = taut;
            dataset[i].properties["SMILES"] = newSmiles;
          }
        } catch (...) {
          // Skip molecules that fail
        }
      });
}

void SmilesHandler::removeInvalid(MoleculeDataset& dataset) {
  std::string operationName = "Removing invalid molecules";
  
  // First, mark invalid molecules in parallel with progress tracking
  std::vector<bool> validMolecule(dataset.size(), false);
  
  parallelProcessWithProgress(
      operationName + " (identifying)", dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        validMolecule[i] = dataset[i].mol && dataset[i].mol->getNumAtoms() > 0;
      });
  
  // Create a new dataset with only valid molecules
  MoleculeDataset filteredDataset;
  filteredDataset.reserve(dataset.size()); // Pre-allocate max possible size
  
  ProgressTracker progressFiltering(operationName + " (filtering)", dataset.size());
  
  for (size_t i = 0; i < dataset.size(); i++) {
    if (validMolecule[i]) {
      filteredDataset.push_back(std::move(dataset[i]));
    }
    progressFiltering.update();
  }
  
  progressFiltering.finish();
  
  // Replace original dataset with filtered one
  dataset = std::move(filteredDataset);
  
  std::cout << "-- Dataset now contains " << dataset.size() << " valid molecules" << '\n';
}

void SmilesHandler::neutralize(MoleculeDataset& dataset) {
  RDKit::MolStandardize::Uncharger uncharger;
  
  std::string operationName = "Neutralizing charged molecules";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;
        
        try {
          std::shared_ptr<RDKit::ROMol> charged(uncharger.uncharge(*dataset[i].mol));
          std::string newSmiles = RDKit::MolToSmiles(*charged);

          #pragma omp critical
          {
            dataset[i].mol = charged;
            dataset[i].properties["SMILES"] = newSmiles;
          }
        } catch (const std::exception& e) {
          #pragma omp critical
          std::cerr << "-- WARNING: Neutralization failed for molecule " << i
                    << ": " << e.what() << '\n';
        }
      });
}

void SmilesHandler::addHydrogens(MoleculeDataset& dataset) {
  std::string operationName = "Adding hydrogens";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;
        
        std::shared_ptr<RDKit::ROMol> mol(RDKit::MolOps::addHs(*dataset[i].mol));
        std::string newSmiles = RDKit::MolToSmiles(*mol);

        #pragma omp critical
        {
          dataset[i].mol = mol;
          dataset[i].properties["SMILES"] = newSmiles;
        }
      });
}

void SmilesHandler::generateStereoisomers(MoleculeDataset& dataset, int count) {
  std::vector<MoleculeRecord> newRecords;

  for (const auto& record : dataset) {
    newRecords.push_back(record);
    if (!record.mol || count <= 0) {
      continue;
    }

    RDKit::ROMol molCopy(*record.mol);
    RDKit::MolOps::assignStereochemistry(molCopy, true, true);

    std::vector<RDKit::Chirality::StereoInfo> sinfoVec =
        RDKit::Chirality::findPotentialStereo(molCopy);
    if (sinfoVec.empty()) {
      continue;
    }

    std::vector<unsigned int> potentialCenters;
    for (const auto& si : sinfoVec) {
      if (si.type == RDKit::Chirality::StereoType::Atom_Tetrahedral) {
        potentialCenters.push_back(si.centeredOn);
      }
    }

    if (potentialCenters.empty()) {
      continue;
    }

    int nGenerated = 0;
    unsigned int nCenters = potentialCenters.size();
    unsigned long maxIsomers = 1UL << nCenters;
    unsigned long isomersToGenerate =
        std::min((unsigned long)count, maxIsomers > 0 ? maxIsomers - 1 : 0);

    std::set<std::string> generatedSmiles;
    generatedSmiles.insert(RDKit::MolToSmiles(molCopy));

    std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<> distrib(0, 1);

    for (unsigned long i = 1;
         i < maxIsomers && nGenerated < count && i <= isomersToGenerate; ++i) {
      RDKit::ROMol isomer(molCopy);
      for (unsigned int centerIdx = 0; centerIdx < nCenters; ++centerIdx) {
        if (((i >> centerIdx) & 1) != 0u) {
          RDKit::Atom* atom =
              isomer.getAtomWithIdx(potentialCenters[centerIdx]);
          if (atom->getChiralTag() != RDKit::Atom::CHI_UNSPECIFIED) {
            atom->invertChirality();
          }
        }
      }

      RDKit::MolOps::assignStereochemistry(isomer, true, true);
      std::string smi = RDKit::MolToSmiles(isomer);

      if (generatedSmiles.find(smi) == generatedSmiles.end()) {
        MoleculeRecord newRecord = record;
        newRecord.mol = std::make_shared<RDKit::ROMol>(isomer);
        newRecord.properties["SMILES"] = smi;
        newRecords.push_back(newRecord);
        generatedSmiles.insert(smi);
        nGenerated++;
      }
    }
  }

  dataset = std::move(newRecords);
}

void SmilesHandler::generateMurckoScaffold(MoleculeDataset& dataset,
                                           const std::string& colName) {
  std::string operationName = "Generating Murcko scaffolds";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) {
          dataset[i].properties[colName] = "";
          return;
        }
        
        RDKit::ROMol* scaffold = RDKit::MurckoDecompose(*dataset[i].mol);
        if ((scaffold != nullptr) && scaffold->getNumAtoms() > 0) {
          std::string scaffoldSmiles = RDKit::MolToSmiles(*scaffold);

          #pragma omp critical
          {
            dataset[i].properties[colName] = scaffoldSmiles;
          }

          delete scaffold;
        } else {
          #pragma omp critical
          {
            dataset[i].properties[colName] = "";
          }

          delete scaffold;
        }
      });
}

void SmilesHandler::standardize(MoleculeDataset& dataset) {
  RDKit::MolStandardize::CleanupParameters params;

  std::string operationName = "Standardizing molecules";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        try {
          if (!dataset[i].mol) {
            return;
          }

          std::shared_ptr<RDKit::ROMol> cleaned(
              RDKit::MolStandardize::cleanup(*dataset[i].mol, params));
          std::shared_ptr<RDKit::ROMol> parent(
              RDKit::MolStandardize::fragmentParent(*cleaned, params));
          std::string newSmiles = RDKit::MolToSmiles(*parent);

#pragma omp critical
          {
            dataset[i].mol = parent;
            dataset[i].properties["SMILES"] = newSmiles;
          }
        } catch (const std::exception& e) {
#pragma omp critical
          std::cerr << "-- WARNING: Standardization failed for molecule " << i
                    << ": " << e.what() << '\n';
        }
      });
}

void SmilesHandler::removeStereochemistry(MoleculeDataset& dataset) {
  std::string operationName = "Removing stereochemistry";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;
        
        auto* mol = new RDKit::ROMol(*dataset[i].mol);
        RDKit::MolOps::removeStereochemistry(*mol);
        std::string newSmiles = RDKit::MolToSmiles(*mol);

        #pragma omp critical
        {
          dataset[i].mol.reset(mol);
          dataset[i].properties["SMILES"] = newSmiles;
        }
      });
}

void SmilesHandler::substructureMatch(MoleculeDataset& dataset,
                                      const std::string& smarts,
                                      const std::string& colName) {
  RDKit::ROMol* pattern = nullptr;
  try {
    pattern = RDKit::SmartsToMol(smarts);
    if (pattern == nullptr) {
      std::cerr << "-- ERROR: Invalid SMARTS pattern: " << smarts << '\n';
      return;
    }
  } catch (const std::exception& e) {
    std::cerr << "-- ERROR: Failed to parse SMARTS pattern: " << e.what()
              << '\n';
    return;
  }

  std::string operationName = "Finding substructure matches for " + smarts;

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) {
          dataset[i].properties[colName] = "0";
          return;
        }
        
        RDKit::MatchVectType match;
        bool hasMatch = RDKit::SubstructMatch(*dataset[i].mol, *pattern, match);

        #pragma omp critical
        {
          dataset[i].properties[colName] = hasMatch ? "1" : "0";
        }
      });

  delete pattern;
}