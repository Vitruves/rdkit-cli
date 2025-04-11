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
  ProgressTracker progress(operationName,
                           dataset.size() * 2);  // Account for two passes

  // Process in chunks to reduce memory usage
  const size_t chunkSize = 10000;

  // First pass: canonicalize all SMILES and count occurrences
  std::unordered_map<std::string, size_t> firstOccurrence;
  std::vector<std::string> canonSmiles(dataset.size());
  std::vector<bool> keepMolecule(dataset.size(), false);

  // Process in chunks
  for (size_t startIdx = 0; startIdx < dataset.size(); startIdx += chunkSize) {
    size_t endIdx = std::min(startIdx + chunkSize, dataset.size());

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
    for (size_t i = startIdx; i < endIdx; i++) {
      if (dataset[i].mol) {
        try {
          std::string smiles = RDKit::MolToSmiles(*dataset[i].mol);
          canonSmiles[i] = smiles;
        } catch (...) {
          canonSmiles[i] = "";  // Skip molecules that fail
        }
      }

#pragma omp critical
      progress.update();
    }

    // Identify duplicates for this chunk
    for (size_t i = startIdx; i < endIdx; i++) {
      if (!canonSmiles[i].empty()) {
        auto it = firstOccurrence.find(canonSmiles[i]);
        if (it == firstOccurrence.end()) {
          firstOccurrence[canonSmiles[i]] = i;
          keepMolecule[i] = true;
        }
      }
    }
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
  for (size_t i = 0; i < dataset.size(); i++) {
    if (keepMolecule[i]) {
      uniqueDataset.push_back(std::move(dataset[i]));
    }
    progress.update();
  }

  // Swap the datasets
  dataset.swap(uniqueDataset);

  progress.finish();
}

void SmilesHandler::generateSynonyms(MoleculeDataset& dataset, int count,
                                     const std::string& method) {
  if (method != "random") {
    std::cerr << "-- ERROR: Unsupported synonym generation method: " << method
              << '\n';
    return;
  }

  std::vector<MoleculeRecord> newRecords;

  for (const auto& record : dataset) {
    newRecords.push_back(record);

    for (int i = 0; i < count; i++) {
      MoleculeRecord newRecord = record;
      std::string newSmiles = RDKit::MolToSmiles(*record.mol, true, false, -1,
                                                 false, false, false, false);
      newRecord.properties["SMILES"] = newSmiles;
      newRecords.push_back(newRecord);
    }
  }

  dataset = std::move(newRecords);
}

void SmilesHandler::fragmentMolecules(MoleculeDataset& dataset, int count,
                                      const std::string& method,
                                      const po::variables_map& vm) {
  std::vector<MoleculeRecord> newRecords;

  for (const auto& record : dataset) {
    if (!record.mol) {
      continue;
    }
    MoleculeDataset frags;
    std::vector<std::shared_ptr<RDKit::ROMol>> fragmentMols;

    try {
      if (method == "recap") {
        // Get bonds to fragment on
        std::vector<unsigned int> bondIndices;
        for (auto* const bond : record.mol->bonds()) {
          if (bond->getBondType() == RDKit::Bond::SINGLE &&
              bond->getBeginAtom()->getAtomicNum() > 1 &&
              bond->getEndAtom()->getAtomicNum() > 1) {
            bondIndices.push_back(bond->getIdx());
          }
        }

        // Fragment at those bonds
        std::shared_ptr<RDKit::ROMol> fragmented(
            RDKit::MolFragmenter::fragmentOnBonds(*record.mol, bondIndices,
                                                  false));
        std::vector<std::shared_ptr<RDKit::ROMol>> recapFrags;
        auto boostFrags = RDKit::MolOps::getMolFrags(*fragmented);
        recapFrags.reserve(boostFrags.size());
        for (const auto& frag : boostFrags) {
          recapFrags.push_back(std::make_shared<RDKit::ROMol>(*frag));
        }
        fragmentMols = recapFrags;
      } else if (method == "brics") {
        // Use BRICS fragmentation from RDKit
        std::vector<unsigned int> bondIndices;

        // Find BRICS bonds using patterns
        RDKit::ROMOL_SPTR bricsPat(
            RDKit::SmartsToMol("[!$(*#*)&!D1]-!@[!$(*#*)&!D1]"));
        std::vector<RDKit::MatchVectType> matches;
        RDKit::SubstructMatch(*record.mol, *bricsPat, matches);

        for (const auto& match : matches) {
          const RDKit::Atom* a1 = record.mol->getAtomWithIdx(match[0].second);
          const RDKit::Atom* a2 = record.mol->getAtomWithIdx(match[1].second);

          // Simple check for carbon-hetero bonds (a rough approximation of
          // BRICS rules)
          if ((a1->getAtomicNum() == 6 && a2->getAtomicNum() != 6) ||
              (a1->getAtomicNum() != 6 && a2->getAtomicNum() == 6)) {
            const RDKit::Bond* bond = record.mol->getBondBetweenAtoms(
                match[0].second, match[1].second);
            if (bond != nullptr) {
              bondIndices.push_back(bond->getIdx());
            }
          }
        }

        if (!bondIndices.empty()) {
          std::shared_ptr<RDKit::ROMol> fragmented(
              RDKit::MolFragmenter::fragmentOnBonds(*record.mol, bondIndices,
                                                    false));
          std::vector<std::shared_ptr<RDKit::ROMol>> bricsFrags;
          auto boostFrags = RDKit::MolOps::getMolFrags(*fragmented);
          bricsFrags.reserve(boostFrags.size());
          for (const auto& frag : boostFrags) {
            bricsFrags.push_back(std::make_shared<RDKit::ROMol>(*frag));
          }
          fragmentMols = bricsFrags;
        }
      } else {
        std::cerr << "-- ERROR: Unsupported fragmentation method: " << method
                  << '\n';
        continue;
      }
    } catch (const std::exception& e) {
      std::cerr << "-- WARNING: Fragmentation call failed for molecule "
                << record.properties.at("SMILES") << ": " << e.what() << '\n';
      continue;
    }

    for (const auto& frag : fragmentMols) {
      if (frag && frag->getNumAtoms() > 0) {
        MoleculeRecord fragRecord = record;
        fragRecord.mol = std::make_shared<RDKit::ROMol>(*frag);
        fragRecord.properties["Fragment_Source"] =
            record.properties.at("SMILES");
        try {
          fragRecord.properties["SMILES"] = RDKit::MolToSmiles(*frag);
          frags.push_back(fragRecord);
        } catch (const std::exception& e) {
          std::cerr << "-- WARNING: Could not generate SMILES for fragment "
                       "from molecule "
                    << record.properties.at("SMILES") << ": " << e.what()
                    << '\n';
        }
      }
    }

    if ((vm.count("keep-original-data") != 0u) || frags.empty()) {
      newRecords.push_back(record);
    }

    // If count > 0, limit the number of fragments
    int added = 0;
    for (const auto& frag : frags) {
      if (count > 0 && added >= count) {
        break;
      }
      newRecords.push_back(frag);
      added++;
    }
  }

  dataset = std::move(newRecords);
}

void SmilesHandler::desalt(MoleculeDataset& dataset) {
  RDKit::MolStandardize::CleanupParameters params;
  RDKit::MolStandardize::LargestFragmentChooser chooser(params);

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (auto& i : dataset) {
    try {
      std::shared_ptr<RDKit::ROMol> cleaned(chooser.choose(*i.mol));
      std::string newSmiles = RDKit::MolToSmiles(*cleaned);

#ifndef NO_OPENMP
#pragma omp critical
#endif
      {
        i.mol = cleaned;
        i.properties["SMILES"] = newSmiles;
      }
    } catch (...) {
    }
  }
}

void SmilesHandler::keepLargestFragment(MoleculeDataset& dataset) {
#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (auto& i : dataset) {
    RDKit::ROMol* mol = i.mol.get();
    if (mol == nullptr) {
      continue;
    }

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

#ifndef NO_OPENMP
#pragma omp critical
#endif
      {
        i.mol = molFrags[maxIdx];
        i.properties["SMILES"] = RDKit::MolToSmiles(*molFrags[maxIdx]);
      }
    }
  }
}

void SmilesHandler::tautomerize(MoleculeDataset& dataset) {
  RDKit::MolStandardize::CleanupParameters params;
  RDKit::MolStandardize::TautomerEnumerator tautomerizer(params);

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (auto& i : dataset) {
    try {
      std::shared_ptr<RDKit::ROMol> taut(tautomerizer.canonicalize(*i.mol));
      std::string newSmiles = RDKit::MolToSmiles(*taut);

#ifndef NO_OPENMP
#pragma omp critical
#endif
      {
        i.mol = taut;
        i.properties["SMILES"] = newSmiles;
      }
    } catch (...) {
    }
  }
}

void SmilesHandler::removeInvalid(MoleculeDataset& dataset) {
  dataset.erase(std::remove_if(dataset.begin(), dataset.end(),
                               [](const MoleculeRecord& record) {
                                 return !record.mol ||
                                        record.mol->getNumAtoms() == 0;
                               }),
                dataset.end());
}

void SmilesHandler::neutralize(MoleculeDataset& dataset) {
  RDKit::MolStandardize::Uncharger uncharger;

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < dataset.size(); i++) {
    try {
      if (!dataset[i].mol) {
        continue;
      }
      std::shared_ptr<RDKit::ROMol> charged(
          uncharger.uncharge(*dataset[i].mol));
      std::string newSmiles = RDKit::MolToSmiles(*charged);

#ifndef NO_OPENMP
#pragma omp critical
#endif
      {
        dataset[i].mol = charged;
        dataset[i].properties["SMILES"] = newSmiles;
      }
    } catch (const std::exception& e) {
      std::cerr << "-- WARNING: Neutralization failed for molecule " << i
                << ": " << e.what() << '\n';
    }
  }
}

void SmilesHandler::addHydrogens(MoleculeDataset& dataset) {
#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (auto& i : dataset) {
    std::shared_ptr<RDKit::ROMol> mol(RDKit::MolOps::addHs(*i.mol));
    std::string newSmiles = RDKit::MolToSmiles(*mol);

#ifndef NO_OPENMP
#pragma omp critical
#endif
    {
      i.mol = mol;
      i.properties["SMILES"] = newSmiles;
    }
  }
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
#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (auto& i : dataset) {
    RDKit::ROMol* scaffold = RDKit::MurckoDecompose(*i.mol);
    if ((scaffold != nullptr) && scaffold->getNumAtoms() > 0) {
      std::string scaffoldSmiles = RDKit::MolToSmiles(*scaffold);

#ifndef NO_OPENMP
#pragma omp critical
#endif
      i.properties[colName] = scaffoldSmiles;

      delete scaffold;
    } else {
#ifndef NO_OPENMP
#pragma omp critical
#endif
      i.properties[colName] = "";

      delete scaffold;
    }
  }
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
#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (auto& i : dataset) {
    auto* mol = new RDKit::ROMol(*i.mol);
    RDKit::MolOps::removeStereochemistry(*mol);
    std::string newSmiles = RDKit::MolToSmiles(*mol);

#ifndef NO_OPENMP
#pragma omp critical
#endif
    {
      i.mol.reset(mol);
      i.properties["SMILES"] = newSmiles;
    }
  }
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

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (auto& i : dataset) {
    RDKit::MatchVectType match;
    bool hasMatch = RDKit::SubstructMatch(*i.mol, *pattern, match);

#ifndef NO_OPENMP
#pragma omp critical
#endif
    i.properties[colName] = hasMatch ? "1" : "0";
  }

  delete pattern;
}