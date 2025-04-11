#include "../include/filters.h"

#include <GraphMol/Descriptors/Lipinski.h>
#include <GraphMol/Descriptors/MolDescriptors.h>

#include <algorithm>
#include <iostream>
#include <thread>

#include "../include/progress.h"
#ifndef NO_OPENMP
#include <omp.h>
#endif

void FilterOptions::addOptions(po::options_description& desc) {
  desc.add_options()("lipinski-filter", po::value<std::string>(),
                     "Filter molecules by Lipinski's Rule of Five")(
      "veber-filter", po::value<std::string>(),
      "Filter molecules by Veber rules")("ghose-filter",
                                         po::value<std::string>(),
                                         "Filter molecules by Ghose rules")(
      "filter-by-property", po::value<std::vector<std::string>>()->multitoken(),
      "Filter by property (property min max)")(
      "sort-by-property", po::value<std::vector<std::string>>()->multitoken(),
      "Sort by property (property asc|desc)");
}

bool FilterHandler::shouldProcess(const po::variables_map& vm) {
  return (vm.count("lipinski-filter") != 0U) ||
         (vm.count("veber-filter") != 0U) || (vm.count("ghose-filter") != 0U) ||
         (vm.count("filter-by-property") != 0U) ||
         (vm.count("sort-by-property") != 0U);
}

void FilterHandler::process(MoleculeDataset& dataset,
                            const po::variables_map& vm) {
  std::cout << "-- Processing filters" << '\n';

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

  std::cout << "-- Using " << numWorkers << " worker threads for filtering"
            << '\n';

  if (vm.count("lipinski-filter")) {
    lipinskiFilter(dataset, vm["lipinski-filter"].as<std::string>());
    std::cout << "-- Lipinski filter - done" << std::endl;
  }

  if (vm.count("veber-filter")) {
    veberFilter(dataset, vm["veber-filter"].as<std::string>());
    std::cout << "-- Veber filter - done" << std::endl;
  }

  if (vm.count("ghose-filter")) {
    ghoseFilter(dataset, vm["ghose-filter"].as<std::string>());
    std::cout << "-- Ghose filter - done" << '\n';
  }

  if (vm.count("filter-by-property")) {
    auto args = vm["filter-by-property"].as<std::vector<std::string>>();
    if (args.size() >= 3) {
      std::string property = args[0];
      double min = std::stod(args[1]);
      double max = std::stod(args[2]);
      filterByProperty(dataset, property, min, max);
      std::cout << "-- Property filter - done" << '\n';
    } else {
      std::cerr << "-- ERROR: filter-by-property requires property, min, and "
                   "max arguments"
                << '\n';
    }
  }

  if (vm.count("sort-by-property")) {
    auto args = vm["sort-by-property"].as<std::vector<std::string>>();
    if (args.size() >= 2) {
      std::string property = args[0];
      bool ascending = (args[1] == "asc");
      sortByProperty(dataset, property, ascending);
      std::cout << "-- Property sorting - done" << '\n';
    } else {
      std::cerr << "-- ERROR: sort-by-property requires property and asc|desc "
                   "arguments"
                << '\n';
    }
  }
}

void FilterHandler::lipinskiFilter(MoleculeDataset& dataset,
                                   const std::string& outputCol) {
  std::string operationName = "Applying Lipinski filter";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;

        int violations = 0;
        double mw = RDKit::Descriptors::calcExactMW(*dataset[i].mol);
        double logp = 0.0;
        double mr = 0.0;
        RDKit::Descriptors::calcCrippenDescriptors(*dataset[i].mol, logp, mr);
        int hba = RDKit::Descriptors::calcLipinskiHBA(*dataset[i].mol);
        int hbd = RDKit::Descriptors::calcLipinskiHBD(*dataset[i].mol);

        if (mw > 500) violations++;
        if (logp > 5) violations++;
        if (hba > 10) violations++;
        if (hbd > 5) violations++;

#pragma omp critical
        dataset[i].properties[outputCol] = (violations <= 1) ? "PASS" : "FAIL";
      });
}

void FilterHandler::veberFilter(MoleculeDataset& dataset,
                                const std::string& outputCol) {
  std::string operationName = "Applying Veber filter";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;

        double tpsa = RDKit::Descriptors::calcTPSA(*dataset[i].mol);
        int rotBonds =
            RDKit::Descriptors::calcNumRotatableBonds(*dataset[i].mol);
        bool pass = (tpsa <= 140 && rotBonds <= 10);

#pragma omp critical
        dataset[i].properties[outputCol] = pass ? "PASS" : "FAIL";
      });
}

void FilterHandler::ghoseFilter(MoleculeDataset& dataset,
                                const std::string& outputCol) {
  std::string operationName = "Applying Ghose filter";

  parallelProcessWithProgress(
      operationName, dataset.size(), omp_get_max_threads(), false,
      [&](size_t i) {
        if (!dataset[i].mol) return;

        double mw = RDKit::Descriptors::calcExactMW(*dataset[i].mol);
        double logp = 0.0;
        double mr = 0.0;
        RDKit::Descriptors::calcCrippenDescriptors(*dataset[i].mol, logp, mr);
        int atomCount = dataset[i].mol->getNumAtoms();
        bool pass = (mw >= 160 && mw <= 480 && logp >= -0.4 && logp <= 5.6 &&
                     atomCount >= 20 && atomCount <= 70);

#pragma omp critical
        dataset[i].properties[outputCol] = pass ? "PASS" : "FAIL";
      });
}

void FilterHandler::filterByProperty(MoleculeDataset& dataset,
                                     const std::string& property, double min,
                                     double max) {
  std::string operationName = "Filtering by property: " + property;

  // First, collect indices to keep in parallel
  std::vector<bool> keepFlags(dataset.size(), false);

  parallelProcessWithProgress(operationName, dataset.size(),
                              omp_get_max_threads(), false, [&](size_t i) {
                                auto it = dataset[i].properties.find(property);
                                if (it == dataset[i].properties.end()) return;

                                try {
                                  double value = std::stod(it->second);
                                  if (value >= min && value <= max) {
                                    keepFlags[i] = true;
                                  }
                                } catch (...) {
                                  // Not a valid number - skip
                                }
                              });

  // Now create a new dataset with only the molecules to keep
  std::vector<MoleculeRecord> newDataset;
  newDataset.reserve(dataset.size());  // Allocate max possible size

  for (size_t i = 0; i < dataset.size(); i++) {
    if (keepFlags[i]) {
      newDataset.push_back(dataset[i]);
    }
  }

  dataset = std::move(newDataset);
  std::cout << "-- Filtered dataset now contains " << dataset.size()
            << " molecules" << '\n';
}

void FilterHandler::sortByProperty(MoleculeDataset& dataset,
                                   const std::string& property,
                                   bool ascending) {
  std::cout << "-- Sorting dataset by property: " << property << " ("
            << (ascending ? "ascending" : "descending") << ")" << '\n';

  // First extract values to sort by in parallel
  std::vector<std::pair<double, size_t>> valueIndices(dataset.size());

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < dataset.size(); i++) {
    auto it = dataset[i].properties.find(property);
    double value = std::numeric_limits<double>::quiet_NaN();

    if (it != dataset[i].properties.end()) {
      try {
        value = std::stod(it->second);
      } catch (...) {
        // Not a valid number - use NaN to sort to end
      }
    }

    valueIndices[i] = {value, i};
  }

  // Sort by value
  if (ascending) {
    std::sort(valueIndices.begin(), valueIndices.end(),
              [](const auto& a, const auto& b) {
                // Handle NaN (sort to end)
                if (std::isnan(a.first) && std::isnan(b.first)) return false;
                if (std::isnan(a.first)) return false;
                if (std::isnan(b.first)) return true;
                return a.first < b.first;
              });
  } else {
    std::sort(valueIndices.begin(), valueIndices.end(),
              [](const auto& a, const auto& b) {
                // Handle NaN (sort to end)
                if (std::isnan(a.first) && std::isnan(b.first)) return false;
                if (std::isnan(a.first)) return false;
                if (std::isnan(b.first)) return true;
                return a.first > b.first;
              });
  }

  // Create new dataset in sorted order
  MoleculeDataset newDataset;
  newDataset.reserve(dataset.size());

  for (const auto& pair : valueIndices) {
    if (!std::isnan(pair.first)) {
      newDataset.push_back(dataset[pair.second]);
    }
  }

  dataset = std::move(newDataset);
}