#include "../include/data.h"

#include <GraphMol/FileParsers/FileParserUtils.h>
#include <GraphMol/FileParsers/FileParsers.h>
#include <GraphMol/FileParsers/MolSupplier.h>
#include <GraphMol/FileParsers/MolWriters.h>
#include <GraphMol/SmilesParse/SmilesParse.h>
#include <GraphMol/SmilesParse/SmilesWrite.h>

#include <boost/algorithm/string.hpp>
#include <fstream>
#include <iostream>
#include <sstream>
#include <thread>
#include <random>  // for std::random_device, std::mt19937, and std::shuffle

#include "../include/progress.h"
#ifndef NO_OPENMP
#include <omp.h>
#endif

void DataOptions::addOptions(po::options_description& desc) {
  desc.add_options()("split-output",
                     po::value<std::string>(),
                     "Split output by percentage");
  // Note: "keep-original-data" is already defined in io options
}

MoleculeDataset DataHandler::loadFile(const po::variables_map& vm) {
  std::string filePath = vm["file"].as<std::string>();
  std::string format = "auto";

  if (vm.count("format")) {
    format = vm["format"].as<std::string>();
  } else {
    // Attempt to detect format from file extension
    size_t dotPos = filePath.find_last_of('.');
    if (dotPos != std::string::npos) {
      std::string ext = filePath.substr(dotPos + 1);
      if (ext == "sdf")
        format = "sdf";
      else if (ext == "smi")
        format = "smi";
      else if (ext == "csv")
        format = "csv";
      else if (ext == "tsv")
        format = "tsv";
      else if (ext == "mol")
        format = "mol";
    }
  }

  std::cout << "-- Loading file: " << filePath << " (format: " << format << ")"
            << '\n';

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

  std::cout << "-- Using " << numWorkers
            << " worker threads for file processing" << '\n';

  MoleculeDataset dataset;

  if (format == "sdf" || format == "mol") {
    return loadSDF(filePath);
  } else if (format == "smi") {
    return loadSMILES(filePath);
  } else if (format == "csv" || format == "tsv") {
    return loadCSV(filePath, format == "tsv" ? '\t' : ',', vm);
  } else {
    throw std::runtime_error("Unsupported format: " + format);
  }

  return dataset;
}

MoleculeDataset DataHandler::loadSDF(const std::string& filePath) {
  MoleculeDataset dataset;
  RDKit::SDMolSupplier supplier(filePath, true);

  // First, count the molecules to estimate progress
  size_t moleculeCount = 0;
  {
    RDKit::SDMolSupplier countSupplier(filePath, true);
    while (!countSupplier.atEnd()) {
      countSupplier.next();
      moleculeCount++;
    }
  }

  std::string operationName = "Loading SDF file";
  ProgressTracker progress(operationName, moleculeCount);

  // Create all molecules first
  std::vector<std::unique_ptr<RDKit::ROMol>> molecules;
  while (!supplier.atEnd()) {
    std::unique_ptr<RDKit::ROMol> mol(supplier.next());
    if (mol) {
      molecules.push_back(std::move(mol));
    }
    progress.update();
  }

  // Process molecules in parallel
  dataset.resize(molecules.size());

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < molecules.size(); i++) {
    if (molecules[i]) {
      dataset[i].mol = std::make_shared<RDKit::ROMol>(*molecules[i]);

      // Extract properties
      const auto& propNames = molecules[i]->getPropList();
      for (const auto& propName : propNames) {
        if (molecules[i]->hasProp(propName)) {
          std::string propValue = molecules[i]->getProp<std::string>(propName);
          dataset[i].properties[propName] = propValue;
        }
      }
    }
  }

  molecules.clear();
  progress.finish();
  std::cout << "-- Loaded " << dataset.size() << " molecules" << '\n';
  return dataset;
}

MoleculeDataset DataHandler::loadSMILES(const std::string& filePath) {
  MoleculeDataset dataset;
  RDKit::SmilesMolSupplier supplier(filePath, "\t", 0, 1, false);

  // First, count the molecules to estimate progress
  size_t moleculeCount = 0;
  {
    std::ifstream file(filePath);
    std::string line;
    while (std::getline(file, line)) {
      moleculeCount++;
    }
  }

  std::string operationName = "Loading SMILES file";
  ProgressTracker progress(operationName, moleculeCount);

  // Create all molecules first
  std::vector<std::unique_ptr<RDKit::ROMol>> molecules;
  while (!supplier.atEnd()) {
    std::unique_ptr<RDKit::ROMol> mol(supplier.next());
    if (mol) {
      molecules.push_back(std::move(mol));
    }
    progress.update();
  }

  // Process molecules in parallel
  dataset.resize(molecules.size());

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
  for (size_t i = 0; i < molecules.size(); i++) {
    if (molecules[i]) {
      dataset[i].mol = std::make_shared<RDKit::ROMol>(*molecules[i]);
    }
  }

  molecules.clear();
  progress.finish();
  std::cout << "-- Loaded " << dataset.size() << " molecules" << '\n';
  return dataset;
}

MoleculeDataset DataHandler::loadCSV(const std::string& filePath,
                                     char delimiter,
                                     const po::variables_map& vm) {
  MoleculeDataset dataset;

  std::ifstream file(filePath);
  if (!file.is_open()) {
    throw std::runtime_error("Could not open file: " + filePath);
  }

  // Read header
  std::string header;
  std::getline(file, header);

  std::vector<std::string> columnNames;
  std::stringstream ss(header);
  std::string columnName;
  while (std::getline(ss, columnName, delimiter)) {
    columnNames.push_back(columnName);
  }

  std::cout << "-- Found " << columnNames.size() << " columns in header"
            << '\n';
  for (size_t i = 0; i < columnNames.size(); i++) {
    std::cout << "--   Column " << i << ": '" << columnNames[i] << "'" << '\n';
  }

  // Determine SMILES column(s)
  std::vector<size_t> smilesColumns;
  if (vm.count("smiles-col")) {
    const auto& requestedCols = vm["smiles-col"].as<std::vector<std::string>>();
    for (const auto& colName : requestedCols) {
      auto it = std::find(columnNames.begin(), columnNames.end(), colName);
      if (it != columnNames.end()) {
        smilesColumns.push_back(std::distance(columnNames.begin(), it));
      } else {
        throw std::runtime_error("SMILES column not found: " + colName);
      }
    }
  } else {
    // Try to auto-detect SMILES column
    std::cout << "-- Attempting to auto-detect SMILES column" << '\n';
    for (size_t i = 0; i < columnNames.size(); i++) {
      const auto& name = columnNames[i];
      if (name == "SMILES" || name == "smiles" || name == "Smiles" ||
          name == "canonical_smiles" || name == "CanonicalSMILES") {
        smilesColumns.push_back(i);
        break;
      }
    }

    if (smilesColumns.empty() && !columnNames.empty()) {
      smilesColumns.push_back(0);  // Default to first column
    }
  }

  if (smilesColumns.empty()) {
    throw std::runtime_error(
        "No SMILES columns specified and auto-detection failed");
  }

  for (auto idx : smilesColumns) {
    std::cout << "-- Auto-detected SMILES column at index: " << idx << " ('"
              << columnNames[idx] << "')" << std::endl;
  }

  // First count lines to estimate progress
  size_t lineCount = 0;
  {
    std::ifstream countFile(filePath);
    std::string line;
    std::getline(countFile, line);  // Skip header
    while (std::getline(countFile, line)) {
      lineCount++;
    }
  }

  // Configure number of threads
  int numWorkers = vm.count("mpu") ? vm["mpu"].as<int>()
                                   : std::thread::hardware_concurrency() - 2;
  numWorkers = std::max(1, numWorkers);

  // Process in chunks to avoid memory issues
  const size_t CHUNK_SIZE = 10000;  // Process 10,000 lines at a time
  std::vector<MoleculeRecord> records;
  std::mutex recordsMutex;

  std::string line;
  std::vector<std::string> chunkLines;
  size_t lineIndex = 0;
  
  std::string operationName = "Loading CSV file";
  ProgressTracker progress(operationName, lineCount);

  while (std::getline(file, line) || !chunkLines.empty()) {
    if (!line.empty()) {
      chunkLines.push_back(line);
    }

    // Process chunk when it reaches CHUNK_SIZE or at end of file
    if (chunkLines.size() >= CHUNK_SIZE || (line.empty() && !chunkLines.empty())) {
      std::vector<std::string> currentChunk = std::move(chunkLines);
      chunkLines.clear();

      parallelProcessWithProgress(
          operationName + " (chunk " + std::to_string(lineIndex) + "-" + std::to_string(lineIndex + currentChunk.size()) + ")",
          currentChunk.size(),
          numWorkers,
          false,
          [&](size_t i) {
            const std::string& currentLine = currentChunk[i];
            std::vector<std::string> values;
            std::stringstream ss(currentLine);
            std::string value;
            while (std::getline(ss, value, delimiter)) {
              values.push_back(value);
            }

            if (values.size() != columnNames.size()) {
              bool quiet = vm.count("quiet");
              if (!quiet) {
              #pragma omp critical
                std::cerr
                    << "-- Warning: Skipping line with incorrect number of columns"
                    << '\n';
              }
              return;
            }

            std::vector<MoleculeRecord> threadRecords;
            
            for (size_t smilesCol : smilesColumns) {
              if (smilesCol >= values.size()) continue;

              const std::string& smilesString = values[smilesCol];
              if (smilesString.empty()) continue;

              try {
                // Parse SMILES - use sanitize=false to catch errors in SMILES
                // handling
                std::unique_ptr<RDKit::ROMol> mol(
                    RDKit::SmilesToMol(smilesString, 0, false));
                if (mol) {
                  try {
                    // Convert to RWMol for sanitization
                    RDKit::RWMol rwmol(*mol);
                    // Now sanitize separately to catch any issues
                    unsigned int failedOp;
                    RDKit::MolOps::sanitizeMol(rwmol, failedOp);

                    // Use the sanitized molecule
                    MoleculeRecord record;
                    record.mol = std::make_shared<RDKit::ROMol>(rwmol);

                    // Store all columns as properties
                    for (size_t j = 0; j < values.size(); j++) {
                      record.properties[columnNames[j]] = values[j];
                    }

                    threadRecords.push_back(record);
                  } catch (const std::exception& e) {
                    if (!vm.count("quiet")) {
                    #pragma omp critical
                      std::cerr << "-- Warning: Molecule failed sanitization: "
                                << smilesString << " (" << e.what() << ")" << '\n';
                    }
                  }
                }
              } catch (const std::exception& e) {
                if (!vm.count("quiet")) {
                #pragma omp critical
                  std::cerr << "-- Warning: Failed to parse SMILES: "
                            << smilesString << " (" << e.what() << ")" << '\n';
                }
              }
            }

            if (!threadRecords.empty()) {
            #pragma omp critical
              {
                records.insert(records.end(), threadRecords.begin(), threadRecords.end());
              }
            }
            
            #pragma omp critical
            {
              progress.update();
            }
          }
      );

      // Update dataset periodically to release memory
      if (records.size() > CHUNK_SIZE * 2) {
        dataset.insert(dataset.end(), records.begin(), records.end());
        records.clear();
      }
      
      lineIndex += currentChunk.size();
    }

    line.clear();  // Ensure line is empty for the next iteration
  }

  // Add any remaining records
  dataset.insert(dataset.end(), records.begin(), records.end());

  progress.finish();
  std::cout << "-- Loaded " << dataset.size() << " molecules" << '\n';
  return dataset;
}

MoleculeDataset DataHandler::loadSmiles(const po::variables_map& vm) {
  std::string smiles;
  try {
    smiles = vm["smiles"].as<std::string>();
  } catch (const boost::bad_any_cast& e) {
    throw std::runtime_error("Invalid SMILES parameter: " +
                             std::string(e.what()));
  }

  MoleculeDataset dataset;

  if (smiles.empty()) {
    throw std::runtime_error("Empty SMILES string provided");
  }

  try {
    RDKit::ROMol* mol = RDKit::SmilesToMol(smiles);
    if (mol) {
      MoleculeRecord record;
      record.mol = std::shared_ptr<RDKit::ROMol>(mol);
      record.properties["SMILES"] = smiles;
      dataset.push_back(record);
      std::cout << "-- Molecule loaded from SMILES with " << mol->getNumAtoms()
                << " atoms" << '\n';
    } else {
      throw std::runtime_error("Failed to parse SMILES string");
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("Error parsing SMILES: " + std::string(e.what()));
  }

  return dataset;
}

void DataHandler::saveData(MoleculeDataset& dataset,
                           const po::variables_map& vm) {
  std::string outputPath = vm["output"].as<std::string>();
  std::string format = "auto";

  if (vm.count("output-format")) {
    format = vm["output-format"].as<std::string>();
  } else {
    // Attempt to detect format from file extension
    size_t dotPos = outputPath.find_last_of('.');
    if (dotPos != std::string::npos) {
      std::string ext = outputPath.substr(dotPos + 1);
      if (ext == "sdf")
        format = "sdf";
      else if (ext == "smi")
        format = "smi";
      else if (ext == "csv")
        format = "csv";
      else if (ext == "tsv")
        format = "tsv";
    }
  }

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

  std::cout << "-- Writing " << dataset.size() << " molecules to " << outputPath
            << " using " << numWorkers << " worker threads" << '\n';

  std::string operationName = "Writing molecules";
  ProgressTracker progress(operationName, dataset.size());

  if (format == "sdf") {
    RDKit::SDWriter writer(outputPath);
    
    // Write molecules sequentially to avoid errors
    for (size_t i = 0; i < dataset.size(); i++) {
      if (dataset[i].mol) {
        try {
          // Create a copy and add properties
          RDKit::ROMol mol(*dataset[i].mol);
          
          // Add properties
          for (const auto& prop : dataset[i].properties) {
            mol.setProp(prop.first, prop.second);
          }
          
          // Write directly without using string serialization
          writer.write(mol);
          writer.flush();
        } catch (const std::exception& e) {
          if (!vm.count("quiet")) {
            std::cerr << "-- WARNING: Failed to write molecule " << i << ": " << e.what() << std::endl;
          }
        }
      }
      progress.update();
    }
    
    writer.close();
  } else if (format == "smi") {
    // Prepare SMILES strings in parallel
    std::vector<std::string> smilesLines(dataset.size());

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < dataset.size(); i++) {
      if (dataset[i].mol) {
        std::stringstream ss;
        ss << RDKit::MolToSmiles(*dataset[i].mol);
        for (const auto& prop : dataset[i].properties) {
          ss << "\t" << prop.second;
        }
        smilesLines[i] = ss.str();
      }
    }

    // Write to file sequentially
    std::ofstream file(outputPath);
    for (size_t i = 0; i < dataset.size(); i++) {
      if (!smilesLines[i].empty()) {
        file << smilesLines[i] << '\n';
      }
      progress.update();
    }
  } else if (format == "csv" || format == "tsv") {
    char delimiter = (format == "csv") ? ',' : '\t';

    // Collect all property names
    std::set<std::string> allProps;
#ifndef NO_OPENMP
#pragma omp parallel
#endif
    {
      std::set<std::string> threadProps;
#ifndef NO_OPENMP
#pragma omp for nowait
#endif
      for (size_t i = 0; i < dataset.size(); i++) {
        for (const auto& prop : dataset[i].properties) {
          threadProps.insert(prop.first);
        }
      }

#ifndef NO_OPENMP
#pragma omp critical
#endif
      {
        allProps.insert(threadProps.begin(), threadProps.end());
      }
    }

    // Remove "SMILES" from properties list since it's already included as the
    // first column
    allProps.erase("SMILES");

    // Write header
    std::ofstream file(outputPath);
    file << "SMILES";

    std::vector<std::string> propNames(allProps.begin(), allProps.end());
    for (const auto& propName : propNames) {
      file << delimiter << propName;
    }
    file << '\n';

    // Prepare data lines in parallel
    std::vector<std::string> dataLines(dataset.size());

#ifndef NO_OPENMP
#pragma omp parallel for
#endif
    for (size_t i = 0; i < dataset.size(); i++) {
      if (dataset[i].mol) {
        std::stringstream ss;
        ss << RDKit::MolToSmiles(*dataset[i].mol);

        // Write properties
        for (const auto& propName : propNames) {
          ss << delimiter;
          auto it = dataset[i].properties.find(propName);
          if (it != dataset[i].properties.end()) {
            ss << it->second;
          }
        }
        dataLines[i] = ss.str();
      }
    }

    // Write data lines sequentially
    for (size_t i = 0; i < dataset.size(); i++) {
      if (!dataLines[i].empty()) {
        file << dataLines[i] << '\n';
      }
      progress.update();
    }
  } else {
    throw std::runtime_error("Unsupported output format: " + format);
  }

  progress.finish();
  std::cout << "-- Successfully wrote data to " << outputPath << '\n';
}

std::string DataHandler::getFileExtension(const std::string& filename) {
  size_t pos = filename.find_last_of(".");
  if (pos != std::string::npos) {
    std::string ext = filename.substr(pos + 1);
    boost::to_lower(ext);
    return ext;
  }
  return "";
}

void DataHandler::splitOutput(MoleculeDataset& dataset, const std::string& outputPath, const std::string& splits) {
  std::cout << "-- Splitting output data" << std::endl;
  
  // Parse splits parameter (should be comma-separated numbers)
  std::vector<float> splitRatios;
  std::stringstream ss(splits);
  std::string item;
  float total = 0.0f;
  
  while (std::getline(ss, item, ',')) {
    try {
      float ratio = std::stof(item);
      total += ratio;
      splitRatios.push_back(ratio);
    } catch (const std::exception& e) {
      std::cerr << "-- ERROR: Invalid split ratio: " << item << std::endl;
      return;
    }
  }
  
  if (splitRatios.size() < 2) {
    std::cerr << "-- ERROR: At least two split ratios are required" << std::endl;
    return;
  }
  
  // Normalize ratios to sum to 1.0
  for (auto& ratio : splitRatios) {
    ratio /= total;
  }
  
  // Get base file path and extension
  std::string basePath = outputPath;
  std::string extension = ".csv"; // Default
  
  // Create split datasets
  size_t datasetSize = dataset.size();
  std::vector<MoleculeDataset> splitDatasets(splitRatios.size());
  
  // Shuffle indices
  std::vector<size_t> indices(datasetSize);
  for (size_t i = 0; i < datasetSize; ++i) {
    indices[i] = i;
  }
  std::random_device rd;
  std::mt19937 g(rd());
  std::shuffle(indices.begin(), indices.end(), g);
  
  // Assign molecules to splits
  size_t currentIndex = 0;
  for (size_t splitIdx = 0; splitIdx < splitRatios.size(); ++splitIdx) {
    size_t splitSize = static_cast<size_t>(splitRatios[splitIdx] * datasetSize);
    if (splitIdx == splitRatios.size() - 1) {
      // Ensure all remaining molecules are added to the last split
      splitSize = datasetSize - currentIndex;
    }
    
    for (size_t i = 0; i < splitSize && currentIndex < datasetSize; ++i, ++currentIndex) {
      splitDatasets[splitIdx].push_back(dataset[indices[currentIndex]]);
    }
  }
  
  // Save split datasets
  std::vector<std::string> splitNames = {"train", "test", "validation"};
  for (size_t i = 0; i < splitDatasets.size(); ++i) {
    std::string splitName = i < splitNames.size() ? splitNames[i] : "split" + std::to_string(i);
    std::string outputFilePath = basePath + "_" + splitName + extension;
    
    po::variables_map tempVm;
    tempVm.insert(std::make_pair("output", po::variable_value(boost::any(outputFilePath), false)));
    tempVm.insert(std::make_pair("output-format", po::variable_value(boost::any(std::string("csv")), false)));
    
    saveData(splitDatasets[i], tempVm);
    std::cout << "-- Created " << splitName << " dataset with " << splitDatasets[i].size() << " molecules" << std::endl;
  }
  
  std::cout << "-- Split operation complete" << std::endl;
}