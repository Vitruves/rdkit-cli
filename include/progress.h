#pragma once

#include <iostream>
#include <mutex>
#include <atomic>
#include <iomanip>
#include <string>
#include <utility>
#include <vector>
#include <chrono>
#include <functional>
#include <sstream>

class ProgressTracker {
private:
    std::string taskName;
    size_t totalItems;
    std::atomic<size_t> processedItems{0};
    std::mutex printMutex;
    std::chrono::steady_clock::time_point startTime;
    double lastReportedPercentage = -1.0;
    bool verbose;
    
    // Report frequency controls
    static constexpr double minProgressStep = 0.01; // 0.01% minimum step
    
public:
    ProgressTracker(const std::string& operationName, size_t total, bool isVerbose = false) 
        : taskName(operationName), totalItems(total), verbose(isVerbose) {
        startTime = std::chrono::steady_clock::now();
    }
    
    // Update progress by given increment (default is 1)
    void update(size_t increment = 1) {
        size_t current = processedItems.fetch_add(increment) + increment;
        
        // Only report if we've reached a new percentage point (at 0.01% granularity)
        double percentage = (static_cast<double>(current) / totalItems) * 100.0;
        
        // Only acquire lock and print if we've made enough progress
        if (percentage - lastReportedPercentage >= minProgressStep) {
            std::lock_guard<std::mutex> lock(printMutex);
            
            // Check again to avoid race conditions where multiple threads reached this point
            if (percentage - lastReportedPercentage >= minProgressStep) {
                // Calculate ETA
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(now - startTime).count() / 1000.0;
                double itemsPerSecond = current / elapsed;
                double etaSeconds = (totalItems - current) / itemsPerSecond;
                
                // Use carriage return to update the same line
                if (verbose) {
                    std::cout << "\r-- " << taskName << " [" 
                              << std::fixed << std::setprecision(2) << std::setw(6) << percentage 
                              << "%] " << current << "/" << totalItems 
                              << " - " << std::setprecision(1) << itemsPerSecond << " items/s"
                              << " - ETA: " << formatTime(etaSeconds) << std::flush;
                } else {
                    std::cout << "\r-- " << taskName << " [" 
                              << std::fixed << std::setprecision(2) << std::setw(6) << percentage 
                              << "%]" << std::flush;
                }
                
                lastReportedPercentage = percentage;
            }
        }
    }
    
    // Mark the operation as completed
    void finish() {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now() - startTime).count() / 1000.0;
        
        std::lock_guard<std::mutex> lock(printMutex);
        // Print a newline first to move to the next line after the progress updates
        std::cout << "\r-- " << taskName << " [100.00%] - Completed in " 
                  << formatTime(elapsed) << '\n';
    }
    
private:
    static std::string formatTime(double seconds) {
        int hrs = static_cast<int>(seconds) / 3600;
        int mins = (static_cast<int>(seconds) % 3600) / 60;
        int secs = static_cast<int>(seconds) % 60;
        
        std::stringstream ss;
        if (hrs > 0) {
            ss << hrs << "h ";
        }
        if (mins > 0 || hrs > 0) {
            ss << mins << "m ";
        }
        ss << secs << "s";
        
        return ss.str();
    }
};

// Helper function for parallel processing with progress reporting
void parallelProcessWithProgress(
    const std::string& operationName,
    size_t itemCount,
    int numThreads,
    bool verbose,
    std::function<void(size_t)> processFunction
); 