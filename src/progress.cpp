#include "../include/progress.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <thread>
#include <sstream>

#ifndef NO_OPENMP
#include <omp.h>
#endif

void parallelProcessWithProgress(const std::string& operationName, size_t itemCount, int numThreads, bool verbose, std::function<void(size_t)> processFunction)
{
    ProgressTracker progress(operationName, itemCount, verbose);
    
#ifndef NO_OPENMP
    #pragma omp parallel for num_threads(numThreads) schedule(dynamic)
#endif
    for(size_t i = 0; i < itemCount; i++) {
        processFunction(i);
        
#ifndef NO_OPENMP
        #pragma omp critical
#endif
        {
            progress.update(1);
        }
    }
    
    progress.finish();
} 