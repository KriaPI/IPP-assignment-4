#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <omp.h>

bool isPrime  (int num) {
    if(num <= 1) return false;
    for(int i=2; i<=std::sqrt(num); i++) {
        if(num % i == 0) return false;
    }
    return true;
}

int main() {

    int maxValue, it, threadCount;
    std::vector<int> seeds;

    std::cout << "Enter the maximum value: ";
    std::cin >> maxValue;

    for(it=2; it<=std::sqrt(maxValue); ++it) {
        if(isPrime(it)) {
            seeds.push_back(it);
        }
    }

    std::cout << "\nSerially computed seed prime values: ";
    for(int num: seeds) std::cout << num << " ";

    std::cout <<"\nEnter the thread count: ";
    std::cin >> threadCount;

    std::vector<char> marked(maxValue + 1, 0);

    int sqrtMaxValue = std::sqrt(maxValue);

    int parallelLowerBound = sqrtMaxValue + 1;
    int parallelUpperBound = maxValue+1;

    int chunkSize = (parallelUpperBound - parallelLowerBound) / threadCount;

    int remainingSize = (parallelUpperBound - parallelLowerBound) % threadCount;

    int currentLowerBound = parallelLowerBound;

    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(threadCount) \
        shared(marked, seeds)
    {
        int id = omp_get_thread_num();

        // Assign nearly equal, non-overlapping chunks
        int lowerBound =
            parallelLowerBound +
            id * chunkSize +
            std::min(id, remainingSize);

        int localChunkSize =
            chunkSize + (id < remainingSize ? 1 : 0);

        int upperBound = lowerBound + localChunkSize;

        for (int seed : seeds) {
            int correctedLowerBound =
                std::max(lowerBound, seed * seed);

            int remainder = correctedLowerBound % seed;

            int firstMultiple =
                correctedLowerBound - remainder
                + seed * (remainder != 0);

            for (int multiple = firstMultiple;
                 multiple < upperBound;
                 multiple += seed) {
                marked[multiple] = 1;
            }
        }
    }

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
    end - start
    );

    std::cout << "\n\nParallelly computed prime values: ";
    for(int num=sqrtMaxValue+1; num<=maxValue; ++num) {
        if(!marked[num]) std::cout << num << " ";
    }

    std::cout << "\nExecution time: " << duration.count() << " ms\n";

    return 0;

}