#include <mpi.h>
#include <algorithm>
#include <cmath>
#include <format>
#include <iostream>
#include <vector>

bool isPrime(int num) {
    if (num <= 1) return false;
    for (int i = 2; i <= std::sqrt(num); i++) {
        if (num % i == 0) return false;
    }
    return true;
}

void getMaxValue(int rank, int& maxValue) {
    if (rank == 0) {
        std::cout << "Enter the maximum value: ";
        std::cin >> maxValue;
    }

    // Send maxValue to all processes
    MPI_Bcast(&maxValue, 1, MPI_INT, 0, MPI_COMM_WORLD);
}

int main() {
    MPI_Init(nullptr, nullptr);

    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int worldSize{0};
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int maxValue {0};
    getMaxValue(rank, maxValue);
    int sqrtMaxValue = std::sqrt(maxValue);

    // primes up to sqrt, computed on root then shared
    std::vector<int> seeds;
    if (rank == 0) {
        for (int it = 2; it <= sqrtMaxValue; ++it) {
            if (isPrime(it)) {
                seeds.push_back(it);
            }
        }
    }

    int seedCount = static_cast<int>(seeds.size());
    MPI_Bcast(&seedCount, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) {
        seeds.resize(seedCount);
    }
    MPI_Bcast(seeds.data(), seedCount, MPI_INT, 0, MPI_COMM_WORLD);

    int parallelLowerBound = sqrtMaxValue + 1;
    int parallelUpperBound = maxValue + 1;

    int range = parallelUpperBound - parallelLowerBound;
    int chunkSize = range / worldSize;
    int remainingSize = range % worldSize;

    int localLowerBound =
        parallelLowerBound + rank * chunkSize + std::min(rank, remainingSize);
    int localChunkSize = chunkSize + (rank < remainingSize ? 1 : 0);
    int localUpperBound = localLowerBound + localChunkSize;


    std::vector<char> marked(localChunkSize, 0);

    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();

    for (int seed : seeds) {
        int correctedLowerBound = std::max(localLowerBound, seed * seed);
        if (correctedLowerBound >= localUpperBound) continue;

        int remainder = correctedLowerBound % seed;
        int firstMultiple =
            correctedLowerBound - remainder + seed * (remainder != 0);

        for (int multiple = firstMultiple; multiple < localUpperBound;
             multiple += seed) {
            marked[multiple - localLowerBound] = 1;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double elapsedMs = (MPI_Wtime() - start) * 1000.0;

    std::vector<int> primes;
    if (rank == 0) {
        primes = seeds;
    }

    for (int i = 0; i < localChunkSize; ++i) {
        if (marked[i] == 0) {
            primes.push_back(localLowerBound + i);
        }
    }

    if (rank != 0) {
        MPI_Send(primes.data(), primes.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        for (int source = 1; source < worldSize; ++source) {
            MPI_Status status{};
            int count{0};
            MPI_Probe(source, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &count);

            int offset = static_cast<int>(primes.size());
            primes.resize(offset + count);

            MPI_Recv(primes.data() + offset, count, MPI_INT, source, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if (rank == 0) {
        std::cout << std::format("Primes: {} \n\n", primes.size());
        for (auto& prime: primes) {
            std::cout << std::format("{}, ", prime);
        }

        std::cout << "\nExecution time: " << elapsedMs << " ms\n";
    }

    MPI_Finalize();
    return 0;
}