#include <mpi.h>
#include <mpi_proto.h>

#include <algorithm>
#include <format>
#include <iostream>
#include <numeric>
#include <vector>

// TODO: add timing for benchmarking

void getMaxValue(int& rank, int& worldSize, int& maxValue) {
    if (rank == 0) {
        std::cout << "Enter the maximum value: ";
        std::cin >> maxValue;

        // Send maxValue to all processes.
        for (int destination = 1; destination < worldSize; ++destination) {
            MPI_Send(&maxValue, 1, MPI_INT, destination, 0, MPI_COMM_WORLD);
        }
    } else {
        // recieve
        MPI_Recv(&maxValue, 1, MPI_INT, 0, 0, MPI_COMM_WORLD,
                 MPI_STATUS_IGNORE);
    }
}

int main() {
    MPI_Init(nullptr, nullptr);

    int rank{0};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int worldSize{0};
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

    int maxValue{};
    getMaxValue(rank, worldSize, maxValue);
    std::cout << std::format("process {} has maxvalue {}\n", rank, maxValue);

    auto localChunkSize =
        rank == 0 ? maxValue / worldSize - 2 : maxValue / worldSize;
    if (rank == worldSize - 1) {
        localChunkSize += maxValue % worldSize;
    }
    auto localLowerBound = rank == 0 ? 2 : (maxValue / worldSize) * rank;
    auto localUpperBound = rank == worldSize - 1
                               ? localLowerBound + localChunkSize
                               : localLowerBound + localChunkSize - 1;

    std::cout << std::format(
        "Local chunk size, lower bound, upper bound: {}, {}, {}\n",
        localChunkSize, localLowerBound, localUpperBound);

    std::vector<int> seeds(localChunkSize);
    std::iota(seeds.begin(), seeds.end(), localLowerBound);
    std::vector<char> marked(localChunkSize, 0);
    std::cout << "Marked size: " << marked.size() << "\n";

    for (int seed : seeds) {
        int correctedLowerBound = std::max(localLowerBound, seed * seed);
        std::cout << "Corrected Lower Bound: " << correctedLowerBound << std::endl;
        int remainder = correctedLowerBound % seed;
        int firstMultiple = correctedLowerBound - remainder + seed * (remainder != 0);

        std::cout << "First multiple: " << firstMultiple << std::endl;

        for (int multiple = firstMultiple; multiple < localUpperBound; multiple += seed) {
            std::cout << "Multiple: " << multiple << std::endl;
            marked[multiple - 1] = 1;
        }
    }

    std::vector<int> primes;

    auto markedIt{marked.begin()};
    auto seedIt{seeds.begin()};
    for (; markedIt != marked.end() && seedIt != seeds.end(); ++markedIt, ++seedIt) {
        if (*markedIt == 0) {
            std::cout << std::format("prime {}\n", *seedIt);
            primes.push_back(*seedIt);
        }
    }

    if (rank != 0) {
        // TODO: make a vector of prime numbers.
        // TODO: send prime numbers to root process.
        MPI_Send(primes.data(), primes.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
    } else {
        // TODO: print out primes in root process.
        for (int source = 1; source < worldSize; ++source) {
            // TODO: probe and find out how many values are sent.
            // TODO: reserve vector space
            // TODO: receive and store in the appropriate place in the vector.
            MPI_Status status{};
            int count{0};
            MPI_Probe(source, 0, MPI_COMM_WORLD, &status);
            MPI_Get_count(&status, MPI_INT, &count);
            primes.reserve(primes.size() + count);

            std::cout << "Primes Get Count: " << count << std::endl;
            MPI_Recv(primes.data() + primes.size(), count, MPI_INT, source, 0,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }

    if (rank == 0) {
        std::cout << "\n\nParallelly computed prime values: ";
        std::cout << std::format("Prime count {}\n", primes.size());

        for (auto& prime : primes) {
            std::cout << std::format("{}, ", prime);
        }
    }

    MPI_Finalize();
    return 0;
}