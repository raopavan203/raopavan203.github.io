## Checkpoint Status Update

- ***Work completed so far:***
1. We have studied basic Rabin fingerprinting algorithm for chunking and understood its intricacies.
2. We have come up with a basic parallel algorithm for Rabin fingerprinting, which can be used for both multi-core as well as GPU versions. Need to implement it.
3. We are waiting on some dependency libraries of CloudFS (libtar-dev, libs3-dev) to be installed on GHC machines so that we can start working on our CUDA versions.
4. We have written a basic test framework which invokes the baseline sequential version of CloudFS and can invoke the parallel versions going forward. The test spawns CloudFS and runs a bunch of file writes (large and small) against the file system. Some of them have high data duplication and others do not. They will help us evaluate the usecases where the system is expected to improve as compared to others where it will not improve. It measures the end-to-end execution time for the writes, execution time for chunking using Rabin fingerprinting and also measures the write throughput of the file system for these requests. It also computes the speedup for CUDA version wrt sequential and multicore versions.

## Schedule

- April 10 - April 16 : 
1. Understand how Rabin fingerprinting algorithm works: DONE
2. Benchmark the current cloudFS starter implementation: Waiting for installation GHC machines

- April 17 - April 25 : 
1. Write a test framework to evaluate speedup of the parallel version: DONE
2. Update checkpoint write-up

- April 26 - April 28 : 
1. Parallelise the Rabin fingerprinting operation across multiple cores of a CPU
2. Observe the performance improvement

- May 2 - May 5:
1. Start parallelizing Rabin fingerprinting using one GPU
2. Complete parallelization of the Rabin fingerprinting using one GPU

- May 6 - May 8 : 
1. Benchmark and explore additional optimizations while integrating with the cloudFS code.
2. Parallelize across multiple GPUs
3. Analysis and evaluation of results

- May 9 - May 11 : 
1. Work on the future goals
2. Finish writeup. Make the project ready for handin.
