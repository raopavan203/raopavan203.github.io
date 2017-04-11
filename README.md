## Summary

We are going to create optimized implementations of the deduplication module in CloudFS (a hybrid cloud-backed local file system, developed in 18-746: Storage Systems), that makes use of Rabin fingerprinting for chunking, on both GPU and multi-core CPU platforms, and perform a detailed analysis of both systems' performance characteristics.

### Background

  In file systems, a common way to avoid redundant computations and storage is to perform deduplication on the data. Hence, it is important to detect duplicate content during any write/update operation in the filesystem. The most effective way to detect duplicate content is to use content-based chunking. The Rabin fingerprinting algorithm is a popular content-based chunking algorithm. This involves defining chunk boundaries in a file, based on the content and not a fixed offset. This leads to variable sized chunks. A scan of the entire new/changed content is made and fingerprints are calculated over a sliding window of data. This calculation involves polynomial division of a polynomial of degree w-1 for a w bit sequence, with an irreducible polynomial of degree k. This is a very computationally intensive operation. The high computational cost is evident especially in writes of large sizes, when these large writes need to be re-chunked each time they are updated. The computation time is this case is greater than the I/O or bandwidth latency of the filesystem and it is thus, compute-bound. 
  
  Hence, content based chunking algorithm will definitely benefit from parallelism as it will directly improve the file system throughput. We plan to analyse the improvement in performance by parallelizing the chunking algorithm across multiple cores of a CPU as well as parallelizing using a GPU. Parallelization can be achieved such that each openMP or CUDA thread will work on a separate section of input and computes the Rabin fingerprint of that region in a sliding window manner. 

### Challenges

- ***Parallelization of Rabin fingerprinting***

Rabin fingerprinting is a content-based rolling hash technique, which uses sliding window mechanism to detect chunk boundaries. Currently, this is a single-threaded library implementation. It will be challenging to come up with an efficient parallel scheme for both multi-core and GPU versions for this task without causing other overheads.

- ***Hiding CPU-GPU data transfer overheads***

When deduplication computations are offloaded to GPU, the file system will need to transfer data to be chunked to the GPU memory and results back to CPU memory, per request. The cudaMemcpy overhead can potentially undo the speedup obtained by using GPU acceleration. This data-transfer latency can be hidden by overlapping the transfer of input data to GPU with computation step of a previous task. Efficiently designing the CPU-GPU interface to overlap these operations will be a challenging task. 

- ***Hiding GPU memory allocation overheads***

Moreover, we need to use non-pageable host memory for DMA between host and device. Since allocating non-pageable host memory is an expensive operation, it makes sense to reuse the buffers used to transport the GPU input data. This poses a challenging task of efficiently managing the non-pageable memory buffers by ensuring that the expensive cudaMalloc and cudaFree calls are amortized across the lifetime of the file system process.

- ***Efficient use of GPU shared memory***

Since the GPU shared memory is much smaller than the amount of data to be processed during Rabin fingerprinting, efficiently using this limited amount of shared memory will be a challenge. Moreover, shared memory is divided into banks and all reads and writes that access the same bank are serialized by the GPU memory subsystem, which would limit potential parallelism. Thus, enforcing the concurrent CUDA threads to access data on different banks will be a challenging task. In addition, trading off the use of high number of threads to hide latency, for amount of shared memory available per thread is another challenge we foresee.

### Resources

- ***Hardware***
8-core (hyperthreaded) 3.20 GHz Intel Xeon i7 processors on GHC machines
NVIDIA GeForce GTX 1080 GPUs on GHC machines

- ***Starter code base***
CloudFS project that we developed in our 18746 (Storage Systems) class, a single threaded hybrid FUSE-based file system.

- ***Reference paper used***
Samer Al-Kiswany, Abdullah Gharaibeh, Matei Ripeanu, GPUs as Storage System Accelerators, IEEE TRANSACTIONS ON PARALLEL AND DISTRIBUTED SYSTEMS, VOL. 24, NO. 8, AUGUST 2013.

Udi Manber, Finding Similar Files in a Large File System, USENIX Winter 1994 Technical Conference Proceedings, Jan. 17-21, 1994, San Francisco, CA.


### Goals and deliverables

- ***PLAN TO ACHIEVE:***
Design compute-intensive workloads for CloudFS (for example: large size writes) and benchmark single-threaded implementation of the file system.
Develop a correct parallel implementation of the deduplication module using multi-core CPU with openMP and GPU using CUDA.
Measure speedup of the deduplication module using multi-core CPU as against the single-core CPU implementation as baseline. Also measuring speedup of the GPU implementation as against the multi-core CPU implementation.
Measure the improvement in file system throughput if any due to speedup in deduplication module.

- ***HOPE TO ACHIEVE:***
Scale the parallel dedup to a multi-GPU implementation and analyze the speedup.
Explore different chunking algorithms in place of Rabin fingerprinting and analyze speedup for them in the same multi-core and multi-GPU environments.
Study the behavior under various workloads - small writes, snapshots, data with no duplication, data with a lot of duplication etc.

- ***DEMO/RESULTS:***
Speedup graphs for deduplication module performance in case of both multi-core version as well as GPU version.
Graphs for file system throughput changes under different workloads for the two versions.
Results of analysis regarding the type of workloads benefited by multi-core parallelism and those benefited by GPU parallelism.

### Platform Choice

CloudFS is implemented in C++. We will be using the CUDA platform to work with the multiple NVIDIA GeForce GTX 1080 GPUs on GHC machines. For the CPU implementation, we will use the 8-core (hyperthreaded) 3.20 GHz Intel Xeon i7 processors on GHC machines. We will use OpenMP for CPU version of multi-core parallelism. The i7 processor can be used as a  benchmark to analyze multi-core CPU performance and then further compare it with the multi-GPU performance. We have chosen these systems to leverage parallelism in the computationally bound components (namely deduplication) of the file system and achieve good utilization of resources. We will use writes of large sizes to simulate the compute-bound file system workloads issued such that the rate of requests being served (file system throughput) will be bound by the processing in deduplication module, while disk and network latencies will no longer be an overhead.


### Schedule

- April 10 - April 16 : 
1. Understand how Rabin fingerprinting algorithm works.
2. Benchmark the current cloudFS starter implementation
- April 17 - April 23 : 
1. Parallelise the Rabin fingerprinting operation across multiple cores of a CPU
2. Observe the performance improvement
3. Start parallelizing Rabin fingerprinting using one GPU
- April 24 - April 30 : 
1. Complete parallelization of the Rabin fingerprinting using one GPU
2. Benchmark and explore additional optimizations while integrating with the cloudFS code.
- May 1 - May 7 : 
1. Parallelize across multiple GPUs
2. Analysis and evaluation of results
- May 8 - May 11 : 
1. Work on the future goals
2. Finish writeup. Make the project ready for handin.

```markdown
Syntax highlighted code block

# Header 1
## Header 2
### Header 3

- Bulleted
- List

1. Numbered
2. List

**Bold** and _Italic_ and `Code` text

[Link](url) and ![Image](src)
```

For more details see [GitHub Flavored Markdown](https://guides.github.com/features/mastering-markdown/).

### Jekyll Themes

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/raopavan203/raopavan203.github.io/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.

### Support or Contact

Having trouble with Pages? Check out our [documentation](https://help.github.com/categories/github-pages-basics/) or [contact support](https://github.com/contact) and we’ll help you sort it out.
