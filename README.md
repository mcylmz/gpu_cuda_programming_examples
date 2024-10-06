##  HARDWARE

* Central Processing Unit (CPU)
- General purpose
- High clock speed
- Few cores
- High cache
- Low latency
- Low throughput

* Graphics Processing Unit (GPU)
- Specialized
- Low clock speed
- Many cores
- Low cache
- High latency
- High throughput

* Tensor Processing Unit (TPU)
- Specialized GPUs for deep learning algoritms

* Field Programmable Gate Array
- Specialized hardware that can be reconfigured to perform specific tasks
- Very low latency
- Very high throughput
- Very high power consumption
- Very high cost

* CPU (host)
- minimize time of one task
- metric: latency in seconds

* GPU (device)
- maximize throughput
- metric: throughput in tasks per second

## Typical CUDA program
1. CPU allocates CPU memory
2. CPU copies data to GPU
3. CPU launches kernel on GPU (processing is done in here)
4. CPU copies results from GPU back to CPU to do something useful with it

## Kernel
A CUDA kernel is a function written in CUDA (Compute Unified Device Architecture), NVIDIA's parallel computing platform and application programming interface (API). Kernels are executed on the GPU, allowing developers to harness the parallel processing power of NVIDIA graphics cards.

* Key Characteristics of CUDA Kernels:
1. **Parallel Execution:** A kernel is executed by multiple threads in parallel. Each thread can process different parts of the data simultaneously, which is particularly useful for tasks like image processing or matrix operations.

2. **Launch Configuration:** When you invoke a kernel, you specify how many threads and blocks you want to use. Threads are grouped into blocks, and blocks are organized into a grid. This hierarchical organization allows for efficient memory usage and execution.

3. **Memory Hierarchy:** CUDA provides different types of memory:
* **Global Memory:** Accessible by all threads but has high latency.
* **Shared Memory:** Faster, on-chip memory shared among threads in the same block, useful for inter-thread communication.
* **Registers:** Fastest but limited in size; each thread has its own.

4. **Synchronization:** Threads within the same block can be synchronized using built-in functions, which is essential when threads need to share data or ensure they are working in lock-step.

## Some terms to remember
* kernels (GPU kernels)
* threads, blocks, and grid
* GEMM (General Matrix Multiplication)
* SGEMM (Single Precision (fp32) General Matrix Multiplication)
* cpu/host/functions vs gpu/device/kernels
* CPU is referred to as the host. It executes functions.
* GPU is referred to as the device. It executes GPU functions called kernels.

