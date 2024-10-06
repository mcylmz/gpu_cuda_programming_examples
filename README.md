##  HARDWARE

### Central Processing Unit (CPU)
- General purpose
- High clock speed
- Few cores
- High cache
- Low latency
- Low throughput

### Graphics Processing Unit (GPU)
- Specialized
- Low clock speed
- Many cores
- Low cache
- High latency
- High throughput

### Tensor Processing Unit (TPU)
- Specialized GPUs for deep learning algoritms

### Field Programmable Gate Array
- Specialized hardware that can be reconfigured to perform specific tasks
- Very low latency
- Very high throughput
- Very high power consumption
- Very high cost

## CPU (host)
- minimize time of one task
- metric: latency in seconds

## GPU (device)
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

## CUDA Compute Capability
CUDA Compute Capability is a versioning system used by NVIDIA to specify the features and capabilities of its GPU architectures. Each GPU is assigned a compute capability version, which indicates the hardware's ability to execute certain CUDA functions and features.

### Key Points about Compute Capability
1. **Versioning:** Compute capability is denoted as a major and minor version number (e.g., 6.1, 7.5). The major version indicates the architecture generation, while the minor version provides additional details about features.

2. **Feature Set:** Different compute capabilities support different features. For example
* Support for specific data types (e.g., FP16, INT8).
* Number of threads per block.
* Memory architecture and bandwidth.
* Availability of shared memory and registers.

3. **Backward Compatibility:** Code written for a lower compute capability can often run on a higher compute capability, but the reverse isn't always true. If you use features specific to a higher compute capability, your code may not work on older GPUs.

4. **Choosing Compute Capability:** When compiling CUDA code, you often specify the target compute capability to ensure optimal performance on specific hardware. You can set this in the compilation command using -gencode arch=compute_xx,code=sm_xx, where xx corresponds to the major and minor version numbers.

5. **Common Compute Capabilities:**
* **3.0:** Kepler architecture.
* **5.0:** Maxwell architecture.
* **6.0:** Pascal architecture.
* **7.0:** Volta architecture.
* **8.0:** Ampere architecture.

## Device vs Host Naming Scheme
**__global__** keyword if a function qualifier that indicates that the function it precedes
is a **kernel**. Kernels are functions that are executed on the GPU and can be called from
the host(CPU) code.

**__device__** qualifier is used to indicate that a function or a variable is intended
to be executed or accessed on the GPU. They cannot be called (function) from host (CPU)
code directly.

**__host__** qualifier indicates that a function or variable is intended to be
executed or accessed on the host (CPU).

## Thread, Block, Warp and Grid
1. **Thread**
**Definition:** A thread is the smallest unit of execution in a CUDA program. Each thread runs a kernel (a function defined with the __global__ qualifier) independently and can execute on the GPU.
**Unique Identifiers:** Each thread has a unique identifier, which can be accessed using built-in variables like threadIdx (for the thread’s position within its block) and blockIdx (for the block’s position within the grid).

2. **Block**
**Definition:** A block is a group of threads that can cooperate among themselves via shared memory and can synchronize their execution. Threads within a block can share data and coordinate their execution using synchronization mechanisms.
**Block Size:** A block can have a maximum of 1024 threads (depending on the GPU architecture) and can be one-dimensional, two-dimensional, or three-dimensional in structure.
**Unique Identifiers:** Each block also has a unique identifier accessed through the blockIdx variable.

3. **Warp**
**Definition:** A warp is a collection of 32 threads (on most NVIDIA architectures) that are executed in lock-step by the GPU. When a kernel is launched, threads are grouped into warps for execution.
**Execution:** All threads in a warp execute the same instruction at any given time but can operate on different data. If threads in a warp diverge (e.g., due to conditional statements), the warp serially executes the paths taken by the diverging threads, potentially leading to reduced efficiency.

4. **Grid**
**Definition:** A grid is a collection of blocks that execute a kernel. When a kernel is launched, you specify the grid size, which determines how many blocks will be created.
**Grid Size:** The grid can also be one-dimensional, two-dimensional, or three-dimensional, depending on the problem being solved. The total number of threads launched is the product of the number of blocks and the number of threads per block.

**Summary**
- Threads are grouped into blocks.
- Blocks are organized into a grid.
- Threads within the same block can share data and synchronize, while threads in different blocks cannot.
- Warps are the execution units for the GPU, with each block divided into warps (32 threads per warp).

