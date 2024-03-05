# Hokuto_AES
Aes-128 ecb_GPU Buteforcer
User Manual for Hokuto Program

I hope I haven't made a mistake by consolidating 
all my work for the program's output,
any verification is welcome.


--------------------------------------------------------


Remember that debug functions in the console are useless for brute force attempts,
they significantly slow down the process, so use them solely for debugging.


-----------------------------------------------



Getting Started

Before you begin using the Hokuto program, 
it is crucial to configure a specific variable within the program to match your hardware setup. 
This configuration ensures optimal performance and compatibility with your graphics processing unit (GPU).

Configuration Steps

1. Set Up the Variable: 
Locate the `const int n` variable within the `int main()` function. 
This variable is essential for tuning the program to work efficiently with your GPU.

2. Determine the Correct Value: 
To set the `const int n` variable appropriately, 
you need to understand the capabilities of your GPU. 
Execute the following command in a terminal to retrieve detailed information about your GPU:

C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\extras\demo_suite\deviceQuery.exe
   

This command will output various details about your GPU, including its model, CUDA cores, memory, and more. For example, a device query output might look like this:

  Device 0: "NVIDIA GeForce RTX 4090 Laptop GPU"
  CUDA Driver Version / Runtime Version          12.4 / 11.8
  CUDA Capability Major/Minor version number:    8.9
  Total amount of global memory:                 16376 MBytes (17170956288 bytes)
  MapSMtoCores for SM 8.9 is undefined.  Default to use 128 Cores/SM
  MapSMtoCores for SM 8.9 is undefined.  Default to use 128 Cores/SM
  (76) Multiprocessors, (128) CUDA Cores/MP:     9728 CUDA Cores
  GPU Max Clock rate:                            2040 MHz (2.04 GHz)
  Memory Clock rate:                             9001 Mhz
  Memory Bus Width:                              256-bit
  L2 Cache Size:                                 67108864 bytes
  Maximum Texture Dimension Size (x,y,z)         1D=(131072), 2D=(131072, 65536), 3D=(16384, 16384, 16384)
  Maximum Layered 1D Texture Size, (num) layers  1D=(32768), 2048 layers
  Maximum Layered 2D Texture Size, (num) layers  2D=(32768, 32768), 2048 layers
  Total amount of constant memory:               zu bytes
  Total amount of shared memory per block:       zu bytes
  Total number of registers available per block: 65536
  Warp size:                                     32
  Maximum number of threads per multiprocessor:  1536
  Maximum number of threads per block:           1024

3. Calculate the `n` Value: 

The value of `n` is determined by the number of Streaming Multiprocessors (SMs) 
on your GPU and the maximum number of threads per SM. 
For example, with 76 Multiprocessors and 1536 threads per SM, calculate `n` as follows:
  
   n = Number of SMs × Maximum number of threads per SM
   n = 76 × 1536  
   This calculation results in `n = 116,736`, which is the optimal number of threads for the program given the example hardware configuration.


Program Modes and Debugging

-Sequential vs. Random Generation: 
The program initiates with a randomly generated key, 
serving as the starting point for subsequent sequential key generation. 
This approach has been found to improve processing speed by up to 400% compared to purely random generation.

- Timing Debug Mode: 
Insert `#define TIMER 1` in your code 
to measure the time taken (in milliseconds) to process a set of 116,736 keys (your "n" value).

 
Note that this mode is intended for debugging purposes and can slow down the program 
due to CPU/GPU synchronization for console output.

- Debug Mode: 
Insert `#define DEBUG_MODE 1` 
to display the cipher, the generated key, and the decrypted cipher 
(which will be compared to the target plaintext) in the console. 

Note Like the timing debug mode, this significantly slows down the program and should only be used for debugging.



Building and Running the Program

- Build the Program: Compile the program using the NVIDIA CUDA Compiler (nvcc) with the following command:

  
  nvcc -o Hokuto Hokuto.cu
  

- Start the Program: Run the compiled program by executing:

  
  ./Hokuto
  



 Additional Notes

- Multi-GPU: Currently, the program has not been tested on multi-GPU setups. Because I simply don't have several.

- Program Logic: The program logic is not guaranteed.  Community contributions to improve the logic are welcome.

- Efficiency: While my program contains some inefficiencies, 
it embodies my best effort. I see myself more as a random Ninja than a programmer.

