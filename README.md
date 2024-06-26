# FLASHNN : A Triton-Powered Kernel Library for LLM Serving

FLASHNN is a pioneering kernel library for Large Language Models (LLMs), providing a high-performance implementation of GPU kernels optimized for LLM serving and inference, involving in comprehensive support for attention kernels and versatile quantization methods. 

By harnessing the power of Triton, FLASHNN is engineered to seamlessly integrate with multiple of hardware platforms, ensuring smooth operability and maximizing the utilization of hardware resources.



## Features

- **Comprehensive Support for Attention Kernels**: FLASHNN offers extensive support for various types of attention mechanisms, enabling it to handle a wide array of LLM architectures with ease. 
- **Multiple Quantization Methods**: FLASHNN incorporates multiple quantization techniques (int8, int4) aimed at optimizing both the computational overhead and the memory footprint of LLMs, making it easier to deploy LLMs in resource-constrained environments.
- **Runtime Overhead**：The primary contributor to the performance discrepancy observed with Triton kernels is the runtime overhead. To address this, we have implemented an ahead-of-time kernel cache for Triton kernels, which significantly mitigates this overhead.
- **Production-Ready Performance**: FLASHNN is meticulously optimized for production scenarios, which delivers state-of-art performance that meets the demanding requirements of real-world applications.
- **Smooth Portability on Multiple Hardware**: Facilitated by the inherent design of the Triton language, FLASHNN simplifying the process of adapting LLM serving solutions to diverse computing environments.


## **Compatibility**

### **Supported Operators**

| Type         | Operators                                               |
| ------------ | ------------------------------------------------------- |
| Gemm         | A8W8, A16W4, A16W8                                      |
| Attention    | PagedAttention V1, PagedAttention V2, FlashAttention V2 |
| Norm         | LayerNorm, RMSNorm                                      |
| Quantization | DynamicQuant, LayerNormDequant, RMSNormDequant          |
| Embedding    | RotaryEmbedding                                         |

### **Supported Platforms**

FlashNN is tested to work in Nvidia and AMD GPUs(e.g. A100, A10, H20, MI210, ...). 

| Platforms   | float16 | float32 | bfloat16 |
| ----------- | ------- | ------- | -------- |
| NVIDIA A100 |         |         |          |
| NVIDIA A10  |         |         |          |
| NVIDIA H20  |         |         |          |
| AMD MI210   |         |         |          |

## **Get Started**

### **Requirements**

FlashNN requires Pytorch and Triton.

### **Installation**

FlashNN operators can be customized to each function-equivalent PyTorch operators by simply replace the corresponding torch function.

The binary wheel distribution(whl) will be available soon.

## Benchmarks



## License

The FLASHNN project is based on [Apache 2.0](https://github.com/AlibabaPAI/FLASHNN/blob/main/LICENSE).
