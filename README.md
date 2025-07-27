# Onnx-CMSIS-DSP-Eigen
ML, signal processing, math libs
# ONNX + CMSIS-DSP + Eigen Inference Demo (C++)

This project demonstrates running a Keras-trained model (converted to ONNX) from C++, using ONNX Runtime on Windows, and augmenting it with:
(*Note this is the next iteration of https://github.com/stevemac321/KerasToOnnx).
- **CMSIS-DSP** (compiled as a static library)
- **Eigen** (header-only)
- Basic signal processing and ML preprocessing pipelines

## Components

- 🧠 **ONNX Runtime**: For model inference
- ⚙️ **CMSIS-DSP**: ARM DSP library built as a static lib (used for FFT and signal processing)
- 📐 **Eigen**: C++ template-based linear algebra library (used for matrix operations)
- 🐍 **Python Scripts**: Convert Keras models to `.onnx` format

## Requirements

- Windows 10/11
- Visual Studio 2022 (C++20)
- ONNX Runtime (download prebuilt from [Microsoft's site](https://onnxruntime.ai/))
- Python 3.10+ (for conversion scripts using `onnx`, `keras2onnx`, etc.)
- CMSIS-DSP source (manually imported and built as a static lib)
- Eigen (just headers)

## Setup Instructions

### 1. **ONNX Runtime**

- Download ONNX Runtime for Windows (CPU version)
- Set up the following in your Visual Studio project:
  - **Include directories**: ONNX Runtime `include/`
  - **Library directories**: ONNX Runtime `lib/`
  - **Runtime DLLs**: Copy ONNX Runtime DLLs next to your `.exe` or set up PATH
  - 
### 2. **Project Configuration**
- Language standard: `ISO C++20` for C++, `C17` for C
- **Eigen is header based under CMSIS/DSP/Include and CMSIS-DSP is compile as a static lib `DSP_Static_Lib.lib`**
- add CMSIS/DSP/Include to your include, Eigen will get included because you will #include <Eigen/whatever>
- Link the `onnxruntime.lib` and your `DSP_Static_Lib.lib`


## Running (I will be adding more tests, DSP and Eigen are just sanity checks, the onnx model is more interesting).

The project includes test files:
- `test_eigen()`: Validates Eigen usage (e.g., matrix inverse)
- `test_fft()`: Uses CMSIS-DSP FFT routines
- `main() `: Loads and runs ONNX model inference
- `sentiment_test_vectors.c` npy generated in the python scripts
So that really is all you need if you want to run it.
---

## Python: Keras to ONNX (see https://github.com/stevemac321/KerasToOnnx for more details).

Scripts provided (in `python_scripts/`):

```bash
python convert_keras_to_onnx.py model.h5 model.onnx
