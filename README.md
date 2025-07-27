# Onnx-CMSIS-DSP-Eigen
ML, signal processing, math libs
# ONNX + CMSIS-DSP + Eigen Inference Demo (C++)

This project demonstrates running a Keras-trained model (converted to ONNX) from C++, using ONNX Runtime on Windows, and augmenting it with:

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

### 1. **Clone Eigen and CMSIS-DSP**

- Place Eigen under `external/Eigen` (only the `Eigen/` and `unsupported/` directories are needed)
- CMSIS-DSP should be built as a static `.lib`. This project includes a minimal build under `DSP_Static_Lib`.

### 2. **ONNX Runtime**

- Download ONNX Runtime for Windows (CPU version)
- Set up the following in your Visual Studio project:
  - **Include directories**: ONNX Runtime `include/`
  - **Library directories**: ONNX Runtime `lib/`
  - **Runtime DLLs**: Copy ONNX Runtime DLLs next to your `.exe` or set up PATH

### 3. **Project Configuration**

- Language standard: `ISO C++20` for C++, `C17` for C
- Add include paths for Eigen and CMSIS-DSP
- Link the `onnxruntime.lib` and your `DSP_Static_Lib.lib`

### 4. **Fix: One Unresolved External**

You may encounter a single unresolved symbol related to a missing function in CMSIS-DSP.  
**Workaround:** Comment out that function (was removed from later source), and rebuild the static lib.

## Running

The project includes test files:
- `test_eigen.cpp`: Validates Eigen usage (e.g., matrix inverse)
- `test_dsp_fft.cpp`: Uses CMSIS-DSP FFT routines
- `main_inference.cpp`: Loads and runs ONNX model inference

Output confirms each component works end-to-end on desktop.

---

## Python: Keras to ONNX

Scripts provided (in `python_scripts/`):

```bash
python convert_keras_to_onnx.py model.h5 model.onnx
