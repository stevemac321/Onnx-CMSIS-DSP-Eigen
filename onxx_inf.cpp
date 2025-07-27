#include "sentiment_test_vectors.h"
#include <iostream>
#include <iomanip>
#include <onnxruntime_cxx_api.h>
#include <Eigen/Dense>


#include <arm_math.h>

#define MODEL_PATH L"full_keras_model.onnx"

int get_max_index(const float *buffer, int length);
void print_vector(const float *vec, int length);
void test_fft();
void test_Eigen();

const char *sentiment_labels[] = {"Extreme Negative", "Strong Negative",
                                  "Moderate Negative", "Positive",
                                  "Strong Positive"};

int main() {
 
  try {
    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
    Ort::SessionOptions session_options;
    Ort::Session session(env, MODEL_PATH, session_options);
    Ort::MemoryInfo memory_info =
        Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name = session.GetInputNameAllocated(0, allocator);
    auto output_name = session.GetOutputNameAllocated(0, allocator);
    std::vector<const char *> input_names{input_name.get()};
    std::vector<const char *> output_names{output_name.get()};

    std::vector<int64_t> input_shape{1, NUM_INPUTS};

    for (int i = 0; i < NUM_SAMPLES; i++) {
      const float *input_data = &test_vectors[i].input[0];

      Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
          memory_info, const_cast<float *>(input_data), NUM_INPUTS,
          input_shape.data(), input_shape.size());

      auto output_tensors =
          session.Run(Ort::RunOptions{nullptr}, input_names.data(),
                      &input_tensor, 1, output_names.data(), 1);

      float *output_data = output_tensors[0].GetTensorMutableData<float>();

      int predicted_class = get_max_index(output_data, 5);
      int expected_class = test_vectors[i].expected;

      std::cout << "Test case " << i << ":\n";
      std::cout << "Text: " << test_vectors[i].text << "\n";
      std::cout << "Expected Classification: " << expected_class << ", "
                << sentiment_labels[expected_class] << "\n";
      std::cout << "Predicted Classification: " << predicted_class << ", "
                << sentiment_labels[predicted_class] << "\n";
      std::cout << "Predicted Confidence: " << std::fixed
                << std::setprecision(6) << output_data[predicted_class] << "\n";

      std::cout << "Predicted Output: ";
      print_vector(output_data, 5);
      std::cout << "\n\n";
    }

  } catch (const Ort::Exception &e) {
    std::cerr << "ONNX Runtime exception: " << e.what() << "\n";
    return -1;
  } catch (const std::exception &e) {
    std::cerr << "Standard exception: " << e.what() << "\n";
    return -1;
  }
  test_fft();
  test_Eigen();
  return 0;
}

int get_max_index(const float *buffer, int length) {
  int max_index = 0;
  float max_value = buffer[0];
  for (int i = 1; i < length; ++i) {
    if (buffer[i] > max_value) {
      max_value = buffer[i];
      max_index = i;
    }
  }
  return max_index;
}

void print_vector(const float *vec, int length) {
  for (int i = 0; i < length; ++i) {
    std::cout << std::fixed << std::setprecision(6) << vec[i] << " ";
  }
  std::cout << "\n";
}
void test_fft()
{
  const uint16_t FFT_SIZE = 512;
#ifndef M_PI
  constexpr float M_PI = 3.1415927f;
#endif

  // Interleaved complex array: [real0, imag0, real1, imag1, ..., realN-1,
  // imagN-1]
  float fft_buffer[2 * FFT_SIZE] = {0};

  for (int i = 0; i < FFT_SIZE; ++i) {
    float t = (float)i / FFT_SIZE;
    float val = 0.6f * sinf(2.0f * M_PI * 4.0f * t) + // 4 Hz
                0.3f * sinf(2.0f * M_PI * 8.0f * t) + // 8 Hz
                0.1f * sinf(2.0f * M_PI * 15.0f * t); // 15 Hz
    fft_buffer[2 * i] = val;                          // real part
    fft_buffer[2 * i + 1] = 0.0f;                     // imag part
  }


  arm_cfft_instance_f32 cfft_instance;
  arm_cfft_init_f32(&cfft_instance, FFT_SIZE);

  // Perform FFT in-place
  arm_cfft_f32(&cfft_instance, fft_buffer, 0, 1);
  for (int i = 0; i < FFT_SIZE; ++i) {
    float real = fft_buffer[2 * i];
    float imag = fft_buffer[2 * i + 1];
    printf("Bin %3d: %+.6f + %+.6fj\n", i, real, imag);
  }

  
}
void test_Eigen()
{
  Eigen::Matrix2f mat;
  mat << 1, 2, 3, 4;

  // Compute the inverse
  Eigen::Matrix2f inv = mat.inverse();

  std::cout << "Matrix:\n" << mat << "\n\n";
  std::cout << "Inverse:\n" << inv << "\n";
}