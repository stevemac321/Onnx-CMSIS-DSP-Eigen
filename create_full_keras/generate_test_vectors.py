import pickle
import numpy as np
from pathlib import Path
from inference_strings import inference_samples_by_class
from tensorflow.keras.models import load_model

# === Load TF-IDF Vectorizer and Model ===
vectorizer_path = Path("models/tfidf_vectorizer.pkl")
model_path = Path("models/full_keras_model.keras")

vectorizer = pickle.loads(vectorizer_path.read_bytes())
model = load_model(model_path)

# === Flatten Samples ===
samples = [(text, label) for label, group in inference_samples_by_class.items() for text in group]
NUM_INPUTS = vectorizer.transform(["dummy"]).shape[1]
NUM_SAMPLES = len(samples)
FLOATS_PER_LINE = 500

# === Format C Float Array ===
def format_float_array(array):
    lines = []
    for i in range(0, len(array), FLOATS_PER_LINE):
        chunk = array[i:i + FLOATS_PER_LINE]
        line = ", ".join(f"{x:.6f}f" for x in chunk)
        lines.append("    " + line)
    return ",\n".join(lines)

def escape_c_string(s):
    return s.replace("\\", "\\\\").replace('"', '\\"')

# === Write Header File ===
header_text = f"""#ifndef SENTIMENT_TEST_VECTORS_H
#define SENTIMENT_TEST_VECTORS_H
#include <stddef.h>

#define NUM_CLASSES 5
#define NUM_INPUTS {NUM_INPUTS}
#define NUM_SAMPLES {NUM_SAMPLES}

#ifdef __cplusplus
extern "C" {{
#endif

typedef struct {{
  float input[NUM_INPUTS];
  int expected;
  const char *text;
}} SentimentTestVector;

extern SentimentTestVector test_vectors[NUM_SAMPLES];
extern size_t test_vector_count;

#ifdef __cplusplus
}}
#endif
#endif // SENTIMENT_TEST_VECTORS_H
"""

Path("output/sentiment_test_vectors.h").write_text(header_text, encoding="utf-8")
print("✅ Header written to: output/sentiment_test_vectors.h")

# === Generate C Test Vectors ===
c_entries = []
for text, label in samples:
    vector = vectorizer.transform([text]).toarray()[0]
    input_block = format_float_array(vector)
    c_safe_text = escape_c_string(text)
    entry = f"""  {{
    .input = {{
{input_block}
    }},
    .expected = {label},
    .text = "{c_safe_text}"
  }}"""
    c_entries.append(entry)

vector_array = ",\n".join(c_entries)
test_vector_count_declaration = f"size_t test_vector_count = {NUM_SAMPLES};"

c_file_text = f"""#include "sentiment_test_vectors.h"

SentimentTestVector test_vectors[{NUM_SAMPLES}] = {{
{vector_array}
}};

{test_vector_count_declaration}
"""

Path("output/sentiment_test_vectors.c").write_text(c_file_text, encoding="utf-8")
print("✅ C file written to: output/sentiment_test_vectors.c")