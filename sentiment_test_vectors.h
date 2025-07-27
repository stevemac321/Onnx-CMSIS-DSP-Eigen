#ifndef SENTIMENT_TEST_VECTORS_H
#define SENTIMENT_TEST_VECTORS_H
#include <stddef.h>

#define NUM_CLASSES 5
#define NUM_INPUTS 170
#define NUM_SAMPLES 50

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  float input[NUM_INPUTS];
  int expected;
  const char *text;
} SentimentTestVector;

extern SentimentTestVector test_vectors[NUM_SAMPLES];
extern size_t test_vector_count;

#ifdef __cplusplus
}
#endif
#endif // SENTIMENT_TEST_VECTORS_H
