#include "stacked_lstm.h"

#include <stdlib.h>
#include <time.h>

typedef float cell_ptr(float a, float b);

int max_element(const int* array, int size) {
  int max = array[0];
  for (int i = 1; i < size; ++i) max = array[i] > max ? array[i] : max;
  return max;
}
int min_element(const int* array, int size) {
  int min = array[0];
  for (int i = 1; i < size; ++i) min = array[i] < min ? array[i] : min;
  return min;
}

int main() {
  srand(time(NULL));

  int batch_size = 4;
  int depth = 3;

  int Min_Len = 4;
  int Max_Len = 20;

  int seq_lens[batch_size];
  for (int i = 0; i < batch_size; ++i)
    seq_lens[i] = rand() % (Max_Len - Min_Len) + Min_Len;

  int max_seq_len = max_element(seq_lens, batch_size);
  int min_seq_len = min_element(seq_lens, batch_size);

  float input[batch_size][max_seq_len];
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < seq_lens[i]; ++j) {
      input[i][j] = (float)rand() / (float)(RAND_MAX);
    }
  }

  float output[batch_size][max_seq_len][depth];
  for (int i = 0; i < batch_size; ++i) {
    for (int j = 0; j < max_seq_len; ++j) {
      for (int k = 0; k < depth; ++k) {
        output[i][j][k] = 0.;
      }
    }
  }

  // pet does not support parse array of function pointers.
  // float (*cells[depth])(float x, float y);
  // cells[0] = cell;
  // cells[1] = cell;
  // cells[2] = cell;
  // StackedLSTM(batch_size, depth, max_seq_len, seq_lens, input, output, cells,
  //             0. /* init_state*/);

  StackedLSTM(batch_size, depth, max_seq_len, seq_lens, input, output,
              0. /* init_state*/);

  for (int i = 0; i < batch_size; ++i) {
    int seq_len = seq_lens[i];
    for (int j = 0; j < seq_len; ++j) {
      for (int k = 0; k < depth; ++k) {
        printf("%.1f ", output[i][j][k]);
      }
      printf("\n");
    }
    printf("\n");
  }
}
