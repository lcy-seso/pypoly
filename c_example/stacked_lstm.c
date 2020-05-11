#include "stacked_lstm.h"

float cell(float a, float b) { return 5.; }

void StackedLSTM(int batch_size, int depth, int max_seq_len,
                 const int seq_lens[batch_size],
                 float input[batch_size][max_seq_len],
                 float output[batch_size][max_seq_len][depth],
                 const float init_state) {
#pragma scop
  for (int i = 0; i < batch_size; ++i) {
    int seq_len = seq_lens[i];
    for (int j = 0; j < seq_len; ++j) {
      for (int k = 0; k < depth; ++k) {
        float h_prev;
        if (j == 0) {
          h_prev = init_state;
        } else {
          h_prev = output[i][j - 1][k];
        }

        float x;
        if (k == 0) {
          x = input[i][j];
        } else {
          x = output[i][j][k - 1];
        }

        float h = cell(x, h_prev);
        output[i][j][k] = h;
      }
    }
  }
#pragma endscop
}
