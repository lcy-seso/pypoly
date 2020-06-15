float cell(float a, float b) { return 5.; }

void StackedLSTM(int batch_size, int depth, const int seq_lens[16],
                 float input[16][50], float output[16][50][3]) {
  float init_state = 0.;
  int max_seq_len = 50;
#pragma scop
  for (int i = 0; i < batch_size; ++i) {
    int seq_len = seq_lens[i];
    for (int j = 0; j < seq_len; ++j) {
      for (int k = 0; k < depth; ++k) {
        float h_prev = 0.;
        if (j == 0) {
          h_prev = init_state;
        } else {
          h_prev = output[i][j - 1][k];
        }

        float x = 0.;
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
