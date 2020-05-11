void StackedLSTM(int batch_size, int depth, int max_seq_len,
                 const int seq_lens[batch_size],
                 float input[batch_size][max_seq_len],
                 float output[batch_size][max_seq_len][depth],
                 const float init_state);
