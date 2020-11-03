stacked lstm
```python
def lstm_cell(state: Tuple[Tensor], x: Tensor, rnn_param: Tuple[Tensor]) -> Tuple[Tensor]:
  ws = [{wi, ui, bi}, {wf, uf, bf}, {wo, uo, bo}, {wc, uc, bc}]
  states = map(ws, lambda idx, weights: x @ weights[0] + h_prev @ weights[1] + weights[2])
  i = sigmoid(index(states, 0))
  f = sigmoid(index(states, 1))
  o = sigmoid(index(states, 2))
  c_candidate = tanh(index(states, 3))
  c = f * c_prev + i * c_candidate
  h = o * tanh(c)
  return h, c


def wavefront_func(idx, x, prev_states):
  # depth < len(xs) by default
  if idx < depth:
    in_xs = x + TensorArray(idx, lambda xs_idx: return prev_states[xs_idx - 1][0])
    in_hs = TensorArray(idx - 1, lambda hs_idx: return prev_states[hs_idx][0]) + h_initials[idx]
    in_cs = TensorArray(idx - 1, lambda cs_idx: return prev_states[cs_idx][1]) + c_initials[idx]
    in_ws = gather(ws, range(0, idx))
  elif idx < len(xs):
    in_xs = x + TensorArray(depth - 1, lambda xs_idx: return prev_states[xs_idx - 1][0])
    in_hs = TensorArray(idx, lambda hs_idx: return prev_states[hs_idx][0])
    in_cs = TensorArray(idx, lambda cs_idx: return prev_states[cs_idx][1])
    in_ws = ws
  else:
    empty_num = idx - len(x) + 1
    in_xs = TensorArray(depth - empty_num, lambda xs_idx: prev_states[xs_idx-1][0])
    in_hs = TensorArray(depth - empty_num, lambda hs_idx: prev_states[hs_idx][0])
    in_cs = TensorArray(depth - empty_num, lambda cs_idx: prev_states[cs_idx][1])
    in_ws = gather(ws, range(empty_num, depth))

  map_out = map(zip(in_xs, in_hs, in_cs, in_ws), lambda idx, (x, h, c, w): lstm_cell({h, c}, x, in_ws))

  if idx < depth:
    return map_out + TensorArray(depth - idx - 1, lambda _: return None)
  elif idx < len(xs):
    return map_out
  else:
    return TensorArray(idx - len(xs) + 1, lambda _: return None) + map_out

embed_xs = xs + TensorArray(depth - 1, lambda _: return None)
skewed_output = scan(embed_xs, wavefront_func, TensorArray(depth, lambda _: return None)
```

grid rnn
```python

def y_map_func():
  map()
  pass

def x_map_func(idx, emb_vecs, prev_states):
  if idx < len(emb_xs) + len(emb_ys) - 1:

  else:
    
  map(y_map_func)
  pass

skewed_input = TensorArray(len(emb_xs) + len(emb_ys) - 1, lambda c0: return TensorArray(mix(len(emb_xs), c0 + 1) - max(0, c0 - len(emb_ys) + 1), lambda c1: return {index(emb_xs, c1), index(emb_ys, c0 - c1)})) + TensorArray(depth - 1, lambda _: return None)

skewed_output = scan(skewed_input, x_map_func)
```
Try to answer following questions
- can we simplify code structures above ?
- replace TensorArray with another interface to 'pack' data ?
- do we need to add redundant 'None' ?