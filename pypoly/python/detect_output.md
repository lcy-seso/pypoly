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
  # assume: depth < len(xs)
  # 3 cases included
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

  # do we need these 'None'? A procedure similar to the collection data above
  # can be used to transform the 'skewed_output' back to 'output'
  # quick note: 'None' here is related to the way how we collect data
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

# emb_vecs: List[{Tensor, Tensor}]
# prev_states: List[List[{Tensor, Tensor}]]
def gen_map_func(idx, emb_vecs, prev_states):
  # state_xs, state_ys, x_ts, y_ts, cur_ws are built in this function, they
  # share a same type List[List{Tensor, Tensor}]. lists are in same length.
  # for simplicity, assume depth < len(emb_xs) < len(emb_ys)
  if idx < depth:
    # TODO
  elif idx < len(emb_xs):
    # TODO
  elif idx < len(emb_ys):
    # TODO
  elif idx < len(emb_xs) + len(emb_ys) - 1:
    # TODO
  else:
    # TODO
  map_context = zip(state_xs, state_ys, x_ts, y_ts, cur_ws)
  return map(map_context, lambda _, item1: return map(item1, lambda _, item2: grid_cell(item2))

skewed_input = TensorArray(len(emb_xs) + len(emb_ys) - 1, lambda c0: return TensorArray(mix(len(emb_xs), c0 + 1) - max(0, c0 - len(emb_ys) + 1), lambda c1: return {index(emb_xs, c1), index(emb_ys, c0 - c1)})) + TensorArray(depth - 1, lambda _: return None)

skewed_output = scan(skewed_input, gen_map_func, None)
```
Try to answer following questions
- can we simplify code structures above ? -> The overall code structure is discussed in below.
- replace TensorArray with another interface to pack or stack data ? -> I personally recommend this strategy to deliver clear information to the optimizer that it does not need to construct a 'true' TensorArray in some cases.
- do we need to add redundant 'None's? -> 'None' is (only?) needed at the top level scan to compromise between the API and transformed code.

One principle of our programming model: if a function accept an Iterable Object as an input, main body of the function is organized by *map*, *reduce* or *scan*. (**Hint, discussion here may not applicable for code when in a block, tensor level operation is launched after a reduce or multiple control constructs, scan + map ?**)In other words, these APIs are used to explicitly describe the operations over lists of tensors and decouple the control flow and the concept of operators in traditional frameworks.

The overall code structure
1. transform the initial state and append some 'None's to the tail. this state is scanned over, e.g. carries the dependence.
2. inputs of the tensor level function call in the deepest 'scan' will be collected (as references?) and orgranized.
3. apply 'map's to the collected data
4. transform the output TensorArray back to its layout in user code

In current model examples, transformations in **step 1** and **step 4** are performed in an affine way.

More about **step 2**: in general, it is a procedure that accepts a variable that represents the predecessor state and other variables that correspond to the values in the scope. As demonstrated by the example above, different ways will be used to collect data when index value of the top level 'scan' changes.

Multiple control statements (only consider scan, map and reduce) in a nested function (preliminary discussion), here the word 'follow' means that the output of the predecessor is consumed by the latter one & the word 'compose' means that compose the two lambda functions
1. a scan followed by a map -> compose
2. a scan followed by a scan -> compose
3. a scan followed by a reduce -> split the scan and reduce into two regions and apply optimizations one by one -> a heuristic strategy -> a possible case, the scan returns List[List[Tensor]], the reduce returns List[Tensor]
4. a map followed by a map -> compose
5. a map followed by a scan -> compose
6. a map followed by a reduce -> compose