# Scan Rewrite

According to listed core optimizations, whether keeping *scan* construct depends on the computation structure. For example, for Stacked LSTM and Grid RNN, rewriting *scan* to (nested) *for* loops is more useful to the polyhedral model.

# Map Rewrite

Similar to scan rewriting, rewriting the map and inside lambda function sometimes exposes more optimization opportunities. For example, when a map statement scales each scalar in a tensor and the predecessor statement is a fully connected layer, we can scale the weight matrix with the same coefficient to reduce computation through algebraic expression analysis.