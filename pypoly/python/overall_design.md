```c++
class Type {
};

// design choice: common operators like +, -, *, / are all represented by function calls
enum ExprType {
  kCall = 0;
  kInt = 1;
  kFloat = 2;
};

class Expr {
 private:
  ExprType type_;
  int integer_val_;
  float float_val_;
  string func_name_;
  vector<Expr*> args_;
};

// let or a function
enum StatementType {
  kLet = 0;
  kFunc = 1;
};

class Statement {
 private:
  StatementType type_;
  // binded name in 'let', empty for kFunc
  string var_name_;
  // AST for the expression of function call
  Expr* expr_;
};

struct StmtScheduleInfo {
  vector<vector<Statement*>> info_;
};

// here Function is defined by user, not system functions
// can we constrain that type specifications for inputs are mandatory?
class Function {
 private:
  // anonymous function
  string name_;
  unordered_map<string, Type*> param2type_;
  // 'return' keyword is omitted, implicitly return elements in the last statement
  //  Tuple elements need to be declared explicitly 
  vector<Type*> ret_types_;
  vector<Statement*> stmts_;

};

class Program {

 private:
  string entry_func_;
  // global func
  unordered_map<string, Function*> name2func_;
  // global data
  unordered_map<string, Type*> name2data_;
};

class Node {
  vector<Edge*> in_edges_;
  vector<Edge*> out_edges_;
};

class Edge {
  Node* src_node_;
  Node* tgt_node_;
};

class FlowNode : Node {
  Statement* stmt_;
};

class FlowEdge : Edge {
  string var_;
};

class Graph {
  vector<Node*> nodes_;
  vector<Edge*> edges_;
};

// add an entry node to make the Graph a DAG
StmtScheduleInfo GreedyScheduler(Graph* graph) {
}

Graph* BuildFlowGraph(vector<Statement*>& stmts) {
}


// Assume that we have done the shape propagation before optimizations
void BlockParallelismDetection(Function* func) {
  // do nothing
  if (func->stmts_.size() == 1) {
    return 
  }
  Graph* graph = BuildFlowGraph(func->stmts_);
  func->parallel_info_ = GreedyScheduler(graph);
}

void ParallelismDetection(Program* prog) {
  Function* entry = prog->name2func_[entry_func_];
  BlockParallelismDetection(entry);
  for (int i = 0; i < entry->stmts_.size(); ++i) {
    Statement* stmt = entry->stmts_[i];
    if (CanScanOptimized(stmt)) {
      OptimizeScan(stmt);
    } else if (CanFoldOptimized(stmt)) {
      OptimizeFold(stmt);
    }
  }
}
```