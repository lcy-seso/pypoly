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
  StmtScheduleInfo schedule_info_;
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
  func->schedule_info_ = GreedyScheduler(graph);
}

Statement* CanScanOptimized(Statement* stmt) {
  if (IsScan(stmt->expr_) == false) {
    return nullptr;
  }
  Function* applied_func = GetScanInnerFunc(stmt->expr_);
  int dep_stmt_cnt = 0;
  Statement* nested_dep_stmt = nullptr;
  for (Statement* item : applied_func->stmts_) {
    // do not recursively check the user function.
    if (IsUserFunc(item->expr_)) {
      return nullptr;
    }
    // do not optimize inner reduce
    if (IsReduce(item->expr_)) {
      return nullptr;
    }
    // do not optimize inner map. Although there may be statements carrying dependence inside the map,
    // it will introduce implementation complexity.
    if (IsMap(item->expr_)) {
      return nullptr;
    }
    if (IsScan(item->expr_) || IsFold(item->expr_)) {
      ++dep_stmt_cnt;
      nested_dep_stmt = item;
    }
  }
  return nested_dep_stmt;
}

// we current focus on the two level scan now
void OptimizeScan(Statement* scan, Statement* nested_stmt) {
  // the dimension naming rule: i0, i1
  Type* out_type = GetScanOutType(scan);
  // according to the constraints, we know that i0 carries a dependence vector of 1 implicitly

  // can we guarantee that the type of initializer must be iterated over by the nested_stmt?
  // as a result, i1 carries a dependence vector of 1 implicitly too

  // for the two level case, the function called by nested_stmt is analyzed.
  // the returned element of this function corresponds to the level-2 item in the out_type
  // we need to trace variables accessed by this function clearly to compute the dependence vectors(the relative position to the returned element).
}

void ParallelismDetection(Program* prog) {
  Function* entry = prog->name2func_[entry_func_];
  BlockParallelismDetection(entry);
  for (int i = 0; i < entry->stmts_.size(); ++i) {
    Statement* stmt = entry->stmts_[i];
    if (CanScanOptimized(stmt) != nullptr) {
      OptimizeScan(stmt);
    } else if (CanFoldOptimized(stmt) != nullptr) {
      OptimizeFold(stmt);
    }
  }
}
```