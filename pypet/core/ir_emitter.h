#ifndef _IR_EMMITTER_H
#define _IR_EMMITTER_H

#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/tree_views.h>

namespace pypet {
struct EmitFor {
  EmitFor(){};
  void operator()(const torch::jit::For& stmt);
};

struct EmitIf {
  EmitIf(){};
  void operator()(const torch::jit::If& stmt);
};

struct EmitWhile {
  EmitWhile(){};
  void operator()(const torch::jit::While& stmt);
};

struct EmitAssignment {
  EmitAssignment(){};
  void operator()(const torch::jit::Assign& stmt);
};

struct EmitAugAssignment {
  EmitAugAssignment(){};
  void operator()(const torch::jit::AugAssign& stmt);
};

struct EmitRaise {
  EmitRaise(){};
  void operator()(const torch::jit::Raise& stmt);
};

struct EmitAssert {
  EmitAssert(){};
  void operator()(const torch::jit::Assert& stmt);
};

struct EmitReturn {
  EmitReturn(){};
  void operator()(const torch::jit::Return& stmt);
};

struct EmitContinue {
  EmitContinue(){};
  void operator()(const torch::jit::Continue& stmt);
};

struct EmitBreak {
  EmitBreak(){};
  void operator()(const torch::jit::Break& stmt);
};

struct EmitClosure {
  EmitClosure(){};
  void operator()(const torch::jit::Def& stmt);
};

struct EmitDelete {
  EmitDelete(){};
  void operator()(const torch::jit::Delete& smt);
};

struct EmitExpr {
  EmitExpr(){};
  void operator()(const torch::jit::Expr& tree);
};

}  // namespace pypet

#endif
