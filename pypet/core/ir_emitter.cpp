#include "pypet/core/ir_emitter.h"

namespace pypet {

void EmitFor::operator()(const torch::jit::For& stmt) {
  std::cout << stmt.range() << std::endl;
}

void EmitIf::operator()(const torch::jit::If& stmt) {}

void EmitWhile::operator()(const torch::jit::While& stmt) {
  throw std::invalid_argument("while statement is not supported.");
}

void EmitAssignment::operator()(const torch::jit::Assign& stmt) {}

void EmitAugAssignment::operator()(const torch::jit::AugAssign& stmt) {}

void EmitRaise::operator()(const torch::jit::Raise& stmt) {}

void EmitAssert::operator()(const torch::jit::Assert& stmt) {}

void EmitReturn::operator()(const torch::jit::Return& stmt) {}

void EmitContinue::operator()(const torch::jit::Continue& stmt) {}

void EmitBreak::operator()(const torch::jit::Break& stmt) {}

void EmitClosure::operator()(const torch::jit::Def& stmt) {}

void EmitDelete::operator()(const torch::jit::Delete& stmt) {}

void EmitExpr::operator()(const torch::jit::Expr& tree) {}

}  // namespace pypet
