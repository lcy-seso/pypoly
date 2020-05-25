#ifndef PYPOLY_CORE_PYPET_IR_EMITTER_H_
#define PYPOLY_CORE_PYPET_IR_EMITTER_H_

#include "pypoly/core/pypet/error.h"
#include "pypoly/core/pypet/pypet.h"
#include "pypoly/core/pypet/tree.h"

#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/tree_views.h>

namespace pypoly {
namespace pypet {

struct EmitStatements {
  EmitStatements(isl_ctx* ctx, std::shared_ptr<PypetScop> scop)
      : ctx(ctx), scop(scop){};
  std::vector<PypetTree*> operator()(
      const torch::jit::List<torch::jit::Stmt>& statements);

 private:
  PypetTree* EmitBlockStatements(
      const torch::jit::List<torch::jit::Stmt>& statements);
  PypetTree* EmitStatement(const torch::jit::Stmt& stmt);
  PypetTree* EmitFor(const torch::jit::For& stmt);

  PypetTree* EmitIf(const torch::jit::If& stmt);
  PypetTree* EmitWhile(const torch::jit::While& stmt);
  PypetTree* EmitAssignment(const torch::jit::Assign& stmt);
  PypetTree* EmitAugAssignment(const torch::jit::AugAssign& stmt);
  PypetTree* EmitRaise(const torch::jit::Raise& stmt);
  PypetTree* EmitAssert(const torch::jit::Assert& stmt);
  PypetTree* EmitReturn(const torch::jit::Return& stmt);
  PypetTree* EmitContinue(const torch::jit::Continue& stmt);
  PypetTree* EmitBreak(const torch::jit::Break& stmt);
  PypetTree* EmitClosure(const torch::jit::Def& stmt);
  PypetTree* EmitDelete(const torch::jit::Delete& smt);
  PypetTree* EmitExpr(const torch::jit::Expr& tree);

  isl_ctx* ctx;
  isl_ctx* get_isl_ctx() { return ctx; };

  std::set<std::string> used_names;

  PypetScopPtr scop;
  PypetScopPtr get_scop() { return scop; };
};

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_CORE_PYPET_IR_EMITTER_H_
