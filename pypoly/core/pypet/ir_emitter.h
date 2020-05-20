#ifndef _IR_EMITTER_H
#define _IR_EMITTER_H

#include "pypoly/core/pypet/error.h"
#include "pypoly/core/pypet/pypet.h"
#include "pypoly/core/pypet/sugared_value.h"
#include "pypoly/core/pypet/tree.h"

#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/tree_views.h>

namespace pypoly {
namespace pypet {

struct EmitStatements {
  EmitStatements(isl_ctx* ctx, PypetScop* scop) : ctx(ctx), scop(scop){};
  PypetTree* Extract(const torch::jit::List<torch::jit::Stmt>& statements);

 private:
  PypetTree* EmitFor(const torch::jit::For& stmt);

  PypetTree* EmitForImpl(const torch::jit::List<torch::jit::Expr>& targets,
                         const torch::jit::List<torch::jit::Expr>& itrs,
                         const torch::jit::SourceRange& loc,
                         const std::function<PypetTree*()>& emit_body);

  std::shared_ptr<SugaredValue> EmitApplyExpr(
      torch::jit::Apply& apply, size_t n_binders,
      const torch::jit::TypePtr& type_hint = nullptr);

  std::shared_ptr<SugaredValue> EmitSugaredExpr(
      const torch::jit::Expr& tree, size_t n_binders,
      const torch::jit::TypePtr& type_hint = nullptr);

  SugaredValuePtr GetSugaredVar(const torch::jit::Ident& ident,
                                bool required = true);

  PypetTree* EmitLoopCommon(
      torch::jit::SourceRange range,
      const std::function<PypetTree*()>& emit_body,
      const SugaredValuePtr& iter_val,
      c10::optional<torch::jit::List<torch::jit::Expr>> targets,
      c10::optional<torch::jit::Expr> cond);

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

  PypetScop* scop;
  PypetScop* get_scop() { return scop; };
};

}  // namespace pypet
}  // namespace pypoly

#endif
