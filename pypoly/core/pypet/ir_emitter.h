#ifndef _IR_EMITTER_H
#define _IR_EMITTER_H

#include "pypoly/core/pypet/error.h"
#include "pypoly/core/pypet/pypet.h"
#include "pypoly/core/pypet/sugared_value.h"

#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/tree_views.h>

namespace pypoly {
namespace pypet {

struct EmitStatements {
  EmitStatements(isl_ctx* ctx, std::shared_ptr<PypetScop> scop)
      : ctx(ctx), scop(scop){};
  void operator()(const torch::jit::List<torch::jit::Stmt>& statements);

 private:
  void EmitFor(const torch::jit::For& stmt);

  void EmitForImpl(const torch::jit::List<torch::jit::Expr>& targets,
                   const torch::jit::List<torch::jit::Expr>& itrs,
                   const torch::jit::SourceRange& loc,
                   const std::function<void()>& emit_body);

  std::shared_ptr<SugaredValue> EmitApplyExpr(
      torch::jit::Apply& apply, size_t n_binders,
      const torch::jit::TypePtr& type_hint = nullptr);

  std::shared_ptr<SugaredValue> EmitSugaredExpr(
      const torch::jit::Expr& tree, size_t n_binders,
      const torch::jit::TypePtr& type_hint = nullptr);

  SugaredValuePtr GetSugaredVar(const torch::jit::Ident& ident,
                                bool required = true);

  void EmitLoopCommon(torch::jit::SourceRange range,
                      const std::function<void()>& emit_body,
                      const SugaredValuePtr& iter_val,
                      c10::optional<torch::jit::List<torch::jit::Expr>> targets,
                      c10::optional<torch::jit::Expr> cond);

  void EmitIf(const torch::jit::If& stmt);
  void EmitWhile(const torch::jit::While& stmt);
  void EmitAssignment(const torch::jit::Assign& stmt);
  void EmitAugAssignment(const torch::jit::AugAssign& stmt);
  void EmitRaise(const torch::jit::Raise& stmt);
  void EmitAssert(const torch::jit::Assert& stmt);
  void EmitReturn(const torch::jit::Return& stmt);
  void EmitContinue(const torch::jit::Continue& stmt);
  void EmitBreak(const torch::jit::Break& stmt);
  void EmitClosure(const torch::jit::Def& stmt);
  void EmitDelete(const torch::jit::Delete& smt);
  void EmitExpr(const torch::jit::Expr& tree);

  isl_ctx* ctx;
  isl_ctx* get_isl_ctx() { return ctx; };

  PypetScopPtr scop;
  PypetScopPtr get_scop() { return scop; };
};

}  // namespace pypet
}  // namespace pypoly

#endif
