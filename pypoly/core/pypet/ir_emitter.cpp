#include "pypoly/core/pypet/ir_emitter.h"

#include "pypoly/core/pypet/tree.h"

namespace pypoly {
namespace pypet {

PypetTree* EmitStatements::Extract(
    const torch::jit::List<torch::jit::Stmt>& statements) {
  PypetTree* tree;
  for (auto begin = statements.begin(); begin != statements.end(); ++begin) {
    auto stmt = *begin;
    switch (stmt.kind()) {
      case torch::jit::TK_IF: {
        tree = EmitIf(torch::jit::If(stmt));
      } break;
      case torch::jit::TK_WHILE: {
        tree = EmitWhile(torch::jit::While(stmt));
      } break;
      case torch::jit::TK_FOR: {
        tree = EmitFor(torch::jit::For(stmt));
      } break;
      case torch::jit::TK_ASSIGN: {
        tree = EmitAssignment(torch::jit::Assign(stmt));
      } break;
      case torch::jit::TK_AUG_ASSIGN: {
        tree = EmitAugAssignment(torch::jit::AugAssign(stmt));
      } break;
      case torch::jit::TK_EXPR_STMT: {
        auto expr = torch::jit::ExprStmt(stmt).expr();
        tree = EmitExpr(expr);
      } break;
      case torch::jit::TK_RAISE: {
        tree = EmitRaise(torch::jit::Raise(stmt));
      } break;
      case torch::jit::TK_ASSERT: {
        tree = EmitAssert(torch::jit::Assert(stmt));
      } break;
      case torch::jit::TK_RETURN: {
        tree = EmitReturn(torch::jit::Return(stmt));
      } break;
      case torch::jit::TK_CONTINUE: {
        tree = EmitContinue(torch::jit::Continue(stmt));
      } break;
      case torch::jit::TK_BREAK: {
        tree = EmitBreak(torch::jit::Break(stmt));
      } break;
      case torch::jit::TK_PASS:
        // Emit nothing for pass
        break;
      case torch::jit::TK_DEF: {
        tree = EmitClosure(torch::jit::Def(stmt));
        break;
      }
      case torch::jit::TK_DELETE: {
        tree = EmitDelete(torch::jit::Delete(stmt));
      } break;
      default:
        throw std::invalid_argument("Unrecognized statement kind ");
    }
  }
  return tree;
}

PypetTree* EmitStatements::EmitForImpl(
    const torch::jit::List<torch::jit::Expr>& targets,
    const torch::jit::List<torch::jit::Expr>& itrs,
    const torch::jit::SourceRange& loc,
    const std::function<PypetTree*()>& emit_body) {
  if (itrs.size() != 1) {
    throw torch::jit::ErrorReport(loc)
        << "List of iterables is not supported currently";
  }

  // Emit loop information for builtinFunction values like range(), zip(),
  // enumerate() or SimpleValue like List, Tensor, Dict, etc.
  SugaredValuePtr sv = EmitSugaredExpr(itrs[0], 1 /* n_binders */);
  SugaredValuePtr iterable = sv->iter(loc);

  EmitLoopCommon(loc, emit_body, iterable, targets, {});

  // TODO(Ying) To decide whether to unroll the loop for iterables that
  // contain computation graphs.
  // if (!iterable->shouldEmitUnrolled()) {
  // } else {
  // }
}

PypetTree* EmitStatements::EmitLoopCommon(
    torch::jit::SourceRange range, const std::function<PypetTree*()>& emit_body,
    const SugaredValuePtr& iter_val,
    c10::optional<torch::jit::List<torch::jit::Expr>> targets,
    c10::optional<torch::jit::Expr> cond) {
  // recursively parse statements.
  return emit_body();
}

PypetTree* EmitStatements::EmitFor(const torch::jit::For& stmt) {
  PypetTree* tree = CreatePypetTreeBlock(
      ctx, stmt.range(), 1 /*whether has its own scop*/, stmt.body().size());

  auto emit_body = [&]() -> PypetTree* {
    EmitStatements emitter(get_isl_ctx(), get_scop());
    return emitter.Extract(stmt.body());
  };
  return EmitForImpl(stmt.targets(), stmt.itrs(), stmt.range(), emit_body);
}

std::shared_ptr<SugaredValue> EmitStatements::EmitApplyExpr(
    torch::jit::Apply& apply, size_t n_binders,
    const torch::jit::TypePtr& type_hint) {
  auto sv = EmitSugaredExpr(apply.callee(), 1);

  auto loc = apply.callee().range();
  if (auto special_form =
          dynamic_cast<torch::jit::SpecialFormValue*>(sv.get())) {
    // this branch handles expressions that look like apply statements
    // but have special evaluation rules for the arguments.
    // TODO(ying) Check code pattens that fall in this branch.
    throw Error(apply.range()) << "Unsupported code pattern.";
  }

  return sv->call(loc);
}

std::shared_ptr<SugaredValue> EmitStatements::EmitSugaredExpr(
    const torch::jit::Expr& tree, size_t n_binders,
    const torch::jit::TypePtr& type_hint) {
  switch (tree.kind()) {
    case torch::jit::TK_VAR:
      return GetSugaredVar(torch::jit::Var(tree).name());
    case '.': {
      LOG(INFO) << ". = " << tree.range().text();
      throw Error(tree) << "Not implemented yet.";
    }
    case torch::jit::TK_APPLY: {
      auto apply = torch::jit::Apply(tree);
      return EmitApplyExpr(apply, n_binders, type_hint);
    } break;
    default:
      throw Error(tree) << "Not implemented yet.";
  }
}

SugaredValuePtr EmitStatements::GetSugaredVar(const torch::jit::Ident& ident,
                                              bool required) {
  LOG(INFO) << "Var = " << ident.range().text();
}

PypetTree* EmitStatements::EmitIf(const torch::jit::If& stmt) {}
PypetTree* EmitStatements::EmitWhile(const torch::jit::While& stmt) {
  throw std::invalid_argument("while statement is not supported.");
}

PypetTree* EmitStatements::EmitAssignment(const torch::jit::Assign& stmt) {}
PypetTree* EmitStatements::EmitAugAssignment(
    const torch::jit::AugAssign& stmt) {}
PypetTree* EmitStatements::EmitRaise(const torch::jit::Raise& stmt) {}
PypetTree* EmitStatements::EmitAssert(const torch::jit::Assert& stmt) {}
PypetTree* EmitStatements::EmitReturn(const torch::jit::Return& stmt) {}
PypetTree* EmitStatements::EmitContinue(const torch::jit::Continue& stmt) {}
PypetTree* EmitStatements::EmitBreak(const torch::jit::Break& stmt) {}
PypetTree* EmitStatements::EmitClosure(const torch::jit::Def& stmt) {}
PypetTree* EmitStatements::EmitDelete(const torch::jit::Delete& stmt) {}
PypetTree* EmitStatements::EmitExpr(const torch::jit::Expr& tree) {}

}  // namespace pypet
}  // namespace pypoly
