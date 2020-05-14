#include "pypoly/core/pypet/ir_emitter.h"

#include "pypoly/core/pypet/tree.h"

namespace pypoly {
namespace pypet {

namespace {

PypetExpr* PypetExprFromIntVal(int val) {
  // TODO
  return nullptr;
}

PypetExpr* ExtractIndexExpr(const torch::jit::Expr& expr) {
  // TODO
  return nullptr;
}

PypetExpr* PypetExprAccessFromIndex() {
  // TODO
  return nullptr;
}

PypetExpr* ExtractAccessExpr(const torch::jit::Expr& expr) {
  // TODO
  return nullptr;
}

PypetExpr* BuildPypetBinaryOpExpr(PypetOpType op_type, PypetExpr* lhs,
                                  PypetExpr* rhs) {
  // TODO
  return nullptr;
}

}  // namespace

void EmitStatements::operator()(
    const torch::jit::List<torch::jit::Stmt>& statements) {
  for (auto begin = statements.begin(); begin != statements.end(); ++begin) {
    auto stmt = *begin;
    switch (stmt.kind()) {
      case torch::jit::TK_IF: {
        EmitIf(torch::jit::If(stmt));
      } break;
      case torch::jit::TK_WHILE: {
        EmitWhile(torch::jit::While(stmt));
      } break;
      case torch::jit::TK_FOR: {
        EmitFor(torch::jit::For(stmt));
      } break;
      case torch::jit::TK_ASSIGN: {
        EmitAssignment(torch::jit::Assign(stmt));
      } break;
      case torch::jit::TK_AUG_ASSIGN: {
        EmitAugAssignment(torch::jit::AugAssign(stmt));
      } break;
      case torch::jit::TK_EXPR_STMT: {
        auto expr = torch::jit::ExprStmt(stmt).expr();
        EmitExpr(expr);
      } break;
      case torch::jit::TK_RAISE: {
        EmitRaise(torch::jit::Raise(stmt));
      } break;
      case torch::jit::TK_ASSERT: {
        EmitAssert(torch::jit::Assert(stmt));
      } break;
      case torch::jit::TK_RETURN: {
        EmitReturn(torch::jit::Return(stmt));
      } break;
      case torch::jit::TK_CONTINUE: {
        EmitContinue(torch::jit::Continue(stmt));
      } break;
      case torch::jit::TK_BREAK: {
        EmitBreak(torch::jit::Break(stmt));
      } break;
      case torch::jit::TK_PASS:
        // Emit nothing for pass
        break;
      case torch::jit::TK_DEF: {
        EmitClosure(torch::jit::Def(stmt));
        break;
      }
      case torch::jit::TK_DELETE: {
        EmitDelete(torch::jit::Delete(stmt));
      } break;
      default:
        throw std::invalid_argument("Unrecognized statement kind ");
    }
  }
}

void EmitStatements::EmitForImpl(
    const torch::jit::List<torch::jit::Expr>& targets,
    const torch::jit::List<torch::jit::Expr>& itrs,
    const torch::jit::SourceRange& loc,
    const std::function<void()>& emit_body) {
  if (itrs.size() != 1) {
    throw torch::jit::ErrorReport(loc)
        << "List of iterables is not supported currently";
  }

  PypetTree* tree = CreatePypetTreeBlock(ctx, loc, 1, 1);

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

void EmitStatements::EmitLoopCommon(
    torch::jit::SourceRange range, const std::function<void()>& emit_body,
    const SugaredValuePtr& iter_val,
    c10::optional<torch::jit::List<torch::jit::Expr>> targets,
    c10::optional<torch::jit::Expr> cond) {
  // recursively parse statements.
  emit_body();
}

void EmitStatements::EmitFor(const torch::jit::For& stmt) {
  // assume the format is: for iter_var in range(a, b, c)
  const torch::jit::List<torch::jit::Expr>& targets = stmt.targets();
  const torch::jit::List<torch::jit::Expr>& itrs = stmt.itrs();

  CHECK_EQ(targets.size(), 1) << "List of iterables is not supported currently";
  CHECK_EQ(itrs.size(), 1) << "List of iterables is not supported currently";

  PypetTree* tree =
      CreatePypetTree(ctx, stmt.range(), PypetTreeType::PYPET_TREE_FOR);
  PypetExpr* iv = ExtractAccessExpr(targets[0]);

  CHECK(itrs[0].kind() == torch::jit::TK_APPLY);
  torch::jit::Apply apply = torch::jit::Apply(itrs[0]);
  CHECK(apply.callee().kind() == torch::jit::TK_VAR);

  const torch::jit::List<torch::jit::Expr>& args = apply.inputs();

  PypetExpr* init = nullptr;
  PypetExpr* bound = nullptr;
  PypetExpr* cond = nullptr;
  PypetExpr* inc = nullptr;

  switch (args.size()) {
    case 1: {
      init = PypetExprFromIntVal(0);
      bound = ExtractAccessExpr(args[0]);
      inc = PypetExprFromIntVal(1);
      break;
    }
    case 2: {
      init = ExtractAccessExpr(args[0]);
      bound = ExtractAccessExpr(args[1]);
      inc = PypetExprFromIntVal(1);
      break;
    }
    case 3: {
      init = ExtractAccessExpr(args[0]);
      bound = ExtractAccessExpr(args[1]);
      inc = ExtractAccessExpr(args[2]);
      break;
    }
    default:
      LOG(ERROR) << "Range parameter num: " << args.size();
      break;
  }
  // TODO: or PYPET_GT
  cond = BuildPypetBinaryOpExpr(PypetOpType::PYPET_LT, iv, bound);

  tree->ast.Loop.iv = iv;
  tree->ast.Loop.init = init;
  tree->ast.Loop.cond = cond;
  tree->ast.Loop.inc = inc;
  //   emitter(stmt.body());
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

void EmitStatements::EmitIf(const torch::jit::If& stmt) {}
void EmitStatements::EmitWhile(const torch::jit::While& stmt) {
  throw std::invalid_argument("while statement is not supported.");
}

void EmitStatements::EmitAssignment(const torch::jit::Assign& stmt) {}
void EmitStatements::EmitAugAssignment(const torch::jit::AugAssign& stmt) {}
void EmitStatements::EmitRaise(const torch::jit::Raise& stmt) {}
void EmitStatements::EmitAssert(const torch::jit::Assert& stmt) {}
void EmitStatements::EmitReturn(const torch::jit::Return& stmt) {}
void EmitStatements::EmitContinue(const torch::jit::Continue& stmt) {}
void EmitStatements::EmitBreak(const torch::jit::Break& stmt) {}
void EmitStatements::EmitClosure(const torch::jit::Def& stmt) {}
void EmitStatements::EmitDelete(const torch::jit::Delete& stmt) {}
void EmitStatements::EmitExpr(const torch::jit::Expr& tree) {}

}  // namespace pypet
}  // namespace pypoly
