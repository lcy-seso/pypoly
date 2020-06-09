#include "pypoly/core/pypet/ir_emitter.h"

namespace pypoly {
namespace pypet {

namespace {

void MarkWrite(PypetExpr* access) {
  access->acc.read = 0;
  access->acc.write = 1;
}

void MarkRead(PypetExpr* access) {
  access->acc.read = 1;
  access->acc.write = 0;
}

PypetExpr* PypetExprAccessFromIndex(const torch::jit::Expr& expr,
                                    PypetExpr* index) {
  // TODO(yizhu1): fill in depth field
  // TODO(yizhu1): fill in type_size field
  return index;
}

PypetExpr* ExtractIndexExprFromIdent(isl_ctx* ctx,
                                     const torch::jit::Ident& ident_expr) {
  const std::string& name = ident_expr.name();
  isl_id* id = isl_id_alloc(ctx, name.c_str(), nullptr);
  isl_space* space = isl_space_alloc(ctx, 0, 0, 0);
  space = isl_space_set_tuple_id(space, isl_dim_out, id);
  return PypetExprFromIndex(isl_multi_pw_aff_zero(space));
}

// TODO(yizhu1): check the type_size field
PypetExpr* ExtractIndexExprFromVar(isl_ctx* ctx, const torch::jit::Var& expr) {
  CHECK(expr.kind() == torch::jit::TK_VAR);
  torch::jit::Ident ident_expr(expr.name());
  PypetExpr* ret = ExtractIndexExprFromIdent(ctx, ident_expr);
  ret->type_size = -32;
  return ret;
}

PypetExpr* ExtractIndexExprFromConst(isl_ctx* ctx,
                                     const torch::jit::Const& expr) {
  CHECK(expr.isIntegral());
  PypetExpr* ret =
      PypetExprFromIntVal(ctx, static_cast<int>(expr.asIntegral()));
  ret->type_size = -32;
  return ret;
}

PypetOpType TorchKind2PypetOpType(int kind) {
  switch (kind) {
    case torch::jit::TK_AND:
      return PypetOpType::PYPET_AND;
    case torch::jit::TK_OR:
      return PypetOpType::PYPET_OR;
    case '<':
      return PypetOpType::PYPET_LT;
    case '>':
      return PypetOpType::PYPET_GT;
    case torch::jit::TK_EQ:
      return PypetOpType::PYPET_EQ;
    case torch::jit::TK_LE:
      return PypetOpType::PYPET_LE;
    case torch::jit::TK_GE:
      return PypetOpType::PYPET_GE;
    case torch::jit::TK_NE:
      return PypetOpType::PYPET_NE;
    case '+':
      return PypetOpType::PYPET_ADD;
    case '-':
      return PypetOpType::PYPET_SUB;
    case '*':
      return PypetOpType::PYPET_MUL;
    case '/':
      return PypetOpType::PYPET_DIV;
    default:
      break;
  }
  UNIMPLEMENTED();
  return PypetOpType::PYPET_UNKNOWN;
}

}  // namespace

std::vector<PypetTree*> EmitStatements::operator()(
    const torch::jit::List<torch::jit::Stmt>& statements) {
  std::vector<PypetTree*> ret(statements.size(), nullptr);
  for (size_t i = 0; i < statements.size(); ++i) {
    ret[i] = EmitStatement(statements[i]);
    std::cout << ret[i];
  }
  return ret;
}

PypetTree* EmitStatements::EmitBlockStatements(
    const torch::jit::List<torch::jit::Stmt>& statements) {
  PypetTree* tree = CreatePypetTreeBlock(ctx, 1, statements.size());
  for (size_t i = 0; i < statements.size(); ++i) {
    tree->ast.Block.children[i] = EmitStatement(statements[i]);
  }
  return tree;
}

PypetTree* EmitStatements::EmitStatement(const torch::jit::Stmt& stmt) {
  switch (stmt.kind()) {
    case torch::jit::TK_IF:
      return EmitIf(torch::jit::If(stmt));
    case torch::jit::TK_WHILE:
      return EmitWhile(torch::jit::While(stmt));
    case torch::jit::TK_FOR:
      return EmitFor(torch::jit::For(stmt));
    case torch::jit::TK_ASSIGN:
      return EmitAssignment(torch::jit::Assign(stmt));
    case torch::jit::TK_AUG_ASSIGN:
      return EmitAugAssignment(torch::jit::AugAssign(stmt));
    case torch::jit::TK_EXPR_STMT: {
      auto expr = torch::jit::ExprStmt(stmt).expr();
      return EmitExpr(expr);
    }
    case torch::jit::TK_RAISE:
      return EmitRaise(torch::jit::Raise(stmt));
    case torch::jit::TK_ASSERT:
      return EmitAssert(torch::jit::Assert(stmt));
    case torch::jit::TK_RETURN:
      return EmitReturn(torch::jit::Return(stmt));
    case torch::jit::TK_CONTINUE:
      return EmitContinue(torch::jit::Continue(stmt));
    case torch::jit::TK_BREAK:
      return EmitBreak(torch::jit::Break(stmt));
    case torch::jit::TK_PASS:
      break;
    case torch::jit::TK_DEF:
      return EmitClosure(torch::jit::Def(stmt));
    case torch::jit::TK_DELETE:
      return EmitDelete(torch::jit::Delete(stmt));
    default:
      UNIMPLEMENTED();
      break;
  }
  return nullptr;
}

PypetTree* EmitStatements::EmitFor(const torch::jit::For& stmt) {
  // assume the format is: for iter_var in range(a, b, c)
  const torch::jit::List<torch::jit::Expr>& targets = stmt.targets();
  const torch::jit::List<torch::jit::Expr>& itrs = stmt.itrs();
  CHECK_EQ(targets.size(), 1) << "List of iterables is not supported currently";
  CHECK_EQ(itrs.size(), 1) << "List of iterables is not supported currently";

  PypetTree* tree =
      CreatePypetTree(ctx, &stmt.range(), PypetTreeType::PYPET_TREE_FOR);
  tree->ast.Loop.declared = 1;
  PypetExpr* iv = ExtractAccessExpr(ctx, targets[0]);
  MarkWrite(iv);

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
      init = PypetExprFromIntVal(ctx, 0);
      bound = ExtractExpr(ctx, args[0]);
      inc = PypetExprFromIntVal(ctx, 1);
      break;
    }
    case 2: {
      init = ExtractExpr(ctx, args[0]);
      bound = ExtractExpr(ctx, args[1]);
      inc = PypetExprFromIntVal(ctx, 1);
      break;
    }
    case 3: {
      init = ExtractExpr(ctx, args[0]);
      bound = ExtractExpr(ctx, args[1]);
      inc = ExtractExpr(ctx, args[2]);
      break;
    }
    default:
      LOG(FATAL) << "Range parameter num: " << args.size();
      break;
  }
  // TODO(yizhu1): or PYPET_GT
  PypetExpr* iv2 = PypetExprDup(iv);
  MarkRead(iv2);
  cond = BuildPypetBinaryOpExpr(ctx, PypetOpType::PYPET_LT, iv2, bound);

  tree->ast.Loop.iv = iv;
  tree->ast.Loop.init = init;
  tree->ast.Loop.cond = cond;
  tree->ast.Loop.inc = inc;

  tree->ast.Loop.body = EmitBlockStatements(stmt.body());
  return tree;
}

PypetTree* EmitStatements::EmitIf(const torch::jit::If& stmt) {
  bool has_false_branch = stmt.falseBranch().size() > 0;
  PypetTree* tree =
      CreatePypetTree(ctx, &stmt.range(),
                      has_false_branch ? PypetTreeType::PYPET_TREE_IF_ELSE
                                       : PypetTreeType::PYPET_TREE_IF);
  tree->ast.IfElse.cond = ExtractExpr(ctx, stmt.cond());
  tree->ast.IfElse.if_body = EmitBlockStatements(stmt.trueBranch());
  if (has_false_branch == true) {
    tree->ast.IfElse.else_body = EmitBlockStatements(stmt.falseBranch());
  }
  return tree;
}

PypetTree* EmitStatements::EmitWhile(const torch::jit::While& stmt) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetTree* EmitStatements::EmitAssignment(const torch::jit::Assign& stmt) {
  PypetTree* tree =
      CreatePypetTree(ctx, &stmt.range(), PypetTreeType::PYPET_TREE_EXPR);
  tree->ast.Expr.expr = ExtractAssignExpr(ctx, stmt);
  return tree;
}

PypetTree* EmitStatements::EmitAugAssignment(
    const torch::jit::AugAssign& stmt) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetTree* EmitStatements::EmitRaise(const torch::jit::Raise& stmt) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetTree* EmitStatements::EmitAssert(const torch::jit::Assert& stmt) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetTree* EmitStatements::EmitReturn(const torch::jit::Return& stmt) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetTree* EmitStatements::EmitContinue(const torch::jit::Continue& stmt) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetTree* EmitStatements::EmitBreak(const torch::jit::Break& stmt) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetTree* EmitStatements::EmitClosure(const torch::jit::Def& stmt) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetTree* EmitStatements::EmitDelete(const torch::jit::Delete& stmt) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetTree* EmitStatements::EmitExpr(const torch::jit::Expr& tree) {
  UNIMPLEMENTED();
  return nullptr;
}

PypetExpr* EmitStatements::ExtractAccessExpr(isl_ctx* ctx,
                                             const torch::jit::Expr& expr) {
  PypetExpr* index = ExtractIndexExpr(ctx, expr);
  if (index->type == PypetExprType::PYPET_EXPR_INT) {
    return index;
  }
  return PypetExprAccessFromIndex(expr, index);
}

PypetExpr* EmitStatements::ExtractAssignExpr(isl_ctx* ctx,
                                             const torch::jit::Assign& stmt) {
  const torch::jit::Expr& lhs = stmt.lhs();
  const torch::jit::Maybe<torch::jit::Expr>& rhs = stmt.rhs();
  const torch::jit::Maybe<torch::jit::Expr>& type = stmt.type();
  CHECK(rhs.present());
  CHECK(!type.present());
  PypetExpr* ret = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_OP);
  // TODO(yizhu1): fill in type_size field
  ret->arg_num = 2;
  ret->args = isl_alloc_array(ctx, PypetExpr*, ret->arg_num);
  ret->args[0] = ExtractExpr(ctx, lhs);
  ret->args[1] = ExtractExpr(ctx, rhs.get());
  ret->op = PypetOpType::PYPET_ASSIGN;
  if (ret->args[0]->type == PypetExprType::PYPET_EXPR_ACCESS) {
    MarkWrite(ret->args[0]);
  }
  return ret;
}

PypetExpr* EmitStatements::ExtractIndexExprFromSubscript(
    isl_ctx* ctx, const torch::jit::Subscript& expr) {
  const torch::jit::Expr& base = expr.value();
  const torch::jit::List<torch::jit::Expr>& indexes = expr.subscript_exprs();
  CHECK_EQ(indexes.size(), 1);
  PypetExpr* base_expr = ExtractIndexExpr(ctx, base);
  PypetExpr* index_expr = ExtractExpr(ctx, indexes[0]);
  index_expr->type_size = -32;
  return PypetExprAccessSubscript(base_expr, index_expr);
}

PypetExpr* EmitStatements::ExtractIndexExpr(isl_ctx* ctx,
                                            const torch::jit::Expr& expr) {
  switch (expr.kind()) {
    case torch::jit::TK_VAR: {
      torch::jit::Var var_expr(expr);
      return ExtractIndexExprFromVar(ctx, var_expr);
    }
    case torch::jit::TK_CONST: {
      torch::jit::Const const_expr(expr);
      return ExtractIndexExprFromConst(ctx, const_expr);
    }
    case torch::jit::TK_SUBSCRIPT: {
      torch::jit::Subscript subscript_expr(expr);
      return ExtractIndexExprFromSubscript(ctx, subscript_expr);
    }
    case '.':
      return ExtractSelectExpr(ctx, expr);
    default:
      LOG(FATAL) << "Unexpected expr kind "
                 << torch::jit::kindToString(expr.kind());
      break;
  }
  return nullptr;
}

PypetExpr* EmitStatements::ExtractBinaryExpr(isl_ctx* ctx,
                                             const torch::jit::Expr& expr) {
  torch::jit::BinOp bin_expr(expr);
  PypetExpr* ret = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_OP);
  ret->op = TorchKind2PypetOpType(expr.kind());
  ret->arg_num = 2;
  ret->args = isl_alloc_array(ctx, PypetExpr*, 2);
  ret->args[0] = ExtractExpr(ctx, bin_expr.lhs());
  ret->args[1] = ExtractExpr(ctx, bin_expr.rhs());
  return ret;
}

PypetExpr* EmitStatements::ExtractSelectExpr(isl_ctx* ctx,
                                             const torch::jit::Expr& expr) {
  torch::jit::Select select_expr(expr);
  const torch::jit::Expr& base = select_expr.value();
  PypetExpr* base_index = ExtractIndexExpr(ctx, base);
  const torch::jit::Ident& selector = select_expr.selector();

  isl_id* id = isl_id_alloc(ctx, selector.name().c_str(),
                            const_cast<void*>(static_cast<const void*>(&expr)));
  return PypetExprAccessMember(base_index, id);
}

PypetExpr* EmitStatements::ExtractApplyExpr(isl_ctx* ctx,
                                            const torch::jit::Expr& expr) {
  torch::jit::Apply apply_expr(expr);
  PypetExpr* ret = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_OP);
  const torch::jit::Expr& callee = apply_expr.callee();
  const torch::jit::List<torch::jit::Expr>& inputs = apply_expr.inputs();
  const torch::jit::List<torch::jit::Attribute>& attributes =
      apply_expr.attributes();
  // TODO(yizhu1): replace PYPET_APPLY with a specific structure for function
  // call
  ret->arg_num = 1 + inputs.size() + attributes.size();
  ret->args = isl_alloc_array(ctx, PypetExpr*, ret->arg_num);
  ret->args[0] = ExtractExpr(ctx, callee);
  ret->op = PypetOpType::PYPET_APPLY;
  for (int i = 0; i < inputs.size(); ++i) {
    ret->args[i + 1] = ExtractExpr(ctx, inputs[i]);
  }
  for (int i = 0; i < attributes.size(); ++i) {
    ret->args[i + 1 + inputs.size()] = ExtractAttributeExpr(ctx, attributes[i]);
  }
  return ret;
}

PypetExpr* EmitStatements::ExtractListLiteralExpr(
    isl_ctx* ctx, const torch::jit::Expr& expr) {
  torch::jit::ListLiteral list_literal_expr(expr);
  PypetExpr* ret = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_OP);
  const torch::jit::List<torch::jit::Expr>& inputs = list_literal_expr.inputs();
  ret->arg_num = inputs.size();
  ret->args = isl_alloc_array(ctx, PypetExpr*, inputs.size());
  for (int i = 0; i < inputs.size(); ++i) {
    ret->args[i] = ExtractExpr(ctx, inputs[i]);
  }
  ret->op = PypetOpType::PYPET_LIST_LITERAL;
  return ret;
}

PypetExpr* EmitStatements::ExtractAttributeExpr(
    isl_ctx* ctx, const torch::jit::Attribute& attribute_expr) {
  PypetExpr* ret = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_OP);
  ret->arg_num = 2;
  ret->args = isl_alloc_array(ctx, PypetExpr*, ret->arg_num);
  ret->args[0] = ExtractIndexExprFromIdent(ctx, attribute_expr.name());
  ret->args[1] = ExtractExpr(ctx, attribute_expr.value());
  ret->op = PypetOpType::PYPET_ATTRIBUTE;
  return ret;
}

PypetExpr* EmitStatements::ExtractAttributeExpr(isl_ctx* ctx,
                                                const torch::jit::Expr& expr) {
  torch::jit::Attribute attribute_expr(expr);
  return ExtractAttributeExpr(ctx, attribute_expr);
}

PypetExpr* EmitStatements::ExtractExpr(isl_ctx* ctx,
                                       const torch::jit::Expr& expr) {
  switch (expr.kind()) {
    case torch::jit::TK_VAR:
    case torch::jit::TK_CONST:
    case torch::jit::TK_SUBSCRIPT:
      return ExtractIndexExpr(ctx, expr);
    case torch::jit::TK_AND:
    case torch::jit::TK_OR:
    case '<':
    case '>':
    case torch::jit::TK_EQ:
    case torch::jit::TK_LE:
    case torch::jit::TK_GE:
    case torch::jit::TK_NE:
    case '+':
    case '-':
    case '*':
    case '/':
      return ExtractBinaryExpr(ctx, expr);
    case '.':
      return ExtractSelectExpr(ctx, expr);
    case torch::jit::TK_APPLY:
      return ExtractApplyExpr(ctx, expr);
    case torch::jit::TK_LIST_LITERAL:
      return ExtractListLiteralExpr(ctx, expr);
    case torch::jit::TK_ATTRIBUTE:
      return ExtractAttributeExpr(ctx, expr);
    default:
      LOG(FATAL) << torch::jit::kindToString(expr.kind()) << std::endl
                 << expr.range();
      break;
  }
  return nullptr;
}

}  // namespace pypet
}  // namespace pypoly
