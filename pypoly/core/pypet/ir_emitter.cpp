#include "pypoly/core/pypet/ir_emitter.h"

namespace pypoly {
namespace pypet {

namespace {

void isl_printer_to_stdout(struct isl_printer* p) {
  char* str = isl_printer_get_str(p);
  isl_printer_free(p);
  puts(str);
  free(str);
}

void print_isl_multi_pw_aff(isl_multi_pw_aff* pw_multi_aff) {
  isl_printer* p = isl_printer_to_str(isl_multi_pw_aff_get_ctx(pw_multi_aff));
  p = isl_printer_print_multi_pw_aff(p, pw_multi_aff);
  isl_printer_to_stdout(p);
}

void print_isl_pw_aff(isl_pw_aff* pw_aff) {
  isl_printer* p = isl_printer_to_str(isl_pw_aff_get_ctx(pw_aff));
  p = isl_printer_print_pw_aff(p, pw_aff);
  isl_printer_to_stdout(p);
}

PypetExpr* ExtractExpr(isl_ctx* ctx, const torch::jit::Expr& expr);

PypetExpr* PypetExprAccessSetIndex(PypetExpr* expr, isl_multi_pw_aff* index) {
  expr = PypetExprCow(expr);
  CHECK(expr);
  CHECK(index);
  CHECK(expr->type == PYPET_EXPR_ACCESS);
  if (expr->acc.index != nullptr) {
    isl_multi_pw_aff_free(expr->acc.index);
  }
  expr->acc.index = index;
  expr->acc.depth = isl_multi_pw_aff_dim(index, isl_dim_out);
  return expr;
}

PypetExpr* PypetExprFromIndex(isl_multi_pw_aff* index) {
  CHECK(index);
  isl_ctx* ctx = isl_multi_pw_aff_get_ctx(index);
  PypetExpr* expr = PypetExprAlloc(ctx, PYPET_EXPR_ACCESS);
  CHECK(expr);
  expr->acc.read = 1;
  expr->acc.write = 0;
  expr->acc.index = nullptr;
  return PypetExprAccessSetIndex(expr, index);
}

PypetExpr* ExtractIndexExprFromVar(isl_ctx* ctx, const torch::jit::Var& expr) {
  CHECK(expr.kind() == torch::jit::TK_VAR);
  torch::jit::Ident ident_expr = torch::jit::Ident(expr.name());
  const std::string& name = ident_expr.name();
  isl_id* id = isl_id_alloc(ctx, name.c_str(), nullptr);
  isl_space* space = isl_space_alloc(ctx, 0, 0, 0);
  space = isl_space_set_tuple_id(space, isl_dim_out, id);
  return PypetExprFromIndex(isl_multi_pw_aff_zero(space));
}

PypetExpr* ExtractIndexExprFromConst(isl_ctx* ctx,
                                     const torch::jit::Const& expr) {
  CHECK(expr.isIntegral());
  return PypetExprFromIntVal(ctx, static_cast<int>(expr.asIntegral()));
}

PypetExpr* ExtractIndexExpr(isl_ctx* ctx, const torch::jit::Expr& expr);

PypetExpr* PypetExprSetNArgs(PypetExpr* expr, int n) {
  CHECK(expr);
  if (expr->arg_num == n) {
    return expr;
  }
  expr = PypetExprCow(expr);
  CHECK(expr);
  if (n < expr->arg_num) {
    for (int i = n; i < expr->arg_num; ++i) {
      PypetExprFree(expr->args[i]);
    }
    expr->arg_num = n;
    return expr;
  }
  PypetExpr** args = isl_realloc_array(expr->ctx, expr->args, PypetExpr*, n);
  CHECK(args);
  expr->args = args;
  for (int i = expr->arg_num; i < n; ++i) {
    expr->args[i] = nullptr;
  }
  expr->arg_num = n;
  return expr;
}

PypetExpr* PypetExprCopy(PypetExpr* expr) {
  CHECK(expr);
  ++expr->ref;
  return expr;
}

PypetExpr* PypetExprGetArg(PypetExpr* expr, int pos) {
  CHECK(expr);
  CHECK_GE(pos, 0);
  CHECK_LT(pos, expr->arg_num);
  return PypetExprCopy(expr->args[pos]);
}

PypetExpr* PypetExprSetArg(PypetExpr* expr, int pos, PypetExpr* arg) {
  CHECK(expr);
  CHECK(arg);
  CHECK_GE(pos, 0);
  CHECK_LT(pos, expr->arg_num);

  if (expr->args[pos] == arg) {
    PypetExprFree(arg);
    return expr;
  }

  expr = PypetExprCow(expr);
  CHECK(expr);
  PypetExprFree(expr->args[pos]);
  expr->args[pos] = arg;
  return expr;
}

isl_space* PypetExprAccessGetAugmentedDomainSpace(PypetExpr* expr) {
  CHECK(expr);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  isl_space* space = isl_multi_pw_aff_get_space(expr->acc.index);
  space = isl_space_domain(space);
  return space;
}

isl_space* PypetExprAccessGetDomainSpace(PypetExpr* expr) {
  isl_space* space = PypetExprAccessGetAugmentedDomainSpace(expr);
  if (isl_space_is_wrapping(space) == true) {
    space = isl_space_domain(isl_space_unwrap(space));
  }
  return space;
}

PypetExpr* PypetExprAccessPullbackMultiAff(PypetExpr* expr,
                                           isl_multi_aff* multi_aff) {
  expr = PypetExprCow(expr);
  CHECK(expr);
  CHECK(multi_aff);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  for (int type = PypetExprAccessType::PYPET_EXPR_ACCESS_BEGIN;
       type < PypetExprAccessType::PYPET_EXPR_ACCESS_END; ++type) {
    if (expr->acc.access[type] == nullptr) {
      continue;
    }
    expr->acc.access[type] = isl_union_map_preimage_domain_multi_aff(
        expr->acc.access[type], isl_multi_aff_copy(multi_aff));
    CHECK(expr->acc.access[type]);
  }
  expr->acc.index =
      isl_multi_pw_aff_pullback_multi_aff(expr->acc.index, multi_aff);
  std::cout << "pullback" << std::endl;
  print_isl_multi_pw_aff(expr->acc.index);
  CHECK(expr->acc.index);
  return expr;
}

PypetExpr* PypetExprInsertArg(PypetExpr* expr, int pos, PypetExpr* arg) {
  CHECK(expr);
  CHECK(arg);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  int n = expr->arg_num;
  CHECK_GE(pos, 0);
  CHECK_LE(pos, n);
  expr = PypetExprSetNArgs(expr, n + 1);
  for (int i = n; i > pos; --i) {
    PypetExprSetArg(expr, i, PypetExprGetArg(expr, i - 1));
  }
  expr = PypetExprSetArg(expr, pos, arg);

  isl_space* space = PypetExprAccessGetDomainSpace(expr);
  space = isl_space_from_domain(space);
  space = isl_space_add_dims(space, isl_dim_out, n + 1);

  isl_multi_aff* multi_aff = nullptr;
  if (n == 0) {
    multi_aff = isl_multi_aff_domain_map(space);
  } else {
    multi_aff = isl_multi_aff_domain_map(isl_space_copy(space));
    isl_multi_aff* new_multi_aff = isl_multi_aff_range_map(space);
    space = isl_space_range(isl_multi_aff_get_space(new_multi_aff));
    isl_multi_aff* proj =
        isl_multi_aff_project_out_map(space, isl_dim_set, pos, 1);
    new_multi_aff = isl_multi_aff_pullback_multi_aff(proj, new_multi_aff);
    multi_aff = isl_multi_aff_range_product(multi_aff, new_multi_aff);
  }
  return PypetExprAccessPullbackMultiAff(expr, multi_aff);
}

isl_multi_pw_aff* PypetArraySubscript(isl_multi_pw_aff* base,
                                      isl_pw_aff* index) {
  int member_access = isl_multi_pw_aff_range_is_wrapping(base);
  CHECK_GE(member_access, 0);
  std::cout << "array subscript" << std::endl;
  print_isl_multi_pw_aff(base);
  print_isl_pw_aff(index);

  if (member_access > 0) {
    isl_id* id = isl_multi_pw_aff_get_tuple_id(base, isl_dim_out);
    isl_multi_pw_aff* domain = isl_multi_pw_aff_copy(base);
    domain = isl_multi_pw_aff_range_factor_domain(domain);
    isl_multi_pw_aff* range = isl_multi_pw_aff_range_factor_range(base);
    range = PypetArraySubscript(range, index);
    isl_multi_pw_aff* access = isl_multi_pw_aff_range_product(domain, range);
    access = isl_multi_pw_aff_set_tuple_id(access, isl_dim_out, id);
    print_isl_multi_pw_aff(access);
    return access;
  } else {
    isl_id* id = isl_multi_pw_aff_get_tuple_id(base, isl_dim_set);
    isl_set* domain = isl_pw_aff_nonneg_set(isl_pw_aff_copy(index));
    index = isl_pw_aff_intersect_domain(index, domain);
    isl_multi_pw_aff* access = isl_multi_pw_aff_from_pw_aff(index);
    access = isl_multi_pw_aff_flat_range_product(base, access);
    access = isl_multi_pw_aff_set_tuple_id(access, isl_dim_set, id);
    print_isl_multi_pw_aff(access);
    return access;
  }
}

PypetExpr* PypetExprAccessSubscript(PypetExpr* expr, PypetExpr* index) {
  expr = PypetExprCow(expr);
  CHECK(expr);
  CHECK(index);
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  int n = expr->arg_num;
  expr = PypetExprInsertArg(expr, n, index);
  isl_space* space = isl_multi_pw_aff_get_domain_space(expr->acc.index);
  isl_local_space* local_space = isl_local_space_from_space(space);
  isl_pw_aff* pw_aff =
      isl_pw_aff_from_aff(isl_aff_var_on_domain(local_space, isl_dim_set, n));
  expr->acc.index = PypetArraySubscript(expr->acc.index, pw_aff);
  CHECK(expr->acc.index);
  return expr;
}

PypetExpr* ExtractIndexExprFromSubscript(isl_ctx* ctx,
                                         const torch::jit::Subscript& expr) {
  const torch::jit::Expr& base = expr.value();
  const torch::jit::List<torch::jit::Expr>& indexes = expr.subscript_exprs();
  CHECK_EQ(indexes.size(), 1);
  PypetExpr* base_expr = ExtractIndexExpr(ctx, base);
  PypetExpr* index_expr = ExtractExpr(ctx, indexes[0]);
  std::cout << base_expr;
  std::cout << index_expr;
  return PypetExprAccessSubscript(base_expr, index_expr);
}

PypetExpr* ExtractIndexExpr(isl_ctx* ctx, const torch::jit::Expr& expr) {
  switch (expr.kind()) {
    case torch::jit::TK_VAR: {
      torch::jit::Var var_expr = torch::jit::Var(expr);
      return ExtractIndexExprFromVar(ctx, var_expr);
    }
    case torch::jit::TK_CONST: {
      torch::jit::Const const_expr = torch::jit::Const(expr);
      return ExtractIndexExprFromConst(ctx, const_expr);
    }
    case torch::jit::TK_SUBSCRIPT: {
      torch::jit::Subscript subscript_expr = torch::jit::Subscript(expr);
      return ExtractIndexExprFromSubscript(ctx, subscript_expr);
    }
    default:
      LOG(FATAL) << "Unexpected expr kind "
                 << torch::jit::kindToString(expr.kind());
      break;
  }
  return nullptr;
}

PypetExpr* PypetExprAccessFromIndex(const torch::jit::Expr& expr,
                                    PypetExpr* index) {
  // TODO: depth
  // TODO: type_size
  return index;
}

PypetExpr* ExtractAccessExpr(isl_ctx* ctx, const torch::jit::Expr& expr) {
  PypetExpr* index = ExtractIndexExpr(ctx, expr);
  if (index->type == PypetExprType::PYPET_EXPR_INT) {
    return index;
  }
  return PypetExprAccessFromIndex(expr, index);
}

PypetExpr* BuildPypetBinaryOpExpr(isl_ctx* ctx, PypetOpType op_type,
                                  PypetExpr* lhs, PypetExpr* rhs) {
  PypetExpr* expr = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_OP);
  // TODO: type_size
  expr->arg_num = 2;
  expr->args = new PypetExpr*[2];
  expr->args[0] = lhs;
  expr->args[1] = rhs;
  expr->op = op_type;
  return expr;
}

PypetExpr* ExtractAssignExpr(isl_ctx* ctx, const torch::jit::Assign& stmt) {
  const torch::jit::Expr& lhs = stmt.lhs();
  const torch::jit::Maybe<torch::jit::Expr>& rhs = stmt.rhs();
  const torch::jit::Maybe<torch::jit::Expr>& type = stmt.type();
  CHECK(rhs.present());
  CHECK(!type.present());
  PypetExpr* ret = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_OP);
  // TODO: type_size
  ret->arg_num = 2;
  ret->args = new PypetExpr*[2];
  ret->args[0] = ExtractExpr(ctx, lhs);
  ret->args[1] = ExtractExpr(ctx, rhs.get());
  ret->op = PypetOpType::PYPET_ASSIGN;
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

PypetExpr* ExtractBinaryExpr(isl_ctx* ctx, const torch::jit::Expr& expr) {
  torch::jit::BinOp bin_expr = torch::jit::BinOp(expr);
  PypetExpr* ret = PypetExprAlloc(ctx, PypetExprType::PYPET_EXPR_OP);
  ret->op = TorchKind2PypetOpType(expr.kind());
  ret->arg_num = 2;
  ret->args = new PypetExpr*[2];
  ret->args[0] = ExtractExpr(ctx, bin_expr.lhs());
  ret->args[1] = ExtractExpr(ctx, bin_expr.rhs());
  return ret;
}

PypetExpr* ExtractSelectExpr(isl_ctx* ctx, const torch::jit::Expr& expr) {
  // TODO
  return nullptr;
}

PypetExpr* ExtractApplyExpr(isl_ctx* ctx, const torch::jit::Expr& expr) {
  // TODO
  return nullptr;
}

PypetExpr* ExtractExpr(isl_ctx* ctx, const torch::jit::Expr& expr) {
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
    default:
      LOG(FATAL) << torch::jit::kindToString(expr.kind());
      break;
  }
  return nullptr;
}

}  // namespace

std::vector<PypetTree*> EmitStatements::operator()(
    const torch::jit::List<torch::jit::Stmt>& statements) {
  std::vector<PypetTree*> ret(statements.size(), nullptr);
  for (size_t i = 0; i < statements.size(); ++i) {
    ret[i] = EmitStatement(statements[i]);
    // std::cout << ret[i];
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

void EmitStatements::EmitForImpl(
    const torch::jit::List<torch::jit::Expr>& targets,
    const torch::jit::List<torch::jit::Expr>& itrs,
    const torch::jit::SourceRange& loc,
    const std::function<void()>& emit_body) {
  if (itrs.size() != 1) {
    throw torch::jit::ErrorReport(loc)
        << "List of iterables is not supported currently";
  }

  // PypetTree* tree = CreatePypetTreeBlock(ctx, loc, 1, 1);

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

PypetTree* EmitStatements::EmitFor(const torch::jit::For& stmt) {
  // assume the format is: for iter_var in range(a, b, c)
  const torch::jit::List<torch::jit::Expr>& targets = stmt.targets();
  const torch::jit::List<torch::jit::Expr>& itrs = stmt.itrs();
  CHECK_EQ(targets.size(), 1) << "List of iterables is not supported currently";
  CHECK_EQ(itrs.size(), 1) << "List of iterables is not supported currently";

  PypetTree* tree =
      CreatePypetTree(ctx, &stmt.range(), PypetTreeType::PYPET_TREE_FOR);
  PypetExpr* iv = ExtractAccessExpr(ctx, targets[0]);

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
      bound = ExtractAccessExpr(ctx, args[0]);
      inc = PypetExprFromIntVal(ctx, 1);
      break;
    }
    case 2: {
      init = ExtractAccessExpr(ctx, args[0]);
      bound = ExtractAccessExpr(ctx, args[1]);
      inc = PypetExprFromIntVal(ctx, 1);
      break;
    }
    case 3: {
      init = ExtractAccessExpr(ctx, args[0]);
      bound = ExtractAccessExpr(ctx, args[1]);
      inc = ExtractAccessExpr(ctx, args[2]);
      break;
    }
    default:
      LOG(FATAL) << "Range parameter num: " << args.size();
      break;
  }
  // TODO: or PYPET_GT
  cond = BuildPypetBinaryOpExpr(ctx, PypetOpType::PYPET_LT, iv, bound);

  tree->ast.Loop.iv = iv;
  tree->ast.Loop.init = init;
  tree->ast.Loop.cond = cond;
  tree->ast.Loop.inc = inc;

  tree->ast.Loop.body = EmitBlockStatements(stmt.body());
  return tree;
}

std::shared_ptr<SugaredValue> EmitStatements::EmitApplyExpr(
    torch::jit::Apply& apply, size_t n_binders,
    const torch::jit::TypePtr& type_hint) {
  auto sv = EmitSugaredExpr(apply.callee(), 1);

  auto loc = apply.callee().range();
  auto special_form = dynamic_cast<torch::jit::SpecialFormValue*>(sv.get());
  if (special_form != nullptr) {
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
  return nullptr;
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

}  // namespace pypet
}  // namespace pypoly
