#include "pypoly/core/pypet/context.h"

#include "pypoly/core/pypet/aff.h"
#include "pypoly/core/pypet/expr.h"
#include "pypoly/core/pypet/expr_arg.h"
#include "pypoly/core/pypet/isl_printer.h"
#include "pypoly/core/pypet/nest.h"
#include "pypoly/core/pypet/tree.h"

namespace pypoly {
namespace pypet {

namespace {

int AddParameter(PypetExpr* expr, void* user) {
  PypetContext* context = static_cast<PypetContext*>(user);
  if (!PypetExprIsScalarAccess(expr)) {
    // TODO(yizhu): get_array_size
    return 0;
  }
  if (!expr->acc.read) {
    return 0;
  }
  if (expr->type_size == 0) {
    return 0;
  }

  isl_id* id = PypetExprAccessGetId(expr);
  if (PypetContextIsAssigned(context, id)) {
    isl_id_free(id);
    return 0;
  }

  isl_space* space = PypetContextGetSpace(context);
  int pos = isl_space_find_dim_by_id(space, isl_dim_param, id);
  if (pos < 0) {
    pos = isl_space_dim(space, isl_dim_param);
    space = isl_space_add_dims(space, isl_dim_param, 1);
    space = isl_space_set_dim_id(space, isl_dim_param, pos, isl_id_copy(id));
  }

  isl_local_space* ls = isl_local_space_from_space(space);
  isl_aff* aff = isl_aff_var_on_domain(ls, isl_dim_param, pos);
  isl_pw_aff* pa = isl_pw_aff_from_aff(aff);
  context = PypetContextSetValue(context, id, pa);
  return 0;
}

int ClearWrite(PypetExpr* expr, void* user) {
  PypetContext** context = static_cast<PypetContext**>(user);
  if (expr->acc.write == 0) {
    return 0;
  }
  if (PypetExprIsScalarAccess(expr) == 0) {
    return 0;
  }

  // Clear mappings for variables that are written in and red at different
  // locations in the source program, like 'seq_len' in stacked_rnn_example.py.
  isl_id* id = PypetExprAccessGetId(expr);
  if (isl_id_get_user(id)) {
    *context = PypetContextClearValue(*context, id);
  } else {
    isl_id_free(id);
  }
  return 0;
}

PypetExpr* AccessPlugInAffineRead(PypetExpr* expr, void* user) {
  CHECK(expr->type == PypetExprType::PYPET_EXPR_ACCESS);
  PypetContext* context = static_cast<PypetContext*>(user);
  if (expr->acc.write) {
    return expr;
  }
  if (!PypetExprIsScalarAccess(expr)) {
    return expr;
  }

  isl_pw_aff* pw_aff = PypetExprExtractAffine(expr, context);
  if (isl_pw_aff_involves_nan(pw_aff)) {
    isl_pw_aff_free(pw_aff);
    return expr;
  }
  PypetExprFree(expr);
  return PypetExprFromIndex(isl_multi_pw_aff_from_pw_aff(pw_aff));
}

PypetExpr* PlugInAffineRead(PypetExpr* expr, PypetContext* context) {
  return PypetExprMapExprOfType(expr, PypetExprType::PYPET_EXPR_ACCESS,
                                AccessPlugInAffineRead, context);
}

int CheckOnlyAffine(PypetExpr* expr, void* user) {
  int* only_affine = static_cast<int*>(user);
  if (!PypetExprIsAffine(expr)) {
    *only_affine = 0;
    return -1;
  } else {
    *only_affine = 1;
    return 0;
  }
}

bool HasOnlyAffineAccessSubExpr(PypetExpr* expr) {
  int only_affine = -1;
  PypetExprForeachAccessExpr(expr, &CheckOnlyAffine, &only_affine);
  return only_affine > 0;
}

PypetExpr* ExprPlugInAffine(PypetExpr* expr, void* user) {
  CHECK(expr);
  PypetExprType type = expr->type;
  PypetContext* context = static_cast<PypetContext*>(user);
  if (type != PypetExprType::PYPET_EXPR_CALL &&
      type != PypetExprType::PYPET_EXPR_OP) {
    return expr;
  }
  if (type == PypetExprType::PYPET_EXPR_OP &&
      expr->op == PypetOpType::PYPET_LIST_LITERAL) {
    return expr;
  }
  bool contains_access = HasOnlyAffineAccessSubExpr(expr);
  if (!contains_access) {
    return expr;
  }

  isl_pw_aff* pw_aff = PypetExprExtractAffine(expr, context);
  CHECK_EQ(isl_pw_aff_involves_nan(pw_aff), 0) << std::endl << expr;

  PypetExprFree(expr);
  expr = PypetExprFromIndex(isl_multi_pw_aff_from_pw_aff(pw_aff));
  return expr;
}

PypetExpr* PlugInAffine(PypetExpr* expr, PypetContext* context) {
  return PypetExprMapTopDown(expr, ExprPlugInAffine, context);
}

PypetExpr* MergeConditionalAccesses(PypetExpr* expr) {
  // TODO(yizhu1): PYPET_COND
  return expr;
}

PypetExpr* PlugInSummaries(PypetExpr* expr, PypetContext* context) {
  // TODO(yizhu1): function summaries
  return expr;
}

PypetExpr* TreeEvaluateExprWrapper(PypetExpr* expr, void* user) {
  PypetContext* pc = static_cast<PypetContext*>(user);
  return PypetContextEvaluateExpr(pc, expr);
}

}  // namespace

/*
 * create PypetContext from a given domain.
 */
__isl_give PypetContext* CreatePypetContext(__isl_take isl_set* domain) {
  if (!domain) return nullptr;

  PypetContext* pc =
      isl_calloc_type(isl_set_get_ctx(domain), struct PypetContext);
  if (!pc) {
    isl_set_free(domain);
    return nullptr;
  }

  pc->ref = 1;
  pc->domain = domain;
  pc->allow_nested = false;
  pc->assignments.clear();
  pc->extracted_affine.clear();
  return pc;
}

/*
 * Free a reference to "pc" and return nullptr.
 */
__isl_null PypetContext* FreePypetContext(__isl_take PypetContext* pc) {
  if (!pc) return nullptr;
  if (--pc->ref > 0) return nullptr;

  isl_set_free(pc->domain);
  for (auto iter = pc->extracted_affine.begin();
       iter != pc->extracted_affine.end(); ++iter) {
    PypetExprFree(iter->first);
    isl_pw_aff_free(iter->second);
  }
  pc->extracted_affine.clear();
  for (auto iter = pc->assignments.begin(); iter != pc->assignments.end();
       ++iter) {
    isl_id_free(iter->first);
    isl_pw_aff_free(iter->second);
  }
  pc->assignments.clear();
  free(pc);
  return nullptr;
}

/*
 * Add an assignment to "pc" for each parameter in "tree".
 */
__isl_give PypetContext* PypetContextAddParameter(__isl_keep PypetTree* tree,
                                                  __isl_keep PypetContext* pc) {
  CHECK_GE(PypetTreeForeachAccessExpr(tree, AddParameter, pc), 0);
  pc = PypetContextClearWritesInTree(pc, tree);
  return pc;
}

PypetContext* ContextAlloc(isl_set* domain, bool allow_nested) {
  PypetContext* context =
      isl_calloc_type(isl_set_get_ctx(domain), struct PypetContext);
  CHECK(context);

  context->ref = 1;
  context->domain = domain;
  context->allow_nested = allow_nested;
  context->assignments.clear();
  context->extracted_affine.clear();
  return context;
}

PypetContext* PypetContextDup(PypetContext* context) {
  CHECK(context);
  PypetContext* dup =
      ContextAlloc(isl_set_copy(context->domain), context->allow_nested);
  for (auto iter = context->assignments.begin();
       iter != context->assignments.end(); ++iter) {
    dup->assignments.insert(
        {isl_id_copy(iter->first), isl_pw_aff_copy(iter->second)});
  }
  return dup;
}

PypetContext* PypetContextCopy(PypetContext* context) {
  CHECK(context);
  ++context->ref;
  return context;
}

PypetContext* PypetContextCow(PypetContext* context) {
  CHECK(context);
  if (context->ref == 1) {
    for (auto iter = context->extracted_affine.begin();
         iter != context->extracted_affine.end(); ++iter) {
      PypetExprFree(iter->first);
      isl_pw_aff_free(iter->second);
    }
    context->extracted_affine.clear();
    return context;
  }
  --context->ref;
  return PypetContextDup(context);
}

int PypetContextDim(PypetContext* pc) {
  CHECK(pc);
  return isl_set_dim(pc->domain, isl_dim_set);
}

isl_space* PypetContextGetSpace(PypetContext* context) {
  CHECK(context);
  isl_space* space = isl_set_get_space(context->domain);
  space = PypetNestedRemoveFromSpace(space);
  return space;
}

int PypetContextGetDim(PypetContext* context) {
  CHECK(context);
  return isl_set_dim(context->domain, isl_dim_set);
}

isl_set* PypetContextGetDomain(PypetContext* context) {
  CHECK(context);
  return isl_set_copy(context->domain);
}

PypetContext* ExtendDomain(PypetContext* context, isl_id* id) {
  CHECK(context);
  context = PypetContextCow(context);
  CHECK(id);
  int pos = PypetContextDim(context);
  context->domain = isl_set_add_dims(context->domain, isl_dim_set, 1);
  context->domain = isl_set_set_dim_id(context->domain, isl_dim_set, pos, id);
  CHECK(context->domain);
  return context;
}

PypetContext* PypetContextIntersectDomain(PypetContext* pc, isl_set* domain) {
  CHECK(pc);
  pc = PypetContextCow(pc);
  pc->domain = isl_set_intersect(pc->domain, domain);
  CHECK(pc->domain);
  return pc;
}

PypetContext* PypetContextSetValue(PypetContext* context, isl_id* id,
                                   isl_pw_aff* pw_aff) {
  context = PypetContextCow(context);
  CHECK(context);
  context->assignments[id] = isl_pw_aff_copy(pw_aff);
  return context;
}

PypetContext* PypetContextClearValue(PypetContext* context, isl_id* id) {
  CHECK(context);
  context = PypetContextCow(context);
  CHECK(context);
  context->assignments.erase(id);
  CHECK(context);
  return context;
}

isl_pw_aff* PypetContextGetValue(PypetContext* context, isl_id* id) {
  CHECK(context);
  CHECK(id);
  isl_pw_aff* pa = isl_pw_aff_copy(context->assignments[id]);
  int dim = isl_pw_aff_dim(pa, isl_dim_in);
  if (dim == isl_set_dim(context->domain, isl_dim_set)) {
    return pa;
  }
  isl_multi_aff* ma = PypetPrefixProjection(PypetContextGetSpace(context), dim);
  pa = isl_pw_aff_pullback_multi_aff(pa, ma);
  return pa;
}

PypetContext* PypetContextAddInnerIterator(PypetContext* context, isl_id* id) {
  CHECK(context);
  CHECK(id);
  int pos = PypetContextGetDim(context);
  context = ExtendDomain(context, isl_id_copy(id));
  CHECK(context);
  isl_space* space = PypetContextGetSpace(context);
  isl_local_space* local_space = isl_local_space_from_space(space);
  isl_aff* aff = isl_aff_var_on_domain(local_space, isl_dim_set, pos);
  isl_pw_aff* pw_aff = isl_pw_aff_from_aff(aff);
  context = PypetContextSetValue(context, id, pw_aff);
  return context;
}

PypetContext* PypetContextClearWritesInTree(PypetContext* context,
                                            PypetTree* tree) {
  CHECK_GE(PypetTreeForeachAccessExpr(tree, &ClearWrite, &context), 0);
  return context;
}

bool PypetContextIsAssigned(PypetContext* context, isl_id* id) {
  return context->assignments.find(id) != context->assignments.end();
}

PypetContext* PypetContextSetAllowNested(PypetContext* context, bool val) {
  CHECK(context);
  if (context->allow_nested == val) {
    return context;
  }
  context = PypetContextCow(context);
  CHECK(context);
  context->allow_nested = val;
  return context;
}

PypetExpr* PypetContextEvaluateExpr(PypetContext* context, PypetExpr* expr) {
  expr = PypetExprInsertDomain(expr, PypetContextGetSpace(context));
  expr = PlugInAffineRead(expr, context);
  expr = PypetExprPlugInArgs(expr, context);
  expr = PlugInAffine(expr, context);
  expr = MergeConditionalAccesses(expr);
  expr = PlugInSummaries(expr, context);
  return expr;
}

PypetTree* PypetContextEvaluateTree(PypetContext* pc, PypetTree* tree) {
  return PypetTreeMapExpr(tree, TreeEvaluateExprWrapper, pc);
}

void ContextPrettyPrinter::Print(std::ostream& out,
                                 const PypetContext* context) {
  CHECK(context);
  isl_printer* p = isl_printer_to_str(isl_set_get_ctx(context->domain));
  CHECK(p);

  int indent = 0;
  p = isl_printer_set_indent(p, indent);
  p = isl_printer_set_yaml_style(p, ISL_YAML_STYLE_BLOCK);
  p = isl_printer_start_line(p);
  p = isl_printer_yaml_start_mapping(p);

  p = isl_printer_print_str(p, "domain");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_set(p, context->domain);
  p = isl_printer_yaml_next(p);

  p = isl_printer_print_str(p, "assignments");
  p = isl_printer_set_indent(p, indent + 2);
  p = isl_printer_yaml_next(p);

  if (!context->assignments.empty()) {
    for (auto it = context->assignments.begin();
         it != context->assignments.end(); ++it) {
      p = isl_printer_yaml_start_sequence(p);
      p = isl_printer_yaml_start_mapping(p);
      p = isl_printer_print_id(p, it->first);
      p = isl_printer_yaml_next(p);
      p = isl_printer_print_pw_aff(p, it->second);
      p = isl_printer_yaml_next(p);
      p = isl_printer_yaml_end_mapping(p);
      p = isl_printer_yaml_end_sequence(p);
    }
  } else {
    p = isl_printer_yaml_start_sequence(p);
    p = isl_printer_print_str(p, "empty");
    p = isl_printer_yaml_end_sequence(p);
  }
  p = isl_printer_yaml_next(p);

  p = isl_printer_set_indent(p, indent);
  p = isl_printer_print_str(p, "extracted_affine");
  p = isl_printer_set_indent(p, indent + 2);
  p = isl_printer_yaml_next(p);
  if (!context->extracted_affine.empty()) {
    for (auto it = context->extracted_affine.begin();
         it != context->extracted_affine.end(); it++) {
      p = isl_printer_yaml_start_sequence(p);
      p = isl_printer_print_pw_aff(p, it->second);
      p = isl_printer_yaml_end_sequence(p);
    }
  } else {
    p = isl_printer_yaml_start_sequence(p);
    p = isl_printer_print_str(p, "empty");
    p = isl_printer_yaml_end_sequence(p);
  }
  p = isl_printer_yaml_next(p);

  p = isl_printer_set_indent(p, indent);
  p = isl_printer_print_str(p, "nesting allowed");
  p = isl_printer_yaml_next(p);
  p = isl_printer_print_int(p, int(context->allow_nested));
  p = isl_printer_yaml_next(p);

  p = isl_printer_yaml_end_mapping(p);

  out << std::string(isl_printer_get_str(p));
  isl_printer_free(p);
}

}  // namespace pypet
}  // namespace pypoly
