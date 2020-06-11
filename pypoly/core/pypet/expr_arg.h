#ifndef PYPOLY_CORE_PYPET_EXPR_ARG_H_
#define PYPOLY_CORE_PYPET_EXPR_ARG_H_

#include "pypoly/core/pypet/expr.h"

namespace pypoly {
namespace pypet {

PypetExpr* PypetExprAccessPlugInArgs(PypetExpr* expr, PypetContext* context);

PypetExpr* PypetExprPlugInArgs(PypetExpr* expr, PypetContext* context);

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_CORE_PYPET_EXPR_ARG_H_
