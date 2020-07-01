#ifndef PYPOLY_CORE_PYPET_TYPE_H_
#define PYPOLY_CORE_PYPET_TYPE_H_

namespace pypoly {
namespace pypet {

enum PypetExprType {
  PYPET_EXPR_ERROR = -1,
  PYPET_EXPR_ACCESS,
  PYPET_EXPR_CALL,
  PYPET_EXPR_OP,
  PYPET_EXPR_INT,
};

enum PypetOpType {
  PYPET_ASSIGN = 0,  // Tensor operation leads to an assignment.
  PYPET_ADD,
  PYPET_SUB,
  PYPET_MUL,
  PYPET_DIV,
  PYPET_MOD,
  PYPET_MINUS,
  PYPET_EQ,
  PYPET_NE,
  PYPET_LE,
  PYPET_GE,
  PYPET_LT,
  PYPET_GT,
  PYPET_COND,
  PYPET_AND,
  PYPET_XOR,
  PYPET_OR,
  PYPET_NOT,
  PYPET_APPLY,
  PYPET_LIST_LITERAL,
  PYPET_ATTRIBUTE,
  PYPET_UNKNOWN,
};

enum PypetExprAccessType {
  PYPET_EXPR_ACCESS_MAY_READ = 0,
  PYPET_EXPR_ACCESS_BEGIN = PYPET_EXPR_ACCESS_MAY_READ,
  PYPET_EXPR_ACCESS_FAKE_KILL = PYPET_EXPR_ACCESS_MAY_READ,
  PYPET_EXPR_ACCESS_MAY_WRITE,
  PYPET_EXPR_ACCESS_MUST_WRITE,
  PYPET_EXPR_ACCESS_END,
  PYPET_EXPR_ACCESS_KILL,
};

enum PypetArgType {
  PYPET_ARG_INT,
  PYPET_ARG_TENSOR,
  PYPET_ARG_ARRAY,
  PYPET_ARG_OTHER,  // int, float, etc. other numeric types.
  // TODO(Ying): Do we need more argument types?
};

enum PypetTreeType {
  PYPET_TREE_ERROR = 0,
  PYPET_TREE_EXPR,
  PYPET_TREE_BLOCK,
  PYPET_TREE_BREAK,
  PYPET_TREE_CONTINUE,
  PYPET_TREE_DECL,
  PYPET_TREE_DECL_INIT,
  PYPET_TREE_IF,      /* An if without an else branch */
  PYPET_TREE_IF_ELSE, /* An if with an else branch */
  PYPET_TREE_FOR,
  PYPET_TREE_RETURN,
};

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_CORE_PYPET_TYPE_H_
