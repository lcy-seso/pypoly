
#ifndef PYPET_EXPR_H
#define PYPET_EXPR_H

namespace pypet {

enum PypetExprType {
  PET_EXPR_ERROR = -1,
  PET_EXPR_ACCESS,
  PET_EXPR_CALL,
  PET_EXPR_OP,
};

enum PypetOpType {
  PET_ASSIGN = 0,
};

struct PypetExpr {
  PypetExpr(){};
  ~PypetExpr(){};

 private:
  int ref_;
  isl_ctx* ctx_;

  enum PypetExprType type_;

  int type_size_;

  std::vector<std::unique_ptr<PetExpr>> args_;
};
}  // namespace pypet

#endif
