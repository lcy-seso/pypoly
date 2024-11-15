#ifndef PYPOLY_CORE_PYPET_ARRAY_H_
#define PYPOLY_CORE_PYPET_ARRAY_H_

#include "pypoly/core/pypet/array.pb.h"
#include "pypoly/core/pypet/pypet.h"

namespace pypoly {
namespace pypet {

struct PypetExpr;

typedef std::map<std::string, size_t> ContextVarTable;
typedef std::map<std::string, size_t>* ContextVarTablePtr;

/* Compare two sequences of identifiers based on their names. */
struct ArrayDescLess {
  bool operator()(isl_id_list* x, isl_id_list* y) {
    int x_n = isl_id_list_n_id(x);
    int y_n = isl_id_list_n_id(y);

    for (int i = 0; i < x_n && i < y_n; ++i) {
      isl_id* x_i = isl_id_list_get_id(x, i);
      isl_id* y_i = isl_id_list_get_id(y, i);
      const char* x_name = isl_id_get_name(x_i);
      const char* y_name = isl_id_get_name(y_i);
      int cmp = strcmp(x_name, y_name);
      isl_id_free(x_i);
      isl_id_free(y_i);
      if (cmp) return cmp < 0;
    }
    return x_n < y_n;
  }
};

struct ArrayDescSet : public std::set<isl_id_list*, ArrayDescLess> {
  explicit ArrayDescSet(const ContextVarTablePtr vars) : var_table(vars) {}
  ~ArrayDescSet() {
    for (auto it = begin(); it != end(); ++it) isl_id_list_free(*it);
  }

  void insert(__isl_take isl_id_list* list) {
    if (find(list) == end())
      set::insert(list);
    else
      isl_id_list_free(list);
  }

  void erase(__isl_keep isl_id_list* list) {
    iterator it;

    it = find(list);
    if (it == end()) return;

    isl_id_list_free(*it);
    set::erase(it);
  }

  static void ExprCollectArrays(__isl_keep PypetExpr* expr,
                                ArrayDescSet& arrays);
  static void AccessCollectArrays(__isl_keep PypetExpr* expr,
                                  ArrayDescSet& arrays);
  static void StmtCollectArrays(__isl_keep PypetStmt* stmt,
                                ArrayDescSet& arrays);
  static void PypetScopCollectArrays(__isl_keep PypetScop* scop,
                                     ArrayDescSet& arrays);

  const ContextVarTablePtr var_table;
};

class ArrayScanner {
 public:
  ArrayScanner(const ContextDesc* ctx_desc) : ctx_desc_(ctx_desc) {
    for (size_t i = 0; i < ctx_desc->vars_size(); ++i) {
      var_table_.insert(std::make_pair(ctx_desc->vars()[i].name(), i));
    }
  }
  ~ArrayScanner() = default;

  __isl_keep PypetScop* ScanArrays(isl_ctx* ctx, __isl_keep PypetScop* scop);

 private:
  const ContextDesc* ctx_desc_;
  ContextVarTable var_table_;

  // int GetVarDimFromId(isl_id* id);
  int GetVarDescPos(const std::string& name);
  int GetContextVarDim(const ContextVar& var);

  __isl_give PypetExpr* GetArraySize(__isl_keep isl_id* id);
  __isl_keep PypetArray* UpdateArraySize(__isl_keep PypetArray* array, int pos,
                                         __isl_take isl_pw_aff* size);
  __isl_keep PypetArray* SetArrayUpperBounds(__isl_keep PypetArray* array);
  __isl_give PypetArray* ExtractArray(isl_ctx* ctx,
                                      __isl_keep isl_id_list* decls);
  __isl_give PypetArray* ExtractArray(isl_ctx* ctx, __isl_keep isl_id* id);
};

struct PypetArray* PypetArrayFree(struct PypetArray* array);
int PypetArrayIsEqual(const struct PypetArray* array1,
                      const struct PypetArray* array2);

}  // namespace pypet
}  // namespace pypoly
#endif  // PYPOLY_CORE_PYPET_ARRAY_H_
