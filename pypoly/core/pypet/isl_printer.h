#ifndef PYPOLY_CORE_PYPET_ISL_PRINTER_H_
#define PYPOLY_CORE_PYPET_ISL_PRINTER_H_

#include "util.h"

namespace pypoly {
namespace pypet {

#define PRINTER_FUNC(isl_type)                                              \
  inline std::ostream& operator<<(std::ostream& out, const isl_type* val) { \
    isl_printer* p = isl_printer_to_str(isl_##isl_type##_get_ctx(val));     \
    p = isl_printer_print_##isl_type(p, val);                               \
    char* str = isl_printer_get_str(p);                                     \
    isl_printer_free(p);                                                    \
    out << std::string(str);                                                \
    free(str);                                                              \
    return out;                                                             \
  }
PRINTER_FUNC(multi_pw_aff)
PRINTER_FUNC(pw_aff)
#undef

#define PRINTER_FUNC2(isl_type)                                             \
  inline std::ostream& operator<<(std::ostream& out, const isl_type* val) { \
    isl_options* options = isl_options_new_with_defaults();                 \
    isl_ctx* ctx = isl_ctx_alloc_with_options(&isl_options_args, options);  \
    isl_printer* p = isl_printer_to_str(ctx);                               \
    p = isl_printer_print_##isl_type(p, val);                               \
    char* str = isl_printer_get_str(p);                                     \
    isl_printer_free(p);                                                    \
    out << std::string(str);                                                \
    free(str);                                                              \
    isl_ctx_free(ctx);                                                      \
    return out;                                                             \
  }
PRINTER_FUNC2(set)
#undef

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_CORE_PYPET_ISL_PRINTER_H_
