#ifndef PYPOLY_CORE_PYPET_ISL_PRINTER_H_
#define PYPOLY_CORE_PYPET_ISL_PRINTER_H_

#include "util.h"

namespace pypoly {
namespace pypet {

#define PRINTER_FUNC(isl_type)                                          \
  static inline std::ostream& operator<<(std::ostream& out,             \
                                         isl_##isl_type* val) {         \
    isl_printer* p = isl_printer_to_str(isl_##isl_type##_get_ctx(val)); \
    p = isl_printer_print_##isl_type(p, val);                           \
    char* str = isl_printer_get_str(p);                                 \
    isl_printer_free(p);                                                \
    out << std::string(str);                                            \
    free(str);                                                          \
    return out;                                                         \
  }
PRINTER_FUNC(multi_pw_aff)
PRINTER_FUNC(multi_aff)
PRINTER_FUNC(pw_aff)
PRINTER_FUNC(aff)
PRINTER_FUNC(id)
PRINTER_FUNC(schedule)

#define PRINTER_FUNC2(isl_type)                                 \
  static inline std::ostream& operator<<(std::ostream& out,     \
                                         isl_##isl_type* val) { \
    isl_ctx* ctx = isl_ctx_alloc();                             \
    isl_printer* p = isl_printer_to_str(ctx);                   \
    p = isl_printer_print_##isl_type(p, val);                   \
    char* str = isl_printer_get_str(p);                         \
    isl_printer_free(p);                                        \
    out << std::string(str);                                    \
    free(str);                                                  \
    isl_ctx_free(ctx);                                          \
    return out;                                                 \
  }
PRINTER_FUNC2(set)
PRINTER_FUNC2(map)
PRINTER_FUNC2(union_set)
PRINTER_FUNC2(union_map)
PRINTER_FUNC2(space)

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_CORE_PYPET_ISL_PRINTER_H_
