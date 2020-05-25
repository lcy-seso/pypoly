#ifndef PYPOLY_CORE_PYPET_UTIL_H_
#define PYPOLY_CORE_PYPET_UTIL_H_

#include "c10/util/Logging.h"

#include <isl/aff.h>
#include <isl/arg.h>
#include <isl/ctx.h>
#include <isl/id.h>
#include <isl/map.h>
#include <isl/schedule.h>
#include <isl/set.h>
#include <isl/space.h>
#include <isl/stream.h>
#include <isl/union_map.h>
#include <isl/union_set.h>
#include <isl/val.h>
#include <string.h>

#include <memory>
#include <vector>

namespace pypoly {
namespace pypet {

#define UNIMPLEMENTED() LOG(FATAL) << "UNIMPLEMENTED"

}  // namespace pypet
}  // namespace pypoly

#endif  // PYPOLY_CORE_PYPET_UTIL_H_
