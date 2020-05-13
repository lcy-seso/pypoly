#include "gtest/gtest.h"
#include "options.h"
#include "pet.h"
#include "scop.h"
#include "scop_yaml.h"

#include <isl/arg.h>
#include <isl/ctx.h>
#include <isl/options.h>

#include <iostream>
constexpr const char* filename = "c_examples/stacked_lstm.c";

struct options {
  struct isl_options* isl;
  struct pet_options* pet;
};

ISL_ARGS_START(struct options, options_args)
ISL_ARG_CHILD(struct options, isl, "isl", &isl_options_args, "isl options")
ISL_ARG_CHILD(struct options, pet, NULL, &pet_options_args, "pet options")
ISL_ARGS_END

ISL_ARG_DEF(options, struct options, options_args)

int main(int argc, char* argv[]) {
  struct options* options = options_new_with_defaults();

  isl_ctx* ctx = isl_ctx_alloc_with_options(isl_options, options);

  struct pet_scop* scop = pet_scop_extract_from_C_source(ctx, filename, NULL);

  if (scop) pet_scop_emit(stdout, scop);

  pet_scop_free(scop);
  isl_ctx_free(ctx);
}
