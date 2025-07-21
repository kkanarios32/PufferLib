/* Pure C demo file for Squared. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets. See Target for that.
 */

#include "repgrid.h"

int main() {
  RepGrid env = {.size = 11, .n_classes = 2, .feature_dim = 1};
  allocate_repgrid(&env);
  init_repgrid(&env);
  c_reset(&env);
  c_render(&env);
  while (!WindowShouldClose()) {
    c_render(&env);
    env.actions[0] = 0;
    // c_step(&env);
  }
  free_allocated_repgrid(&env);
  c_close(&env);
}
