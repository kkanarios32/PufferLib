/* Pure C demo file for Squared. Build it with:
 * bash scripts/build_ocean.sh target local (debug)
 * bash scripts/build_ocean.sh target fast
 * We suggest building and debugging your env in pure C first. You
 * get faster builds and better error messages. To keep this example
 * simple, it does not include C neural nets. See Target for that.
 */

#include "repgrid.h"

int main() {
  RepGrid env = {.size = 5};
  allocate_repgrid(&env);
  init_repgrid(&env);
  c_reset(&env);
  c_render(&env);
  while (!WindowShouldClose()) {
    if (IsKeyDown(KEY_LEFT_SHIFT)) {
      env.actions[0] = 4;
      if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
        env.actions[0] = UP;
      if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
        env.actions[0] = DOWN;
      if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
        env.actions[0] = LEFT;
      if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
        env.actions[0] = RIGHT;
    } else {
      env.actions[0] = rand() % 5;
    }
    c_step(&env);
    c_render(&env);
  }
  free_allocated_repgrid(&env);
  c_close(&env);
}
