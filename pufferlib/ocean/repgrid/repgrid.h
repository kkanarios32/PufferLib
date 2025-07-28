/* Squared: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include "raylib.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// clang-format off

const char one_goal[] = {
    'G', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
    'O', 'O', 'O', 'O', 'O', 'C', 'C', 'C', 'C',
    'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C', 'C',
    'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
    'O', 'O', 'O', 'D', 'A', 'T', 'O', 'O', 'O',
    'O', 'O', 'O', 'O', 'C', 'O', 'O', 'O', 'O',
    'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
    'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
    'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O',
};

const int NUM_ANIMALS = 3;
const int ANIMAL_FEATURES = 3;

float ANIMALS[NUM_ANIMALS][ANIMAL_FEATURES] = {
  {1.0, 2.0, 3.0},
  {3.0, 2.0, 1.0},
  {3.0, 1.0, 2.0}
};

const unsigned char DOWN = 0;
const unsigned char UP = 1;
const unsigned char LEFT = 2;
const unsigned char RIGHT = 3;
const unsigned char NOOP = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char GOAL = 2;
const unsigned char DOG = 3;
const unsigned char CAT = 4;
const unsigned char TIGER = 5;
const unsigned char WALL = 6;

// Required struct. Only use floats!
typedef struct {
  float perf;  // Recommended 0-1 normalized single real number perf metric
  float score; // Recommended unnormalized single real number perf metric
  float episode_return; // Recommended metric: sum of agent rewards over episode
  float episode_length; // Recommended metric: number of steps of agent episode
  // Any extra fields you add here may be exported to Python in binding.c
  float n; // Required as the last field
} Log;

typedef struct Map {
  const char *layout;
  const uint8_t columns;
  const uint8_t rows;
} Spec;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
  Log log; // Required field. Env binding code uses this to aggregate logs
  float *observations; // size x size x (feature_dim + 1)
  unsigned char *obs_assets;    // size x size
  float *map; // map_rows x map_cols x (feature_dim + 1)
  unsigned char *asset_map;    // map_rows x map_cols
  Spec *spec;
  int *actions;                // up, down, left, right
  float *rewards;              // +1 for feature rn...
  unsigned char
      *terminals; // Required. We don't yet have truncations as standard yet
  uint8_t size;
  int tick;
  uint8_t r;
  uint8_t c;
  int start_pos;
} RepGrid;

Spec initial_layout = {
    .layout = one_goal,
    .columns = 9,
    .rows = 9,
};

void add_log(RepGrid *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

float gaussian_noise() {
  static int has_spare = 0;
  static float spare;

  if (has_spare) {
    has_spare = 0;
    return spare;
  }

  float u, v, s;
  do {
    u = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
    v = ((float)rand() / RAND_MAX) * 2.0 - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s == 0.0);

  s = sqrt(-2.0 * log(s) / s);
  spare = v * s;
  has_spare = 1;
  return u * s;
}

void allocate_repgrid(RepGrid *env) {
  env->observations = (float *)calloc(
      env->size * env->size * (ANIMAL_FEATURES + 1), sizeof(float));
  env->actions = (int *)calloc(1, sizeof(int));
  env->rewards = (float *)calloc(1, sizeof(float));
  env->terminals = (unsigned char *)calloc(1, sizeof(unsigned char));
  env->obs_assets =
      (unsigned char *)calloc(env->size * env->size, sizeof(unsigned char));
}

void free_allocated_repgrid(RepGrid *env) {
  free(env->rewards);
  free(env->observations);
  free(env->obs_assets);
  free(env->actions);
  free(env->terminals);
}

void alloc_assets(RepGrid *env) {
  env->spec = &initial_layout;
  env->map = (float *)calloc(
      env->spec->rows * env->spec->columns * (ANIMAL_FEATURES + 1), sizeof(float));
  env->asset_map =
      (unsigned char *)calloc(env->spec->rows * env->spec->columns, sizeof(unsigned char));
}

void init_repgrid(RepGrid *env) {
  alloc_assets(env);
  env->tick = 0;

  for (int r = 0; r < env->spec->rows; r++) {
    for (int c = 0; c < env->spec->columns; c++) {
      int idx = env->spec->columns * r + c;
      // track true animals for rendering
      char curr = env->spec->layout[idx];
      if (curr == 'O') { 
        env->map[idx] = EMPTY;
        env->asset_map[idx] = EMPTY;
      } else if (curr == 'G') { 
        env->map[idx] = GOAL;
        env->asset_map[idx] = GOAL;
      } else if (curr == 'A') { 
        env->start_pos = idx;
      } else {
        unsigned char ANIMAL_TYPE;
        if (curr == 'D') { 
          env->asset_map[idx] = DOG;
          ANIMAL_TYPE = DOG;
        } else if (curr == 'C') { 
          env->asset_map[idx] = CAT;
          ANIMAL_TYPE = CAT;
        } else if (curr == 'T') { 
          env->asset_map[idx] = TIGER;
          ANIMAL_TYPE = TIGER;
        }
        // fill out the features in the obs space
        int tile = env->spec->rows * env->spec->columns;
        for (int j = 0; j < ANIMAL_FEATURES; j++) {
          env->map[(j + 1) * tile + idx] = ANIMALS[(ANIMAL_TYPE - 3)][j] + gaussian_noise();
        }
      }
    }
  }
}

void set_feature_obs(RepGrid *env) {
  int half_obs = env->size / 2;
  int obs_tile = env->size * env->size;
  int map_tile = env->spec->rows * env->spec->columns;
  memset(env->observations, 0, obs_tile * (ANIMAL_FEATURES + 1) * sizeof(float));
  memset(env->obs_assets, 0, obs_tile * sizeof(unsigned char));

  for (int dy = -half_obs; dy <= half_obs; dy++) {
      for (int dx = -half_obs; dx <= half_obs; dx++) {
          int y = env->r + dy;
          int x = env->c + dx;
          if (x >= 0 && x < env->spec->columns && y >= 0 && y < env->spec->rows) {
            for (int i = 0; i < ANIMAL_FEATURES + 1; i++) {
              env->observations[i*obs_tile + (dy + half_obs)*env->size + (dx + half_obs)] = env->map[i * map_tile + y*env->spec->columns + x];
            }
            env->obs_assets[(dy + half_obs)*env->size + (dx + half_obs)] = env->asset_map[y*env->spec->columns + x];
          } else {
            env->observations[(dy + half_obs)*env->size + (dx + half_obs)] = WALL;
            env->obs_assets[(dy + half_obs)*env->size + (dx + half_obs)] = WALL;
          }
      }
  }
  env->observations[obs_tile / 2] = AGENT;
  env->obs_assets[obs_tile / 2] = AGENT;
}

void c_reset(RepGrid *env) {
  int start_pos = env->start_pos;
  env->r = start_pos / env->spec->rows;
  env->c = start_pos % env->spec->columns;

  env->tick = 0;
  set_feature_obs(env);
}

// Required function
void c_step(RepGrid *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  env->observations[env->r * env->size + env->c] = EMPTY;
  env->asset_map[env->r * env->size + env->c] = EMPTY;

  if (action == DOWN) {
    env->r += 1;
  } else if (action == RIGHT) {
    env->c += 1;
  } else if (action == UP) {
    env->r -= 1;
  } else if (action == LEFT) {
    env->c -= 1;
  }

  if (env->r < 0 || env->c < 0 || env->r >= env->spec->rows || env->c >= env->spec->columns) {
    env->terminals[0] = 1;
    env->rewards[0] = -1.0;
    add_log(env);
    c_reset(env);
    return;
  }

  int pos = env->r * env->size + env->c;
  if (env->asset_map[pos] == TIGER) {
    env->terminals[0] = 1;
    env->rewards[0] = -0.1;
    add_log(env);
    c_reset(env);
    return;
  } else if (env->asset_map[pos] == GOAL) {
    env->terminals[0] = 1;
    env->rewards[0] = 1.0;
    add_log(env);
    c_reset(env);
    return;
  } else if (env->asset_map[pos] == DOG) {
    env->rewards[0] = 0.1;
  } else {
    env->rewards[0] = 0.0;
  }

  add_log(env);

  env->observations[pos] = AGENT;
  env->asset_map[pos] = AGENT;
  set_feature_obs(env);
}

// Required function. Should handle creating the client on first call
void c_render(RepGrid *env) {
  if (!IsWindowReady()) {
    InitWindow(64 * env->size, 64 * env->size, "PufferLib Squared");
    SetTargetFPS(5);
  }

  // Standard across our envs so exiting is always the same
  if (IsKeyDown(KEY_ESCAPE)) {
    exit(0);
  }

  BeginDrawing();
  ClearBackground((Color){6, 24, 24, 255});

  int px = 64;
  for (int i = 0; i < env->size; i++) {
    for (int j = 0; j < env->size; j++) {
      int tex = env->obs_assets[i * env->size + j];
      Color color;
      if (tex == WALL) {
        color = (Color){179, 179, 179, 255};  // Pastel Gray
      } else if (tex == AGENT) {
        color = (Color){173, 216, 230, 255};  // Pastel Blue (Light Blue)
      } else if (tex == DOG) {
        color = (Color){144, 238, 144, 255};  // Pastel Green (Light Green)
      } else if (tex == CAT) {
        color = (Color){255, 182, 193, 255};  // Pastel Red (Light Pink)
      } else if (tex == TIGER) {
        color = (Color){255, 218, 185, 255};  // Pastel Orange (Peach Puff)
      } else {
        color = (Color){0, 0, 0, 255};        // Black (Unknown)
      }
      DrawRectangle(j * px, i * px, px, px, color);
    }
  }
  EndDrawing();
}

// Required function. Should clean up anything you allocated
// Do not free env->observations, actions, rewards, terminals
void c_close(RepGrid *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
  free(env->asset_map);
  free(env->map);
}
