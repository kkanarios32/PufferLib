/* Squared: a sample single-agent grid env.
 * Use this as a tutorial and template for your first env.
 * See the Target env for a slightly more complex example.
 * Star PufferLib on GitHub to support. It really, really helps!
 */

#include "raylib.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const unsigned char NOOP = 0;
const unsigned char DOWN = 1;
const unsigned char UP = 2;
const unsigned char LEFT = 3;
const unsigned char RIGHT = 4;

const unsigned char EMPTY = 0;
const unsigned char AGENT = 1;
const unsigned char DOG = 2;
const unsigned char CAT = 3;

// Required struct. Only use floats!
typedef struct {
  float perf;  // Recommended 0-1 normalized single real number perf metric
  float score; // Recommended unnormalized single real number perf metric
  float episode_return; // Recommended metric: sum of agent rewards over episode
  float episode_length; // Recommended metric: number of steps of agent episode
  // Any extra fields you add here may be exported to Python in binding.c
  float n; // Required as the last field
} Log;

// Required that you have some struct for your env
// Recommended that you name it the same as the env file
typedef struct {
  Log log; // Required field. Env binding code uses this to aggregate logs
  unsigned char *observations; // size x size x (feature_dim + 1)
  unsigned char *asset_map;    // size x size
  int *actions;                // up, down, left, right
  float *rewards;              // +1 for feature rn...
  unsigned char
      *terminals; // Required. We don't yet have truncations as standard yet
  int size;
  int feature_dim;
  int n_classes;
  float *animals; // n_classes x feature_dim
  int tick;
  int r;
  int c;
} RepGrid;

void add_log(RepGrid *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1 : 0;
  env->log.score += env->rewards[0];
  env->log.episode_length += env->tick;
  env->log.episode_return += env->rewards[0];
  env->log.n++;
}

double gaussian_noise() {
  static int has_spare = 0;
  static double spare;

  if (has_spare) {
    has_spare = 0;
    return spare;
  }

  double u, v, s;
  do {
    u = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    v = ((double)rand() / RAND_MAX) * 2.0 - 1.0;
    s = u * u + v * v;
  } while (s >= 1.0 || s == 0.0);

  s = sqrt(-2.0 * log(s) / s);
  spare = v * s;
  has_spare = 1;
  return u * s;
}

void allocate_repgrid(RepGrid *env) {
  env->observations = (unsigned char *)calloc(
      env->size * env->size * (env->feature_dim + 1), sizeof(unsigned char));
  env->actions = (int *)calloc(1, sizeof(int));
  env->rewards = (float *)calloc(1, sizeof(float));
  env->terminals = (unsigned char *)calloc(1, sizeof(unsigned char));
}

void free_allocated_repgrid(RepGrid *env) {
  free(env->rewards);
  free(env->observations);
  free(env->actions);
  free(env->terminals);
}

void init_repgrid(RepGrid *env) {
  env->asset_map =
      (unsigned char *)calloc(env->size * env->size, sizeof(unsigned char));
  env->animals =
      (float *)calloc(env->feature_dim * env->n_classes, sizeof(float));
  for (int i = 0; i < env->n_classes; i++) {
    for (int j = 0; j < env->feature_dim; j++) {
      env->animals[i * env->feature_dim + j] = i * 3;
    }
  }
}
void init_animals(float *animals) {}

// Required function
void c_reset(RepGrid *env) {
  int tiles = env->size * env->size;
  int ntiles = tiles * env->feature_dim;
  memset(env->observations, 0, ntiles * sizeof(unsigned char));
  memset(env->asset_map, 0, tiles * sizeof(unsigned char));

  env->observations[tiles / 2] = AGENT;
  env->asset_map[tiles / 2] = AGENT;
  env->r = env->size / 2;
  env->c = env->size / 2;
  env->tick = 0;
  int animal_idx;
  for (int n = 0; n < env->n_classes; n++) {
    int animal = n + 2;
    int n_animals = rand() % 4;
    for (int i = 0; i < n_animals; i++) {
      do {
        animal_idx = rand() % tiles;
      } while (env->asset_map[animal_idx] != EMPTY);
      // track true animals for rendering
      env->asset_map[animal_idx] = animal;
      // fill out the features in the obs space
      for (int j = 0; j < env->feature_dim; j++) {
        env->observations[tiles * env->feature_dim + animal_idx] =
            env->animals[n * env->feature_dim + j] + gaussian_noise();
      }
    }
  }
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

  if (env->tick > 3 * env->size || env->r < 0 || env->c < 0 ||
      env->r >= env->size || env->c >= env->size) {
    env->terminals[0] = 1;
    env->rewards[0] = -1.0;
    add_log(env);
    c_reset(env);
    return;
  }

  int pos = env->r * env->size + env->c;
  if (env->asset_map[pos] == CAT) {
    env->terminals[0] = 1;
    env->rewards[0] = -1.0;
    add_log(env);
    c_reset(env);
    return;
  }

  env->rewards[0] = 1.0;
  add_log(env);

  env->observations[pos] = AGENT;
  env->asset_map[pos] = AGENT;
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
      int tex = env->asset_map[i * env->size + j];
      Color color;
      if (tex == EMPTY) {
        continue;
      } else if (tex == AGENT) {
        color = (Color){187, 0, 0, 255};
      } else if (tex == DOG) {
        color = (Color){0, 187, 187, 255};
      } else {
        color = (Color){187, 187, 0, 255};
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
  free(env->animals);
}
