#include "raylib.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NUM_ANIMALS 3
#define MAX_FEATURE_DIM 49
#define SAMPLING_MNIST 0
#define SAMPLING_SPIRAL 1
#define SAMPLING_GAUSSIAN 2

// clang-format off

const char one_goal[] = {
    'G', 'C', 'O', 'T', 'O', 'C', 'O', 'D', 'O', 'C', 'O', 'T', 'O', 'D', 'O',
    'O', 'O', 'D', 'O', 'C', 'O', 'T', 'O', 'C', 'O', 'D', 'O', 'C', 'O', 'T',
    'C', 'T', 'O', 'C', 'O', 'D', 'O', 'C', 'O', 'T', 'O', 'C', 'O', 'D', 'O',
    'D', 'C', 'C', 'O', 'T', 'O', 'C', 'O', 'T', 'O', 'C', 'O', 'D', 'O', 'C',
    'D', 'O', 'C', 'O', 'O', 'O', 'T', 'C', 'O', 'D', 'O', 'T', 'O', 'C', 'O',
    'T', 'D', 'D', 'O', 'O', 'O', 'D', 'O', 'C', 'O', 'T', 'O', 'D', 'O', 'C',
    'O', 'C', 'O', 'D', 'O', 'C', 'O', 'D', 'T', 'O', 'C', 'O', 'T', 'O', 'D',
    'C', 'O', 'D', 'O', 'T', 'O', 'C', 'K', 'O', 'D', 'O', 'C', 'O', 'T', 'O',
    'O', 'D', 'O', 'C', 'O', 'T', 'O', 'C', 'D', 'O', 'T', 'O', 'C', 'O', 'D',
    'C', 'O', 'T', 'O', 'D', 'O', 'C', 'O', 'T', 'O', 'C', 'O', 'D', 'O', 'C',
    'O', 'D', 'O', 'C', 'O', 'D', 'O', 'T', 'O', 'C', 'O', 'T', 'O', 'C', 'O',
    'T', 'O', 'C', 'O', 'T', 'O', 'D', 'O', 'C', 'O', 'D', 'O', 'T', 'O', 'C',
    'O', 'C', 'O', 'D', 'O', 'C', 'O', 'T', 'O', 'D', 'O', 'C', 'O', 'T', 'O',
    'D', 'O', 'T', 'O', 'C', 'O', 'D', 'O', 'C', 'O', 'T', 'O', 'C', 'O', 'D',
    'O', 'C', 'O', 'T', 'O', 'D', 'O', 'C', 'O', 'T', 'O', 'C', 'O', 'D', 'A',
};

// clang-format on

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
const unsigned char KEY = 7;

// Required struct. Only use floats!
typedef struct {
  float perf;  // Recommended 0-1 normalized single real number perf metric
  float score; // Recommended unnormalized single real number perf metric
  float episode_return; // Recommended metric: sum of agent rewards over episode
  float episode_length; // Recommended metric: number of steps of agent episode
  float dogs;   // Count of dogs in current observation
  float cats;   // Count of cats in current observation 
  float tigers; // Count of tigers in current observation
  float keys;
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
  float *observations;       // size x size x (feature_dim + 1)
  unsigned char *obs_assets; // size x size
  float *map;                // map_rows x map_cols x (feature_dim + 1)
  unsigned char *asset_map;  // map_rows x map_cols
  unsigned char *base_asset_map; // map_rows x map_cols
  Spec *spec;
  int *actions;
  float *rewards;
  unsigned char *terminals; // Required. We don't yet have truncations as standard yet
  uint8_t size;
  int tick;
  uint8_t r;
  uint8_t c;
  int start_pos;
  int random_sampling;
  int key_door;
  int has_key;
  int pred_dogs;
  int pred_cats;
  int pred_tigers;
  int pred_keys;
  int sampling_mode;
  int feature_dim;
  float *mnist_0;
  float *mnist_1;
  float *mnist_2;
  float *mnist_3;
  float *mnist_4;
  int mnist_0_count;
  int mnist_1_count;
  int mnist_2_count;
  int mnist_3_count;
  int mnist_4_count;
} RepGrid;

Spec initial_layout = {
    .layout = one_goal,
    .columns = 15,
    .rows = 15,
};

void add_log(RepGrid *env) {
  env->log.perf += (env->rewards[0] > 0) ? 1.0 : 0.0;
  env->log.score += env->rewards[0];
  env->log.episode_return += env->rewards[0];
  env->log.episode_length += env->tick;
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

void sample_spiral(int class_id, float *features) {
  float t = ((float)rand() / RAND_MAX) * 4.0 * M_PI;
  float offset = class_id * 2.0 * M_PI / 5.0;
  float radius = 0.5 + 0.25 * t;

  features[0] = radius * cos(t + offset);
  features[1] = radius * sin(t + offset);
  features[2] = 0.3 * t;

  float noise_scale = 0.15;
  features[0] += noise_scale * gaussian_noise();
  features[1] += noise_scale * gaussian_noise();
  features[2] += noise_scale * gaussian_noise();
}

void sample_gaussian(int class_id, float *features) {
  const float MEANS[5][3] = {
    {1.0, 2.0, 3.0},
    {2.0, 3.0, 1.0},
    {3.0, 1.0, 2.0},
    {2.0, 2.0, 2.0},
    {0.0, 3.0, 0.0},
  };

  for (int i = 0; i < 3; i++) {
    features[i] = MEANS[class_id][i] + gaussian_noise();
  }
}

void sample_features(RepGrid *env, int class_id, float *features) {
  if (env->sampling_mode == SAMPLING_MNIST) {
    float *buffer;
    int count;
    if (class_id == 0) {
      buffer = env->mnist_0;
      count = env->mnist_0_count;
    } else if (class_id == 1) {
      buffer = env->mnist_1;
      count = env->mnist_1_count;
    } else if (class_id == 2) {
      buffer = env->mnist_2;
      count = env->mnist_2_count;
    } else if (class_id == 4) {
      buffer = env->mnist_4;
      count = env->mnist_4_count;
    } else {
      buffer = env->mnist_3;
      count = env->mnist_3_count;
    }

    if (buffer != NULL && count > 0) {
      int idx = rand() % count;
      for (int i = 0; i < env->feature_dim; i++) {
        features[i] = buffer[idx * env->feature_dim + i];
      }
    }
  } else if (env->sampling_mode == SAMPLING_SPIRAL) {
    sample_spiral(class_id, features);
  } else {
    sample_gaussian(class_id, features);
  }
}

void allocate_repgrid(RepGrid *env) {
  env->observations = (float *)calloc(
      env->size * env->size * (env->feature_dim + 1), sizeof(float));
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

void alloc_assets(RepGrid *env) {
  env->spec = &initial_layout;
  env->map = (float *)calloc(env->spec->rows * env->spec->columns *
                                 (env->feature_dim + 1),
                             sizeof(float));
  env->asset_map = (unsigned char *)calloc(env->spec->rows * env->spec->columns,
                                           sizeof(unsigned char));
  env->base_asset_map = (unsigned char *)calloc(env->spec->rows * env->spec->columns,
                                           sizeof(unsigned char));
  env->obs_assets =
      (unsigned char *)calloc(env->size * env->size, sizeof(unsigned char));
}

void init_repgrid(RepGrid *env) {
  alloc_assets(env);
  env->tick = 0;

  float features[MAX_FEATURE_DIM];

  for (int r = 0; r < env->spec->rows; r++) {
    for (int c = 0; c < env->spec->columns; c++) {
      int idx = env->spec->columns * r + c;
      char curr = env->spec->layout[idx];
      if (curr == 'O') {
        env->map[idx] = EMPTY;
        env->asset_map[idx] = EMPTY;
        env->base_asset_map[idx] = EMPTY;
        sample_features(env, 3, features);
        int tile = env->spec->rows * env->spec->columns;
        for (int j = 0; j < env->feature_dim; j++) {
          env->map[(j + 1) * tile + idx] = features[j];
        }
      } else if (curr == 'G') {
        env->map[idx] = GOAL;
        env->asset_map[idx] = GOAL;
        env->base_asset_map[idx] = GOAL;
        sample_features(env, 3, features);
        int tile = env->spec->rows * env->spec->columns;
        for (int j = 0; j < env->feature_dim; j++) {
          env->map[(j + 1) * tile + idx] = features[j];
        }
      } else if (curr == 'A') {
        env->start_pos = idx;
        sample_features(env, 3, features);
        int tile = env->spec->rows * env->spec->columns;
        for (int j = 0; j < env->feature_dim; j++) {
          env->map[(j + 1) * tile + idx] = features[j];
        }
      } else {
        int class_id = -1;
        if (curr == 'D') {
          env->asset_map[idx] = DOG;
          env->base_asset_map[idx] = DOG;
          class_id = 0;
        } else if (curr == 'C') {
          env->asset_map[idx] = CAT;
          env->base_asset_map[idx] = CAT;
          class_id = 1;
        } else if (curr == 'T') {
          env->asset_map[idx] = TIGER;
          env->base_asset_map[idx] = TIGER;
          class_id = 2;
        } else if (curr == 'K') {
          env->asset_map[idx] = KEY;
          env->base_asset_map[idx] = KEY;
          class_id = 4;
        }
        if (class_id >= 0) {
          sample_features(env, class_id, features);
          int tile = env->spec->rows * env->spec->columns;
          for (int j = 0; j < env->feature_dim; j++) {
            env->map[(j + 1) * tile + idx] = features[j];
          }
        }
      }
    }
  }
}

void resample(RepGrid *env) {
  float features[MAX_FEATURE_DIM];

  for (int r = 0; r < env->spec->rows; r++) {
    for (int c = 0; c < env->spec->columns; c++) {
      int idx = env->spec->columns * r + c;
      char curr = env->spec->layout[idx];
      int class_id;
      if (curr == 'D') {
        env->asset_map[idx] = DOG;
        env->base_asset_map[idx] = DOG;
        class_id = 0;
      } else if (curr == 'C') {
        env->asset_map[idx] = CAT;
        env->base_asset_map[idx] = CAT;
        class_id = 1;
      } else if (curr == 'T') {
        env->asset_map[idx] = TIGER;
        env->base_asset_map[idx] = TIGER;
        class_id = 2;
      } else if (curr == 'K') {
        env->asset_map[idx] = KEY;
        env->base_asset_map[idx] = KEY;
        class_id = 4;
      } else {
        class_id = 3;
      }
      sample_features(env, class_id, features);
      int tile = env->spec->rows * env->spec->columns;
      for (int j = 0; j < env->feature_dim; j++) {
        env->map[(j + 1) * tile + idx] = features[j];
      }
    }
  }
}

void set_feature_obs(RepGrid *env) {
  int half_obs = env->size / 2;
  int obs_tile = env->size * env->size;
  int map_tile = env->spec->rows * env->spec->columns;
  memset(env->observations, 0,
         obs_tile * (env->feature_dim + 1) * sizeof(float));
  memset(env->obs_assets, 0, obs_tile * sizeof(unsigned char));

  for (int dy = -half_obs; dy <= half_obs; dy++) {
    for (int dx = -half_obs; dx <= half_obs; dx++) {
      int y = env->r + dy;
      int x = env->c + dx;
      if (x >= 0 && x < env->spec->columns && y >= 0 && y < env->spec->rows) {
        for (int i = 0; i < env->feature_dim + 1; i++) {
          env->observations[i * obs_tile + (dy + half_obs) * env->size +
                            (dx + half_obs)] =
              env->map[i * map_tile + y * env->spec->columns + x];
        }
        env->obs_assets[(dy + half_obs) * env->size + (dx + half_obs)] =
            env->asset_map[y * env->spec->columns + x];
      } else {
        env->observations[(dy + half_obs) * env->size + (dx + half_obs)] = WALL;
        env->obs_assets[(dy + half_obs) * env->size + (dx + half_obs)] = WALL;
      }
    }
  }
  env->observations[obs_tile / 2] = AGENT;
  env->obs_assets[obs_tile / 2] = AGENT;
}

void count_animals(RepGrid *env) {
  env->log.dogs = 0;
  env->log.cats = 0;
  env->log.tigers = 0;
  env->log.keys = 0;

  int obs_tile = env->size * env->size;
  for (int i = 0; i < obs_tile; i++) {
    unsigned char asset = env->obs_assets[i];
    if (asset == DOG) env->log.dogs++;
    else if (asset == CAT) env->log.cats++;
    else if (asset == TIGER) env->log.tigers++;
    else if (asset == KEY) env->log.keys++;
  }
}


void c_reset(RepGrid *env) {
    int start_pos = env->start_pos;
    env->r = start_pos / env->spec->columns;
    env->c = start_pos % env->spec->columns;

    env->tick = 0;
    env->has_key = 0;

    for (int r = 0; r < env->spec->rows; r++) {
      for (int c = 0; c < env->spec->columns; c++) {
        int idx = env->spec->columns * r + c;
        char curr = env->spec->layout[idx];
        if (curr == 'D') {
          env->asset_map[idx] = DOG;
          env->base_asset_map[idx] = DOG;
        } else if (curr == 'K') {
          env->asset_map[idx] = KEY;
          env->base_asset_map[idx] = KEY;
        }
      }
    }

    set_feature_obs(env);
    count_animals(env);
}

void c_step(RepGrid *env) {
  env->tick += 1;

  int action = env->actions[0];
  env->terminals[0] = 0;
  int oldPos = env->r * env->spec->columns + env->c;
  env->asset_map[oldPos] = EMPTY;

  if (action == DOWN) {
    env->r += 1;
  } else if (action == RIGHT) {
    env->c += 1;
  } else if (action == UP) {
    env->r -= 1;
  } else if (action == LEFT) {
    env->c -= 1;
  }

  if (env->random_sampling) {
    env->r = rand() % env->spec->rows;
    env->c = rand() % env->spec->columns;
    resample(env);
  }

  if (env->r < 0 || env->c < 0 || env->r >= env->spec->rows ||
      env->c >= env->spec->columns) {
    env->terminals[0] = 1;
    env->rewards[0] = -1.0;
    add_log(env);
    c_reset(env);
    return;
  }

  int mapPos = env->r * env->spec->columns + env->c;

  if (env->asset_map[mapPos] == TIGER) {
    env->terminals[0] = 1;
    env->rewards[0] = -1.0;
    add_log(env);
    c_reset(env);
    return;
  } else if (env->asset_map[mapPos] == GOAL) {
    if (!env->key_door || env->has_key) {
      env->terminals[0] = 1;
      env->rewards[0] = 1.0;
      add_log(env);
      c_reset(env);
      return;
    }
    env->rewards[0] = -0.01;
    env->log.episode_return += -0.01;
    env->log.score += -0.01;
  } else if (env->asset_map[mapPos] == KEY) {
    if (env->key_door) {
      env->has_key = 1;
    }
    env->asset_map[mapPos] = EMPTY;
    env->base_asset_map[mapPos] = EMPTY;
    env->rewards[0] = -0.01;
    env->log.episode_return += -0.01;
    env->log.score += -0.01;
  } else if (env->asset_map[mapPos] == DOG) {
    env->rewards[0] = 0.1;
    env->log.episode_return += 0.1;
    env->log.score += 0.1;
    env->asset_map[mapPos] = EMPTY;
    env->base_asset_map[mapPos] = EMPTY;
  } else {
    env->rewards[0] = -0.01;
    env->log.episode_return += -0.01;
    env->log.score += -0.01;
  }

  env->asset_map[oldPos] = env->base_asset_map[oldPos];
  env->asset_map[mapPos] = AGENT;
  set_feature_obs(env);
  count_animals(env);
}

void c_render(RepGrid *env) {
  if (!IsWindowReady()) {
    InitWindow(64 * env->size, 64 * env->size, "PufferLib Squared");
    SetTargetFPS(3);
  }

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
        color = (Color){179, 179, 179, 255};
      } else if (tex == AGENT) {
        color = (Color){173, 216, 230, 255};
      } else if (tex == DOG) {
        color = (Color){144, 238, 144, 255};
      } else if (tex == CAT) {
        color = (Color){255, 182, 193, 255};
      } else if (tex == TIGER) {
        color = (Color){255, 218, 185, 255};
      } else if (tex == KEY) {
        color = (Color){200, 100, 255, 255};
      } else if (tex == GOAL) {
        color = (Color){255, 255, 0, 255};
      } else {
        color = (Color){0, 0, 0, 255};
      }
      DrawRectangle(j * px, i * px, px, px, color);
    }
  }

  char buf[64];
  snprintf(buf, sizeof(buf), "Dogs: %d", (int)env->log.dogs);
  DrawText(buf, 8, 8, 20, (Color){255, 255, 255, 255});
  snprintf(buf, sizeof(buf), "Cats: %d", (int)env->log.cats);
  DrawText(buf, 8, 32, 20, (Color){255, 255, 255, 255});
  snprintf(buf, sizeof(buf), "Tigers: %d", (int)env->log.tigers);
  DrawText(buf, 8, 56, 20, (Color){255, 255, 255, 255});
  snprintf(buf, sizeof(buf), "Keys: %d", (int)env->log.keys);
  DrawText(buf, 8, 80, 20, (Color){255, 255, 255, 255});

  snprintf(buf, sizeof(buf), "Probe Dogs: %d", (int)env->pred_dogs);
  DrawText(buf, 8, 112, 20, (Color){255, 255, 255, 255});
  snprintf(buf, sizeof(buf), "Probe Cats: %d", (int)env->pred_cats);
  DrawText(buf, 8, 136, 20, (Color){255, 255, 255, 255});
  snprintf(buf, sizeof(buf), "Probe Tigers: %d", (int)env->pred_tigers);
  DrawText(buf, 8, 160, 20, (Color){255, 255, 255, 255});
  snprintf(buf, sizeof(buf), "Probe Keys: %d", (int)env->pred_keys);
  DrawText(buf, 8, 184, 20, (Color){255, 255, 255, 255});
  snprintf(buf, sizeof(buf), "Reward: %f", env->rewards[0]);
  DrawText(buf, 8, 208, 20, (Color){255, 255, 255, 255});
  EndDrawing();
}

void c_close(RepGrid *env) {
  if (IsWindowReady()) {
    CloseWindow();
  }
  free(env->asset_map);
  free(env->map);
  free(env->obs_assets);
}
