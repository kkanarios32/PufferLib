#include "repgrid.h"

#define Env RepGrid
#include "../env_binding.h"

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->size = unpack(kwargs, "size");
  env->random_sampling = unpack(kwargs, "random_sampling");
  init_repgrid(env);
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "perf", log->perf);
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
  assign_to_dict(dict, "dogs", log->dogs);
  assign_to_dict(dict, "cats", log->cats);
  assign_to_dict(dict, "tigers", log->tigers);
  return 0;
}

static PyObject* vec_get_counts(PyObject* self, PyObject* args) {
    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }
    
    PyObject* list = PyList_New(vec->num_envs);
    if (list == NULL) {
        return NULL;
    }
    
    for (int i = 0; i < vec->num_envs; i++) {
        RepGrid* env = (RepGrid*)vec->envs[i];
        PyObject* counts = PyTuple_New(3);
        PyTuple_SetItem(counts, 0, PyFloat_FromDouble(env->log.dogs));
        PyTuple_SetItem(counts, 1, PyFloat_FromDouble(env->log.cats));
        PyTuple_SetItem(counts, 2, PyFloat_FromDouble(env->log.tigers));
        PyList_SetItem(list, i, counts);
    }
    
    return list;
}

#define MY_METHODS {"vec_get_counts", vec_get_counts, METH_VARARGS, "Get ground truth animal counts"},
