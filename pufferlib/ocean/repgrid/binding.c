#include "repgrid.h"
#include <Python.h>
#include <numpy/arrayobject.h>

#define Env RepGrid

static PyObject* vec_get_counts(PyObject* self, PyObject* args);
static PyObject* vec_set_probe_counts(PyObject* self, PyObject* args);
#define MY_METHODS \
    {"vec_get_counts", vec_get_counts, METH_VARARGS, "Get ground truth animal counts"}, \
    {"vec_set_probe_counts", vec_set_probe_counts, METH_VARARGS, "Set probe-predicted animal counts"}

#include "../env_binding.h"

static char* unpack_string(PyObject* kwargs, char* key) {
  PyObject* val = PyDict_GetItemString(kwargs, key);
  if (val == NULL) {
    return NULL;
  }
  if (!PyUnicode_Check(val)) {
    return NULL;
  }
  return (char*)PyUnicode_AsUTF8(val);
}

static float* unpack_float_array(PyObject* kwargs, char* key, int* out_size) {
  PyObject* val = PyDict_GetItemString(kwargs, key);
  if (val == NULL || !PyArray_Check(val)) {
    *out_size = 0;
    return NULL;
  }
  PyArrayObject* arr = (PyArrayObject*)val;
  if (!PyArray_ISCONTIGUOUS(arr)) {
    *out_size = 0;
    return NULL;
  }
  *out_size = (int)PyArray_DIM(arr, 0);
  return (float*)PyArray_DATA(arr);
}

static int my_init(Env *env, PyObject *args, PyObject *kwargs) {
  env->size = unpack(kwargs, "size");
  env->random_sampling = unpack(kwargs, "random_sampling");
  env->key_door = (int)unpack(kwargs, "key_door");
  env->feature_dim = (int)unpack(kwargs, "feature_dim");

  // Default to spiral mode if not specified
  env->sampling_mode = SAMPLING_SPIRAL;

  char* mode_str = unpack_string(kwargs, "sampling_mode");
  if (mode_str != NULL) {
    if (strcmp(mode_str, "mnist") == 0) {
      env->sampling_mode = SAMPLING_MNIST;
    } else if (strcmp(mode_str, "spiral") == 0) {
      env->sampling_mode = SAMPLING_SPIRAL;
    } else if (strcmp(mode_str, "gaussian") == 0) {
      env->sampling_mode = SAMPLING_GAUSSIAN;
    }
  }

  // Load MNIST buffers if in MNIST mode
  env->mnist_0 = NULL;
  env->mnist_1 = NULL;
  env->mnist_2 = NULL;
  env->mnist_3 = NULL;  // empty spaces
  env->mnist_4 = NULL;  // keys
  env->mnist_0_count = 0;
  env->mnist_1_count = 0;
  env->mnist_2_count = 0;
  env->mnist_3_count = 0;
  env->mnist_4_count = 0;

  if (env->sampling_mode == SAMPLING_MNIST) {
    env->mnist_0 = unpack_float_array(kwargs, "mnist_0", &env->mnist_0_count);
    env->mnist_1 = unpack_float_array(kwargs, "mnist_1", &env->mnist_1_count);
    env->mnist_2 = unpack_float_array(kwargs, "mnist_2", &env->mnist_2_count);
    env->mnist_3 = unpack_float_array(kwargs, "mnist_3", &env->mnist_3_count);
    env->mnist_4 = unpack_float_array(kwargs, "mnist_4", &env->mnist_4_count);
  }

  init_repgrid(env);
  return 0;
}

static int my_log(PyObject *dict, Log *log) {
  assign_to_dict(dict, "perf", log->perf);
  assign_to_dict(dict, "score", log->score);
  assign_to_dict(dict, "episode_return", log->episode_return);
  assign_to_dict(dict, "episode_length", log->episode_length);
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
        
        PyObject* counts = PyTuple_New(4);
        PyTuple_SetItem(counts, 0, PyLong_FromLong((long)env->log.dogs));
        PyTuple_SetItem(counts, 1, PyLong_FromLong((long)env->log.cats));
        PyTuple_SetItem(counts, 2, PyLong_FromLong((long)env->log.tigers));
        PyTuple_SetItem(counts, 3, PyLong_FromLong((long)env->log.keys));
        
        PyList_SetItem(list, i, counts);
    }
    
    return list;
}

static PyObject* vec_set_probe_counts(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 6) {
        PyErr_SetString(PyExc_TypeError, "vec_set_probe_counts requires 6 arguments");
        return NULL;
    }

    VecEnv* vec = unpack_vecenv(args);
    if (!vec) {
        return NULL;
    }

    PyObject* env_id_arg = PyTuple_GetItem(args, 1);
    if (!PyObject_TypeCheck(env_id_arg, &PyLong_Type)) {
        PyErr_SetString(PyExc_TypeError, "env_id must be an integer");
        return NULL;
    }
    int env_id = PyLong_AsLong(env_id_arg);
    if (env_id < 0 || env_id >= vec->num_envs) {
        PyErr_SetString(PyExc_IndexError, "env_id out of range");
        return NULL;
    }

    RepGrid* env = (RepGrid*)vec->envs[env_id];

    long dogs = PyLong_AsLong(PyTuple_GetItem(args, 2));
    long cats = PyLong_AsLong(PyTuple_GetItem(args, 3));
    long tigers = PyLong_AsLong(PyTuple_GetItem(args, 4));
    long keys = PyLong_AsLong(PyTuple_GetItem(args, 5));

    env->pred_dogs = (int)dogs;
    env->pred_cats = (int)cats;
    env->pred_tigers = (int)tigers;
    env->pred_keys = (int)keys;

    Py_RETURN_NONE;
}
