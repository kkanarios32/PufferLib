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
        
        // Create tuple with the three integer counts
        PyObject* counts = PyTuple_New(3);
        PyTuple_SetItem(counts, 0, PyLong_FromLong((long)env->log.dogs));
        PyTuple_SetItem(counts, 1, PyLong_FromLong((long)env->log.cats));
        PyTuple_SetItem(counts, 2, PyLong_FromLong((long)env->log.tigers));
        
        PyList_SetItem(list, i, counts);
    }
    
    return list;
}

static PyObject* vec_set_probe_counts(PyObject* self, PyObject* args) {
    if (PyTuple_Size(args) != 5) {
        PyErr_SetString(PyExc_TypeError, "vec_set_probe_counts requires 5 arguments");
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

    env->pred_dogs = (int)dogs;
    env->pred_cats = (int)cats;
    env->pred_tigers = (int)tigers;

    Py_RETURN_NONE;
}
