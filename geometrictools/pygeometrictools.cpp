#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

// disable NDEBUG, load cassert and re-enable NDEBUG if it was set
// from https://stackoverflow.com/a/24275850
#ifdef NDEBUG
# define NDEBUG_DISABLED
# undef NDEBUG
#endif
#include <cassert>
#ifdef NDEBUG_DISABLED
# define NDEBUG        // re-enable NDEBUG if it was originally enabled
#endif

#include <iostream>
#include <Python.h>
#include <numpy/arrayobject.h> // numpy
#include "domains/interval.h"
#include "utils/hyperbox.h"
#include "utils/image.h"
#include <libgeometrictools.h>


using namespace std;

float degToRad(float deg) {
  return deg * Constants::PI / 180.0f;
}

float radToDeg(float rot) {
  return rot * 180.0f / Constants::PI;
}

Image toImg(PyObject *img) {
  if (PyTuple_Check(img)) {
    assert(PyTuple_Size(img) == 2);
    PyObject *lower = PyTuple_GetItem(img, 0);
    PyObject *upper = PyTuple_GetItem(img, 1);
    assert(PyObject_TypeCheck(lower, &PyArray_Type));
    assert(PyObject_TypeCheck(upper, &PyArray_Type));
    PyArrayObject* lower_np = (PyArrayObject*) lower;
    PyArrayObject* upper_np = (PyArrayObject*) upper;
    int lower_typ = PyArray_TYPE(lower_np);
    int upper_typ = PyArray_TYPE(upper_np);
    assert(lower_typ == NPY_FLOAT32 || lower_typ == NPY_UINT8);
    assert(upper_typ == NPY_FLOAT32 || upper_typ == NPY_UINT8);
    Image lower_img = Image(lower_np);
    Image upper_img = Image(upper_np);
    for(size_t i = 0; i < lower_img.nRows(); ++i)
      for(size_t j = 0; j < lower_img.nCols(); ++j)
        for(size_t k = 0; k < lower_img.nChannels(); ++k) {
          const size_t m = lower_img.img->rcc_to_idx(i, j, k);
          lower_img.img->a[m] = Interval(lower_img.img->a[m].inf,
                                         upper_img.img->a[m].sup);
        }    
    return lower_img;    
  } else {
    assert(PyObject_TypeCheck(img, &PyArray_Type));
    PyArrayObject* img_np = (PyArrayObject*) img;
    int img_typ = PyArray_TYPE(img_np);
    assert(img_typ == NPY_FLOAT32 || img_typ == NPY_UINT8);
    return Image(img_np);    
  }
}

PyObject* toNumpy(Image& img_in, bool lower, const size_t startR, const size_t startC, const size_t dR, const size_t dC) {
  Image img = img_in.to_cpu(); 
  size_t nRows = img.nRows();
  size_t nCols = img.nCols();
  size_t nChannels = img.nChannels();
  assert(startR + dR <= nRows);
  assert(startC + dC <= nCols);
  npy_intp dims[3];  
  dims[0]= dR;
  dims[1]= dC;
  dims[2]= nChannels;
  PyArrayObject *array = (PyArrayObject *) PyArray_SimpleNew(3, dims, NPY_FLOAT32);  
  float *p = (float *) PyArray_DATA(array);

  for(size_t i = 0; i < dR; ++i)
    for(size_t j = 0; j < dC; ++j)
      for(size_t k = 0; k < nChannels; ++k) {
        const size_t idx = i * dC * nChannels + j * nChannels + k;
        const size_t m = img.img->rcc_to_idx(startR + i, startC + j, k);
        assert(idx <= dR*dC*nChannels);
        Interval v = img.img->a[m];
        p[idx] =  (lower) ? v.inf : v.sup;
      }
  return (PyObject*) array;
};

PyObject* toNumpy(Image& img_in, bool lower) {
  return toNumpy(img_in, lower, 0, 0, img_in.nRows(), img_in.nCols());
};


PyObject* toNumpy(Image& img_in) {
  return Py_BuildValue("(OO)", toNumpy(img_in, true, 0, 0, img_in.nRows(), img_in.nCols()),
                               toNumpy(img_in, false, 0, 0, img_in.nRows(), img_in.nCols())); 
};


const char* doc_rotate = "Roates an image by gamma degrees. Arguments:\n"
  "image: Numpy array (2d/3d in CHW order) containing the image or a tuple of them denoting lower/upper.\n"
  "gamma: Float. Number of degrees to rotate the image.\n"
  "roundToInt: Bool. Indicating whether to round the result to k/255.\n"
  "gpu: (Optional) Integer. Id of the GPU to use. -1 to disable it. [Default: -1]\n"
  "\n"
  "Returns: a tuple of numpy arrays, denoting the lower and upper bound.\n";
static PyObject* py_rotate(PyObject* self, PyObject* args, PyObject* kwargs) {
  PyObject *image;
  float gamma;
  int gpu = -1;
  int roundToInt;
  char *kwlist[] = {
    "image",
    "gamma",
    "roundToInt",
    "gpu",
    NULL};
  int status = PyArg_ParseTupleAndKeywords(args, kwargs, "Ofp|i", kwlist,
                                           &image,
                                           &gamma,
                                           &roundToInt,
                                           &gpu);
  assert(status);
  Image img = toImg(image);
  if(gpu > -1) {
    img = img.to_gpu(gpu);
  }
  gamma = degToRad(gamma);
  Image img_rot = img.rotate(gamma, roundToInt);
  return Py_BuildValue("O", toNumpy(img_rot));
}

const char* doc_translate = "Translates an image by dx, dy. Arguments:\n"
  "image: Numpy array (2d/3d in CHW order) containing the image or a tuple of them denoting lower/upper.\n"
  "dx: Float. Distance in x direction.\n"
  "dy: Float. Distance in x direction.\n"
  "roundToInt: Bool. Indicating whether to round the result to k/255.\n"
  "gpu: (Optional) Integer. Id of the GPU to use. -1 to use CPU. [Default: -1]\n"
  "\n"
  "Returns: a tuple of numpy arrays, denoting the lower and upper bound.\n";
static PyObject* py_translate(PyObject* self, PyObject* args, PyObject* kwargs) {
  PyObject *image;
  float dx, dy;
  int gpu = -1;
  int roundToInt;
  char *kwlist[] = {
    "image",
    "dx",
    "dy",
    "roundToInt",
    "gpu",
    NULL};
  int status = PyArg_ParseTupleAndKeywords(args, kwargs, "Offp|i", kwlist,
                                           &image,
                                           &dx,
                                           &dy,
                                           &roundToInt,
                                           &gpu);
  assert(status);
  Image img = toImg(image);
  if(gpu > -1) {
    img = img.to_gpu(gpu);
  }
  Image img_rot = img.translate(dx, dy, roundToInt);
  return Py_BuildValue("O", toNumpy(img_rot));
}


const char* doc_getE = "Computes the maximal interpolation error for the given transformation. Arguments:\n"
  "image: Numpy array (2d/3d in CHW order) containing the image or a tuple of them denoting lower/upper.\n"
  "transformation: \"rot\" or \"trans\" to indicate either rotation or translation.\n"
  "targetE: Float. Error to aim for. If reached stop computation.\n"
  "stopE: Float. If the minimal error on any region is larger than this value, abort computation.\n"
  "gamma0: Float. Lower bound of the transformation parameter space. Degrees for rotation, pixels (both dx/dx) for translation.\n"
  "gamma1: Float. Upper bound of the transformation parameter space. Degrees for rotation, pixels (both dx/dx) for translation.\n"
  "betas: Numpy Array (1d/2d) containing sampled values for beta. For rotation shape is (1, n) or (n, ) and for translation (2, n).\n"
  "invertT: (Optional.) Bool. If True it is assume that the given image was perturbed and the original is computed by inverting it. If False treats the input as the original and computes the transformed image by interval transformation. [Default: False]\n"  
  "resizePostTransform: (Optional.) Integer. Target size for a resize operation after transformation. Values <= 0 disables the resize. [Default: 0]\n"
  "centerCropPostTranform: (Optional.) Integer. Target size for a center crop operation after transformation (and resize). Values <= 0 disables the center crop. [Default: 0]\n"
  "filterSigma: (Optional.) Float. Standard deviation of the Gaussian filter applied to the image after the transformation. Values <= 0 disable the filter. [Default: 0]\n"
  "filterSize: (Optional.) Integer. Size of the Gaussian filter applied to the transformation after the transformation. Values <= 0 disable the filter. [Default: 0]\n"
  "radiusDecrease: (Optional.) Float. Denotes how many pixels to discount from the radius of the inscribed vingette. 0 denotes the largest circular vingette. Values < 0 disable vingetting. [Default: -1]\n"
  "initialSplits: (Optional.) Integer. Number of splits for the [gamma0, gamma1] region before starting adaptive splitting. [Default: 10]\n"
  "batchSize: (Optional.) Integer. Images to compute at the same time. Only used on GPU. [Default: 1]\n"
  "threads: (Optional.) Integer. Number of threads to use. [Default: 1]\n"
  "gpu: (Optional.) Bool. Use GPUs. [Default: False]\n"
  "debug: (Optional.) Bool. Write debug output. [Default: False]\n"
  "doInt: (Optional.) Bool. Account for integer errors in all steps of the computation. [Default: True]\n"
  "refinements: (Optional.) Integer. Number of refinement steps to use in the inverse computation. Not used if invertT is False. [Default: 10]\n"
  "timeout: (Optional.) Integer. timeout in seconds for the computation. [Default: 120]\n"
  "\n"
  "Returns: A tuple with the list of errors for each beta along with the maximal absolute parameters for which a non-empty inverse was observed.\n";
static PyObject* py_getE(PyObject* self, PyObject* args, PyObject* kwargs)
{
  arg_t getE_args;
  PyObject *image;
  int inv_gamma;
  float gamma0, gamma1;
  PyObject *betas;
  int debug = 0;
  int doInt = 1;
  int gpu = 0;
  char* transformation;
  getE_args.resize_postTransform = 0;
  getE_args.center_crop_postTranform = 0;
  getE_args.filter_sigma = 0;
  getE_args.filter_size = 0;
  getE_args.radiusDecrease = -1;
  getE_args.initialSplits = 10;
  getE_args.batch_size = 1;
  getE_args.threads = 1;
  getE_args.refinements = 10;
  getE_args.timeout = 120;
  
  char *kwlist[] = {
    "image",
    "transformation",
    "targetE",
    "stopE",
    "gamma0",
    "gamma1",
    "betas",
    "invertT",  
    "resizePostTransform",
    "centerCropPostTranform",
    "filterSigma",
    "filterSize",
    "radiusDecrease",
    "initialSplits",
    "batchSize",
    "threads",
    "gpu",
    "debug",
    "doInt",
    "refinements",
    "timeout",           
    NULL};

  int status = PyArg_ParseTupleAndKeywords(args, kwargs, "OsffffOp|kkfkfkkkpppkk", kwlist,
                                           &image,
                                           &transformation,
                                           &getE_args.target,
                                           &getE_args.stopErr,
                                           &gamma0,
                                           &gamma1,
                                           &betas,
                                           &inv_gamma,
                                           &getE_args.resize_postTransform,
                                           &getE_args.center_crop_postTranform,
                                           &getE_args.filter_sigma,
                                           &getE_args.filter_size,
                                           &getE_args.radiusDecrease,
                                           &getE_args.initialSplits,
                                           &getE_args.batch_size,
                                           &getE_args.threads,
                                           &gpu,
                                           &debug,
                                           &doInt,
                                           &getE_args.refinements,
                                           &getE_args.timeout);
  assert(status);
  assert(getE_args.target > 0);
  getE_args.inv = inv_gamma;
  getE_args.debug = debug;
  getE_args.doInt = doInt;
  getE_args.gpu = gpu;
  
  Image img = toImg(image);
  getE_args.img = img;
  
  size_t dims;
  string transformation_str = transformation;
  if (transformation_str == "rot") {
    getE_args.params = HyperBox({Interval(degToRad(gamma0), degToRad(gamma1))});
    getE_args.transform = Transform::rotation;
    dims = 1;
  } else if (transformation_str == "trans") {
    getE_args.params = HyperBox({Interval(gamma0, gamma1), Interval(gamma0, gamma1)});
    getE_args.transform = Transform::translation;
    dims = 2;
  } else return NULL;

  assert(PyObject_TypeCheck(betas, &PyArray_Type));

  PyArrayObject* betas_np = (PyArrayObject*) betas;  
  int betas_dims = PyArray_NDIM(betas_np);
  assert(betas_dims == 1 || betas_dims == 2);

  size_t nBetas, dimBetas;
  if (betas_dims == 1) {
    nBetas = PyArray_DIM(betas_np, 0);
    dimBetas = 1;
  } else if (betas_dims == 2) {
    nBetas = PyArray_DIM(betas_np, 1);
    dimBetas = PyArray_DIM(betas_np, 0);
  }
  assert(dimBetas == dims); 
    
  int betas_typ = PyArray_TYPE(betas_np);
  vector<vector<float>> betas_;
  for(size_t i = 0; i < dims; ++i) {
    betas_.push_back(vector<float>());
  }

  void* pbeta = PyArray_DATA(betas_np);
  auto getV = [&](size_t k) {
    float v;
    switch(betas_typ) {
    case NPY_FLOAT32:
      v = ((float*)pbeta)[k];
      break;
    case NPY_DOUBLE:
      v = (float) ((double*)pbeta)[k];
      break;
    default:
      cout << "unsupported data type " << endl;
      assert(false);
    }
    if (getE_args.transform == Transform::rotation) {
      v = degToRad(v);
    }
    return v;
  };

    
  for(size_t k = 0; k < nBetas; ++k) {
    for(size_t d = 0; d < dims; ++d) {
      float v = getV(k + d * nBetas);
      betas_[d].push_back(v);
    }
  }
  getE_args.betas = betas_;

  vector<float> e;
  vector<float> maxParam;
  tie(e, maxParam) = getE(getE_args);
    
  PyObject* listErr = PyList_New(e.size());
  if(listErr) {
    for(size_t k = 0; k < e.size(); ++k) {
      PyObject* err = Py_BuildValue("f", e[k]);
      PyList_SetItem(listErr, k, err);
    }
  }
  
  npy_intp maxParam_dims[1];  
  maxParam_dims[0] = maxParam.size();
  PyArrayObject *maxParam_array = (PyArrayObject *) PyArray_SimpleNew(1, maxParam_dims, NPY_FLOAT32);  
  float *p = (float *) PyArray_DATA(maxParam_array);
  for(size_t i = 0; i < maxParam.size(); ++i) {
    float v = maxParam[i];
    if (getE_args.transform == Transform::rotation) {
      v = radToDeg(v);
    }
    p[i] = v;
  }
    
  return Py_BuildValue("OO", listErr, maxParam_array);
}

static PyMethodDef module_methods[] = {
  {"getE", (PyCFunction) py_getE, METH_VARARGS | METH_KEYWORDS, doc_getE},
  {"rotate", (PyCFunction) py_rotate, METH_VARARGS | METH_KEYWORDS, doc_rotate},
  {"translate", (PyCFunction) py_translate, METH_VARARGS | METH_KEYWORDS, doc_translate},
  {NULL}
};

  static struct PyModuleDef PyGeometricTools =
  {
    PyModuleDef_HEAD_INIT,
    "geometrictools", /* name of module */
    "Library for analyzing geometric image transformations.", /* module documentation */
    -1,   /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    module_methods
  };

  PyMODINIT_FUNC PyInit_geometrictools(void)
  {
    PyObject* o = PyModule_Create(&PyGeometricTools);
    import_array(); //init numpy
    initGeometricTools();
    return o;
  }
