import os, sys
from distutils.core import setup, Extension
import numpy as np
from sysconfig import get_paths as gp
from pathlib import Path

static_libraries = ['pygeometrictools']
static_lib_dir = os.path.join(os.getcwd(), 'lib')
libraries = ['cudart', 'python3.6m', 'pthread', 'dl', 'util', 'rt', 'm']
library_dirs = ['/usr/local/cuda/lib64',
                str(Path(gp()['stdlib']).parent)]
include_dirs = [os.getcwd(),
                os.path.join(os.getcwd(), 'indicators', 'include'),
                gp()['include'],
                np.get_include(),
                '/usr/local/cuda/include']

if sys.platform == 'win32':
    libraries.extend(static_libraries)
    library_dirs.append(static_lib_dir)
    extra_objects = []
else: # POSIX
    extra_objects = ['{}/lib{}.a'.format(static_lib_dir, l) for l in static_libraries]

    ext = Extension('geometrictools',
                    sources=['pygeometrictools.cpp'],
                    libraries=libraries,
                    library_dirs=library_dirs,
                    include_dirs=include_dirs,
                    extra_objects=extra_objects,
                    extra_compile_args=['-Wno-write-strings'])

setup(name='geometrictools', version='1.0',
      ext_modules=[ext])
