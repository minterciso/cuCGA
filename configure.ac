# ------------------------------------------------------------------------------
AC_PREREQ([2.64])
AC_INIT([cuCga], [1.0.0], [zarnick@geekvault.org])
AM_INIT_AUTOMAKE([cuCga], [1.0.0])
AC_CONFIG_SRCDIR([src/main.c])
AC_CONFIG_HEADERS([config.h])

ARCH=`uname -m | grep -o 64`
# Setup CUDA paths
# ------------------------------------------------------------------------------
AC_ARG_WITH([cuda],
   [  --with-cuda=PATH  		    prefix where cuda is installed [default=auto]])
if test -n "$with_cuda"
then
   CUDA_CFLAGS="-I$with_cuda/include"
   CUDA_LIBS="-L$with_cuda/lib$ARCH -lcudart"
   NVCC="$with_cuda/bin/nvcc"
   CUDA_PATH="$with_cuda"
else
   CUDA_CFLAGS="-I/usr/local/cuda/include"
   CUDA_LIBS="-L/usr/local/cuda/lib$ARCH -lcudart"
   NVCC="nvcc"
   CUDA_PATH="/usr/local/cuda/"
fi
AC_SUBST(CUDA_CFLAGS)
AC_SUBST(CUDA_LIBS)
AC_SUBST(NVCC)
AC_SUBST(CUDA_PATH)

AC_ARG_ENABLE([debug],
   [  --enable-debug    Turn on device debug for CUDA],
   [case "${debugvall}" in
       yes) DEBUG=true;;
       no)  DEBUG=false;;
       *) AC_MSG_ERROR([bad value ${debugval} for --enable-debug]);;
   esac],
   [DEBUG=false]
)
AC_ARG_ENABLE([emu],
   [  --enable-emu    Turn on device emulation for CUDA],
   [case "${enableval}" in
       yes) EMULATION=true;;
       no)  EMULATION=false;;
       *) AC_MSG_ERROR([bad value ${enableval} for --enable-emu]);;
   esac],
   [EMULATION=false]
)

# ------------------------------------------------------------------------------
# Setup nvcc flags
# ------------------------------------------------------------------------------
if test x$DEBUG = xtrue 
then
   NVCCFLAGS="-g -G"
else
   NVCCFLAGS="-O3 -use_fast_math"
fi
if test x$EMULATION = xtrue 
then
   NVCCFLAGS+=" -deviceemu -DEMULATION"
   NVCC_CFLAGS+=" -deviceemu -DEMULATION"
fi
AC_SUBST(NVCCFLAGS)
AC_SUBST(NVCC_CFLAGS)


# Checks for programs.
AC_PROG_CC
AM_PROG_CC_C_O

# Checks for libraries.
AC_CHECK_LIB([m], [pow])

# Checks for header files.
AC_CHECK_HEADERS([limits.h stdlib.h string.h math.h $CUDA_PATH/include/cuda.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_TYPE_SIZE_T

# Checks for library functions.

AC_CONFIG_FILES([Makefile
                 src/Makefile])

AC_OUTPUT
