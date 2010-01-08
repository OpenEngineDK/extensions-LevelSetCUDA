
SET( EXTENSION_NAME "Extensions_LevelSetCUDA")

CUDA_ADD_LIBRARY("CUDA_KERNELS"
  LevelSet/LevelSet.cu
#   Renderers/CUDA/RayCaster.cu
#   Utils/CUDA/DoseCalc.cu
#   Utils/CUDA/DozeSetup.cu
# #  Utils/CUDA/Superposition.cu
)


# Create the extension library
ADD_LIBRARY(${EXTENSION_NAME}
  LevelSet/CUDAStrategy.cpp
  LevelSet/CUDAStrategy.h
)

TARGET_LINK_LIBRARIES( ${EXTENSION_NAME}
  "CUDA_KERNELS"
)
