// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include "CUDAStrategy.h"
#include <Meta/CUDA.h>
#include "LevelSet.h"
#include <Resources/Tex.h>
#include <LevelSet/SDF.h>
#include <Logging/Logger.h>

namespace OpenEngine {
namespace LevelSet {

CUDAStrategy::CUDAStrategy() {
    INITIALIZE_CUDA();
    logger.info << PRINT_CUDA_DEVICE_INFO() << logger.end;
    //cu_Init();
}

void CUDAStrategy::Reinitialize(SDF* sdf, unsigned int iterations) {

    // copy phi to cuda...
    Tex<float> phi = sdf->GetPhi();
    cu_Reinit(phi.GetData(),phi.GetWidth(),phi.GetHeight(),iterations);
    sdf->SetPhi(phi);
    
}

} // NS LevelSet
} // NS OpenEngine
