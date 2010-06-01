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

void CUDAStrategy::BuildGradient(SDF* sdf) {
    const unsigned int width=sdf->GetWidth(),
        height=sdf->GetHeight();
    Tex<float> phi = sdf->GetPhi();
    Tex<Vector<2,float> > gradient = sdf->GetGradient();

    const unsigned int Y = height;
    const unsigned int X = width;

    float dx = 1;
    float dy = 1;
    float cdX, cdY;
    for (unsigned int x=0; x<X; x++)
        for (unsigned int y=0; y<Y; y++) {
      
            //lower left corner
            if (x == 0 && y == 0) {
                cdX = -(phi(x, y) - phi(x+1, y)) / dx;
                cdY = -(phi(x, y) - phi(x, y+1)) / dy;

            } 
            //lower right corner
            else if (x == X - 1 && y == 0) {
                cdX = (phi(x, y) - phi(x-1, y)) / dx;
                cdY = -(phi(x, y) - phi(x, y+1)) / dy;
            }
            //upper left corner
            else if (x == 0 && y == Y - 1) {
                cdX = -(phi(x, y) - phi(x+1, y)) / dx;
                cdY = (phi(x, y) - phi(x, y-1)) / dy;

            }      
            //upper right corner
            else if (x == X - 1 && y == Y - 1) {
                cdX = (phi(x, y) - phi(x-1, y)) / dx;
                cdY = (phi(x, y) - phi(x, y-1)) / dy;

            }

            // upper border
            else if (y == 0 && (x > 0 && x < X - 1)) {
                cdX = -(phi(x-1, y) - phi(x+1, y)) / 2 * dx;
                cdY = -(phi(x, y)   - phi(x, y+1)) / dy;

            }       
            // lower border
            else if (y == Y - 1 && (x > 0 && x < X - 1)) {
                cdX = -(phi(x-1, y) - phi(x+1, y)) / 2 * dx;
                cdY = (phi(x, y)   - phi(x, y-1)) / dy;

            }
            // left border
            else if (x == 0 && (y > 0 && y < Y - 1)) {
                cdX = -(phi(x, y)   - phi(x+1, y)) / dx;
                cdY = -(phi(x, y-1) - phi(x, y+1)) / 2 * dy;

            }
            // right border
            else if (x == X - 1 && (y > 0 && y < Y - 1)) {
                cdX = (phi(x, y)   - phi(x-1, y)) / dx;
                cdY = -(phi(x, y-1) - phi(x, y+1)) / 2 * dy;

            }
            // Normal case
            else {
	
                // central differences
                cdX = -(phi(x-1, y) - phi(x+1, y)) / 2 * dx;
                cdY = -(phi(x, y-1) - phi(x, y+1)) / 2 * dy;
            }

            
            Vector<2, float> g(cdX, cdY);
            // if (g.IsZero()) 
            //     g = Vector<2,float>(0,1);
                
            // g.Normalize();
            gradient(x,y) = g;

        }   
    sdf->SetGradient(gradient);
}


} // NS LevelSet
} // NS OpenEngine
