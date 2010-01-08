// 
// -------------------------------------------------------------------
// Copyright (C) 2007 OpenEngine.dk (See AUTHORS) 
// 
// This program is free software; It is covered by the GNU General 
// Public License version 2 or any later version. 
// See the GNU General Public License for more details (see LICENSE). 
//--------------------------------------------------------------------

#include <LevelSet/Strategy.h>

namespace OpenEngine {
namespace LevelSet {

/**
 * Short description.
 *
 * @class CUDAStrategy CUDAStrategy.h ons/LevelSetCUDA/LevelSet/CUDAStrategy.h
 */
class CUDAStrategy : public Strategy {
public:
    CUDAStrategy();
    void Reinitialize(SDF* sdf, unsigned int iterations);
};

} // NS LevelSet
} // NS OpenEngine
