// hello world

#include <Meta/CUDA.h>

#define GetPhi(phi,x,y,w) phi[x+w*(y)]

void cu_Init() {
    

}

__global__ void reinit(float *phi,float* phi0, float* phin, 
                       unsigned int width, unsigned int height) {
    uint x = __umul24(blockIdx.x, blockDim.x) + threadIdx.x;
    uint y = __umul24(blockIdx.y, blockDim.y) + threadIdx.y;

    if (x > width || y > height)
        return;
    
    float xy = GetPhi(phi,x,y,width);

    float phiXPlus = 0.0f;
    float phiXMinus = 0.0f;
    float phiYPlus = 0.0f;
    float phiYMinus = 0.0f;        	
    if (x != width-1) phiXPlus  = (GetPhi(phi,x+1, y,width) - xy);
    if (x != 0)       phiXMinus = (xy - GetPhi(phi,x-1, y,width));
    
    if (y !=height-1) phiYPlus  = (GetPhi(phi,x, y+1,width) - xy);
    if (y != 0)       phiYMinus = (xy - GetPhi(phi,x, y-1,width));

    /* GetPhi(phin,x,y,width) = phiYPlus; */
    /* return; */


    float dXSquared = 0;
    float dYSquared = 0;
    float a = GetPhi(phi0,x,y,width);
    if (a > 0) {
        // formula 6.3 page 58
        float _max = max(phiXMinus, 0.0f);
        float _min = min(phiXPlus, 0.0f);
        dXSquared = max(_max*_max, _min*_min);
                    
        _max = max(phiYMinus, 0.0f);
        _min = min(phiYPlus, 0.0f);
        dYSquared = max(_max*_max, _min*_min);
    } else {
        // formula 6.4 page 58
        float _max = max(phiXPlus, 0.0f);
        float _min = min(phiXMinus, 0.0f);
        dXSquared = max(_max*_max, _min*_min);
                    
        _max = max(phiYPlus, 0.0f);
        _min = min(phiYMinus, 0.0f);
        dYSquared = max(_max*_max, _min*_min);        				
    }

    float normSquared = dXSquared + dYSquared;           
    float norm = sqrt(normSquared);

    // Using the S(phi) sign formula 7.6 on page 67
    //float sign = phi(x,y) / sqrt(phi(x,y)*phi(x,y) + normSquared);
    float sign = GetPhi(phi0,x,y,width) / 
        sqrt(GetPhi(phi0,x,y,width)*GetPhi(phi0,x,y,width) + 1);
    float t = 0.3; // A stabil CFL condition
    GetPhi(phin,x,y,width) = GetPhi(phi,x,y,width) - sign*(norm - 1)*t;


}

void cu_Reinit(float* data, 
               unsigned int w,
               unsigned int h,
               unsigned int iterations) {
    float* phiData;
    float* phi0Data;
    float* phinData;
    /* int phiPitch; */
    /* int phi0Pitch; */
    /* int phinPitch; */

    /* cudaArray* phiData; */
    /* cudaArray* phi0Data; */
    /* cudaArray* phinData; */

    /* cudaChannelFormatDesc channelDesc = */
    /*     cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat); */
    

    /* cudaMallocArray(&phiData,  &channelDesc, w, h); */
    /* cudaMallocArray(&phi0Data, &channelDesc, w, h); */
    /* cudaMallocArray(&phinData, &channelDesc, w, h); */

    /* cudaMemcpyToArray(phiData,  0, 0, data, sizeof(float)*w*h, cudaMemcpyHostToDevice); */
    /* cudaMemcpyToArray(phi0Data, 0, 0, data, sizeof(float)*w*h, cudaMemcpyHostToDevice); */
    /* cudaMemcpyToArray(phinData, 0, 0, data, sizeof(float)*w*h, cudaMemcpyHostToDevice); */


    cudaMalloc((void**)&phiData, sizeof(float)*w*h);
    cudaMalloc((void**)&phi0Data, sizeof(float)*w*h);
    cudaMalloc((void**)&phinData, sizeof(float)*w*h);
    cudaMemcpy((void*)phiData,(void*)data, sizeof(float)*w*h,cudaMemcpyHostToDevice);
    cudaMemcpy((void*)phi0Data,(void*)data, sizeof(float)*w*h,cudaMemcpyHostToDevice);
    cudaMemcpy((void*)phinData,(void*)data, sizeof(float)*w*h,cudaMemcpyHostToDevice);

    /* cudaMallocPitch((void**)&phiData, &phiPitch, sizeof(float)*w,h); */
    /* cudaMallocPitch((void**)&phi0Data, &phi0Pitch, sizeof(float)*w,h); */
    /* cudaMallocPitch((void**)&phinData, &phinPitch, sizeof(float)*w,h); */
    /* cudaMemcpy((void*)phiData,(void*)data, sizeof(float)*w*h,cudaMemcpyHostToDevice); */
    /* cudaMemcpy((void*)phi0Data,(void*)data, sizeof(float)*w*h,cudaMemcpyHostToDevice); */
    /* cudaMemcpy((void*)phinData,(void*)data, sizeof(float)*w*h,cudaMemcpyHostToDevice); */


    CHECK_FOR_CUDA_ERROR();

    const dim3 blockSize(32,16,1);
    const dim3 gridSize(w/blockSize.x, h/blockSize.y);

    //printf("%i,%i\n",w,h);

    //iterations=1;
    for (unsigned int i=0;i<iterations;i++) {
        reinit<<<gridSize,blockSize>>>(phiData,phi0Data,phinData,w,h);
        cudaMemcpy((void*)phiData,(void*)phinData,sizeof(float)*w*h,cudaMemcpyDeviceToDevice);
        cudaThreadSynchronize();
        CHECK_FOR_CUDA_ERROR();
    }

    cudaMemcpy((void*)data,(void*)phiData, sizeof(float)*w*h,cudaMemcpyDeviceToHost);
    CHECK_FOR_CUDA_ERROR();
    cudaFree(phiData);
    cudaFree(phi0Data);
    cudaFree(phinData);


}