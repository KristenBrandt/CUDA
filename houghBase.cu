/*
 ============================================================================
 Author        : G. Barlas
 Version       : 1.0
 Last modified : December 2014
 License       : Released under the GNU GPL 3.0
 Description   :
 To build use  : make
 ============================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <string.h>
#include <iostream>
#include "pgm.h"
using namespace std;

const int degreeInc = 2;
const int degreeBins = 180 / degreeInc;
const int rBins = 100;
const float radInc = degreeInc * M_PI / 180;
//*****************************************************************
// The CPU function returns a pointer to the accummulator
void CPU_HoughTran (unsigned char *pic, int w, int h, int **acc)
{
  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  *acc = new int[rBins * degreeBins];
  memset (*acc, 0, sizeof (int) * rBins * degreeBins);
  int xCent = w / 2;
  int yCent = h / 2;
  float rScale = 2 * rMax / rBins;

  for (int i = 0; i < w; i++)
    for (int j = 0; j < h; j++)
      {
        int idx = j * w + i;
        if (pic[idx] > 0)
          {
            int xCoord = i - xCent;
            int yCoord = yCent - j;  // y-coord has to be reversed
            float theta = 0;         // actual angle
            for (int tIdx = 0; tIdx < degreeBins; tIdx++) //add 1 to all lines in that pixel
              {
                float r = xCoord * cos (theta) + yCoord * sin (theta);
                int rIdx = (r + rMax) / rScale;
                (*acc)[rIdx * degreeBins + tIdx]++;
                theta += radInc;
              }
          }
      }
}


__constant__ float d_Cos[degreeBins];
__constant__ float d_Sin[degreeBins];



// GPU kernel. One thread per image pixel is spawned.
// The accummulator memory needs to be allocated by the host in global memory
__global__ void GPU_HoughTran (unsigned char *pic, int w, int h, int *acc, float rMax, float rScale, float *d_Cos, float *d_Sin)
{

  int i;
  int gloID = blockIdx.x * blockDim.x + threadIdx.x ;
  if (gloID > w * h) return;      // in case of extra threads in block

  int locID = threadIdx.x;
  int xCent = w / 2;
  int yCent = h / 2;


  int xCoord = gloID % w - xCent;
  int yCoord = yCent - gloID / w;


  __shared__ int localAcc [ degreeBins * rBins ] ; // each block is using a shared memory , local accummulator


  for ( i = locID; i < degreeBins * rBins; i += blockDim.x )
    localAcc[i] = 0;

  __syncthreads ( ) ;

  if (pic[gloID] > 0)
    {
      for (int tIdx = 0; tIdx < degreeBins; tIdx++)
        {


          float r = xCoord * d_Cos[tIdx] + yCoord * d_Sin[tIdx];
          int rIdx = (r + rMax) / rScale;

          atomicAdd (localAcc + (rIdx * degreeBins + tIdx), 1);
        }
    }


  __syncthreads();
  // use atomic operations to add the local findings to the global memory accummulator
  for (i = locID; i < degreeBins * rBins; i += blockDim . x)
    atomicAdd (acc + i, localAcc [i]) ;
}

//*****************************************************************
int main (int argc, char **argv)
{
  int i;

  //cuda events
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  PGMImage inImg (argv[1]);

  int *cpuht;
  int w = inImg.x_dim;
  int h = inImg.y_dim;

  float* d_Cos;
  float* d_Sin;

  cudaMalloc ((void **) &d_Cos, sizeof (float) * degreeBins);
  cudaMalloc ((void **) &d_Sin, sizeof (float) * degreeBins);

  // CPU calculation
  CPU_HoughTran(inImg.pixels, w, h, &cpuht);

  // pre-compute values to be stored
  float *pcCos = (float *) malloc (sizeof (float) * degreeBins);
  float *pcSin = (float *) malloc (sizeof (float) * degreeBins);
  float rad = 0;
  for (i = 0; i < degreeBins; i++)
  {
    pcCos[i] = cos (rad);
    pcSin[i] = sin (rad);
    rad += radInc;
  }

  float rMax = sqrt (1.0 * w * w + 1.0 * h * h) / 2;
  float rScale = 2 * rMax / rBins;


  cudaMemcpy(d_Cos, pcCos, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);
  cudaMemcpy(d_Sin, pcSin, sizeof (float) * degreeBins, cudaMemcpyHostToDevice);


  unsigned char *d_in, *h_in;
  int *d_hough, *h_hough;

  h_in = inImg.pixels;

  h_hough = (int *) malloc (degreeBins * rBins * sizeof (int));

  cudaMalloc ((void **) &d_in, sizeof (unsigned char) * w * h);
  cudaMalloc ((void **) &d_hough, sizeof (int) * degreeBins * rBins);
  cudaMemcpy (d_in, h_in, sizeof (unsigned char) * w * h, cudaMemcpyHostToDevice);
  cudaMemset (d_hough, 0, sizeof (int) * degreeBins * rBins);

  // execution configuration uses a 1-D grid of 1-D blocks, each made of 256 threads

  int blockNum = ceil (w * h / 256);

  cudaEventRecord(start);
  GPU_HoughTran <<< blockNum, 256 >>> (d_in, w, h, d_hough, rMax, rScale, d_Cos, d_Sin);
  cudaDeviceSynchronize();

  cudaEventRecord(stop);


  // get results from device
  cudaMemcpy (h_hough, d_hough, sizeof (int) * degreeBins * rBins, cudaMemcpyDeviceToHost);


  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
   printf("Elapsed time : %.2f ms\n" ,milliseconds);


  for (i = 0; i < degreeBins * rBins; i++)
  {
    if (cpuht[i] != h_hough[i])
      printf ("Calculation mismatch at : %i %i %i\n", i, cpuht[i], h_hough[i]);
  }
  printf("Done!\n");


  CUDA_CHECK_RETURN (cudaFree ((void *) d_Cos));
  CUDA_CHECK_RETURN (cudaFree ((void *) d_Sin));
  CUDA_CHECK_RETURN (cudaFree ((void *) d_in));
  CUDA_CHECK_RETURN (cudaFree ((void *) d_hough));
  delete[]h_in;
  delete[]h_hough;
  delete[]pcCos;
  delete[]pcSin;

  return 0;
}
