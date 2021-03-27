
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define SOFTENING 1e-9f

// 使用cuda runtime API cudaOccupancyMaxPotentialBlockSize 来获取使得设备利用率最大的 blockSize
// API说明: Returns numBlocks and block size that achieves maximum potential occupancy for a device function.
#define BLOCKSIZE 768

/*
* Each body contains x, y, and z coordinate positions,
* as well as velocities in the x, y, and z directions.
*/
typedef struct
{
    float x, y, z, vx, vy, vz;
} Body;
/*
* Do not modify this function. A constraint of this exercise is
* that it remain a host function.
*/

void randomizeBodies(float* data, int n)
{
    for (int i = 0; i < n; i++)
    {
        data[i] = 2.0f * (rand() / (float)RAND_MAX) - 1.0f;
    }
}

// 将body之间力的计算写成一个可供GPU调用的函数，方便对一个block中的循环进行展开从而得到优化
__device__ void bodyInteraction(float3& body, float3& body2, float3& F)
{
    float dx = body.x - body2.x;
    float dy = body.y - body2.y;
    float dz = body.z - body2.z;
    float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
    float invDist = rsqrtf(distSqr);
    float invDist3 = invDist * invDist * invDist;
    F.x += dx * invDist3;
    F.y += dy * invDist3;
    F.z += dz * invDist3;
}

/*
* This function calculates the gravitational impact of all bodies in the system
* on all others, but does not update their positions.
*/

// 思路：一个线程计算一个body
__global__ void bodyForce(Body* p, float dt, int nbodies)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x; // 计算线程位置，对应到数组的下标
    int tx = threadIdx.x; // 线程位于块中的位置
    __shared__ float3 pShare[BLOCKSIZE]; // 使用shared_memory, 将数据载入shared_memory, 提升读写性能，每个块中线程共同访问shared memory
    if (i < nbodies)
    {
        float3 F = make_float3(0.0f, 0.0f, 0.0f); // 原Fx, Fy, Fz
        float3 pCur = make_float3(p[i].x, p[i].y, p[i].z); // 将有用数据放入寄存器

        // 将计算分为几个block进行，每个block的线程都进行gridDim.x(nbodies / blockDim.x)次循环，计算完一个body
        for (int block = 0; block < gridDim.x; ++block)
        {
            // 由于Body中仅仅有x,y,z是对下面计算有用的数据，因此仅读取Body中的x,y,z数据，减少shared memory的空间浪费
            Body temp = p[block * BLOCKSIZE + tx];
            pShare[tx] = make_float3(temp.x, temp.y, temp.z); // 多个线程读取到shared memory
            __syncthreads(); // 块内同步，防止数据未读取完就进入下一步

            // 进行一个block循环BLOCKSIZE次计算，本次计算后，将会重新载入shared memory，进入下一个循环运算，直到运算完nbodies次

            // 编译期参数，循环展开用
            #pragma unroll
            for (int j = 0; j < BLOCKSIZE; j++)
            {
                // 循环展开 计算body间力
                bodyInteraction(pShare[j], pCur, F); ++j;
                bodyInteraction(pShare[j], pCur, F); ++j;
                bodyInteraction(pShare[j], pCur, F); ++j;
                bodyInteraction(pShare[j], pCur, F); ++j;
            }

            __syncthreads(); // 块内同步，防止在部分线程未运算结束的情况下重载shared memory

            // 原子加操作
            atomicAdd(&p[i].vx, dt * F.x);
            atomicAdd(&p[i].vy, dt * F.y);
            atomicAdd(&p[i].vz, dt * F.z);
        }
    }
}

// 将原串行代码的每个循环分配到每个线程
__global__ void integratePosition(Body* p, float dt, int nbodies)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < nbodies)
    {
        p[i].x += p[i].vx * dt;
        p[i].y += p[i].vy * dt;
        p[i].z += p[i].vz * dt;
    }
}

int main(const int argc, const char** argv)
{

    /*
     * Do not change the value for `nBodies` here. If you would like to modify it,
     * pass values into the command line.
     */

    int nBodies = 2 << 11;
    int salt = 0;
    if (argc > 1)
        nBodies = 2 << atoi(argv[1]);

    /*
     * This salt is for assessment reasons. Tampering with it will result in automatic failure.
     */

    if (argc > 2)
        salt = atoi(argv[2]);

    const float dt = 0.01f; // time step
    const int nIters = 10;  // simulation iterations

    int bytes = nBodies * sizeof(Body);
    float* buf;

    cudaMallocManaged(&buf, bytes);
    Body* p = (Body*)buf;

    int numBlocks = (nBodies + BLOCKSIZE - 1) / BLOCKSIZE;

    /*
    * As a constraint of this exercise, `randomizeBodies` must remain a host function.
    */

    randomizeBodies(buf, 6 * nBodies); // Init pos / vel data

    double totalTime = 0.0;
    /*
    * This simulation will run for 10 cycles of time, calculating gravitational
    * interaction amongst bodies, and adjusting their positions to reflect.
    */

  /*******************************************************************/
  // Do not modify this line of code.
    for (int iter = 0; iter < nIters; iter++)
    {
        /*******************************************************************/

        /*
        * You will likely wish to refactor the work being done in `bodyForce`,
        * as well as the work to integrate the positions.
        */

        bodyForce <<<numBlocks, BLOCKSIZE>>> (p, dt, nBodies); // compute interbody forces

        /*
       * This position integration cannot occur until this round of `bodyForce` has completed.
       * Also, the next round of `bodyForce` cannot begin until the integration is complete.
       */

        integratePosition <<<numBlocks, BLOCKSIZE>>> (p, dt, nBodies);
        cudaFree(buf);
    }
}