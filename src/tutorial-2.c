#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

const char *program_source =
"__kernel void matrix_multiply (__global float *output_C, int width_A, int height_A, int width_B, int height_B, __global float *input_A, __global float *input_B) { \n"

"    int row = get_global_id (1); \n"
"    int col = get_global_id (0); \n"

"    float sum = 0.0f; \n"

"    for (int i = 0; i < width_A; i++) { \n"
"        sum += input_A[row * width_A + i] * input_B[i * width_B + col]; \n"
"    } \n"

"    output_C[row * width_B + col] = sum; \n"
"}";

int main () {

    cl_int err;
    cl_platform_id platform = NULL;
    cl_device_id device = NULL;
    cl_context context = NULL;
    cl_command_queue cmd_queue = NULL;
    cl_program program = NULL;
    cl_kernel kernel = NULL ;

    cl_mem buffer_A, buffer_B, buffer_C;
    buffer_A = buffer_B = buffer_C = NULL;

    const int width_A = 1024;
    const int height_A = 1024;
    const int width_B = 1024;
    const int height_B = width_A;
    const int width_C = width_B;
    const int height_C = height_A;

    size_t size_A = width_A * height_A * sizeof (float);
    size_t size_B = width_B * height_B * sizeof (float);
    size_t size_C = width_C * height_C * sizeof (float);

    float *A = (float *) malloc (size_A);
    float *B = (float *) malloc (size_B);
    float *C = (float *) malloc (size_C);

    for (int i = 0; i < width_A * height_A; i++) {
        A[i] = i * 0.123f;
    }
    for (int i = 0; i < width_B * height_B; i++) {
        B[i] = i * 0.321f;
    }

    err = clGetPlatformIDs (1, &platform, NULL);
    if (err != CL_SUCCESS) {
        printf ("Failed to get platform ID.\n");
        goto error;
    }

    err = clGetDeviceIDs (platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        printf ("Failed to get device ID.\n");
        goto error;
    }

    cl_context_properties cps[3] = { CL_CONTEXT_PLATFORM, (cl_context_properties) platform, 0 };

    context = clCreateContext (cps, 1, &device, NULL, NULL, &err);
    if (err != CL_SUCCESS) {
        printf ("Failed to create OpenCL context.\n");
        goto error;
    }

    cmd_queue = clCreateCommandQueue (context, device, 0, &err);
    if (err != CL_SUCCESS) {
        printf ("Failed to create command queue for device.\n");
        goto error;
    }

    buffer_A = clCreateBuffer (context, CL_MEM_READ_ONLY, size_A, NULL, &err);
    buffer_B = clCreateBuffer (context, CL_MEM_READ_ONLY, size_B, NULL, &err);
    buffer_C = clCreateBuffer (context, CL_MEM_WRITE_ONLY, size_C, NULL, &err);

    err = clEnqueueWriteBuffer (cmd_queue, buffer_A, CL_TRUE, 0, size_A, (void *) A, 0, NULL, NULL);
    err |= clEnqueueWriteBuffer (cmd_queue, buffer_B, CL_TRUE, 0, size_B, (void *) B, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf ("Failed to copy data from host to device.\n");
        goto error;
    }

    program = clCreateProgramWithSource (context, 1, (const char **) &program_source, NULL, &err);
    if (err != CL_SUCCESS) {
        printf ("Failed to create OpenCL program from source.\n");
        goto error;
    }

    err = clBuildProgram (program, 0, NULL, NULL, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf ("Failed to build program.\n");
        char build_log[16384];
        clGetProgramBuildInfo (program, device, CL_PROGRAM_BUILD_LOG, sizeof (build_log), build_log, NULL);
        printf ("Error in kernel: %s\n", build_log);
        goto error;
    }

    kernel = clCreateKernel (program, "matrix_multiply", &err);
    if (err != CL_SUCCESS) {
        printf ("Failed to create kernel.\n");
        goto error;
    }

    err  = clSetKernelArg (kernel, 0, sizeof (cl_mem), (void *) &buffer_C);
    err |= clSetKernelArg (kernel, 1, sizeof (cl_int), (void *) &width_A);
    err |= clSetKernelArg (kernel, 2, sizeof (cl_int), (void *) &height_A);
    err |= clSetKernelArg (kernel, 3, sizeof (cl_int), (void *) &width_B);
    err |= clSetKernelArg (kernel, 4, sizeof (cl_int), (void *) &height_B);
    err |= clSetKernelArg (kernel, 5, sizeof (cl_mem), (void *) &buffer_A);
    err |= clSetKernelArg (kernel, 6, sizeof (cl_mem), (void *) &buffer_B);
    if (err != CL_SUCCESS) {
        printf ("Failed to set kernel arguments.\n");
        goto error;
    }

    size_t local_work_size[2] = { 16, 16 };
    size_t global_work_size[2] = { width_C, height_C };

    err = clEnqueueNDRangeKernel (cmd_queue, kernel, 2, NULL, global_work_size, local_work_size, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf ("Failed to queue kernel for execution.\n");
        goto error;
    }

    err = clEnqueueWriteBuffer (cmd_queue, buffer_C, CL_TRUE, 0, size_C, (void *) C, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        printf ("Failed to copy data from device to host.\n");
        goto error;
    }

error:

    clReleaseKernel (kernel);
    clReleaseProgram (program);
    clReleaseCommandQueue (cmd_queue);
    clReleaseMemObject(buffer_A);
    clReleaseMemObject(buffer_B);
    clReleaseMemObject(buffer_C);
    clReleaseContext (context);

    free (A);
    free (B);
    free (C);

    return err;
}
