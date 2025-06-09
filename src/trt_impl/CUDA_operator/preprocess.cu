#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include "trt_impl/yolo_trt_utils.h"
#include "trt_impl/taskflow_trt.h"


/*
    预处理流程：

*/
__global__ void preprocess_kernel(uint8_t* img_buffer_device, float* input_buff_device, int src_w, int src_h, int dst_w, int dst_h, float* d2s_matrix, int edge){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if(position >= edge) return;    // 因为cuda并行计算的特性，可能有部分线程是多余的，不需要实际参与计算
    int dx = position % dst_w;
    int dy = position / dst_w;
	float sx = d2s_matrix[0] * dx + d2s_matrix[1] * dy + d2s_matrix[2];
	float sy = d2s_matrix[3] * dx + d2s_matrix[4] * dy + d2s_matrix[5];

    uint8_t default_value = 128;
    int channel = 3;    // cv::Mat应该都是3通道的
    float c0,c1,c2;
    // 有效范围为实际范围外延一个像素。对于不在有效范围内的点，直接填充默认值（128）
    if(sx <= -1  || sx >= src_w || sy <= -1 || sy >= src_h){
        c0 = c1 = c2 = default_value;
    } else {
        int y_low = sy;
        int x_low = sx;
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        // 每个目标点的实际颜色是周围四个点的加权值之和
        float x_h_weight = sx - x_low;
        float y_h_weight = sy - y_low;
        float x_l_weight = 1 - x_h_weight;
        float y_l_weight = 1 - y_h_weight;

        uint8_t default_point[3] = {default_value, default_value, default_value};
        uint8_t* left_top = default_point;
        uint8_t* right_top = default_point;
        uint8_t* left_bottom = default_point;
        uint8_t* right_bottom = default_point;
        if(y_low >= 0){
            if(x_low >= 0){
                left_top = img_buffer_device + (y_low * src_w + x_low) * channel;
            }
            if (x_high < src_w){
                right_top = img_buffer_device + (y_low * src_w + x_high) * channel;
            }
        }
        if(y_high < src_h){
            if(x_low >= 0){
                left_bottom = img_buffer_device + (y_high * src_w + x_low) * channel;
            }
            if(x_high < src_w){
                right_bottom = img_buffer_device + (y_high * src_w + x_high) * channel;
            }
        }
        // cv::Mat的通道顺序是BGR，而yolo的输入是RGB，所以需要交换
        c2 = left_top[0] * x_l_weight * y_l_weight + right_top[0] * x_h_weight * y_l_weight + left_bottom[0] * x_l_weight * y_h_weight + right_bottom[0] * x_h_weight * y_h_weight;
        c1 = left_top[1] * x_l_weight * y_l_weight + right_top[1] * x_h_weight * y_l_weight + left_bottom[1] * x_l_weight * y_h_weight + right_bottom[1] * x_h_weight * y_h_weight;
        c0 = left_top[2] * x_l_weight * y_l_weight + right_top[2] * x_h_weight * y_l_weight + left_bottom[2] * x_l_weight * y_h_weight + right_bottom[2] * x_h_weight * y_h_weight;
    }
    // input_buff_device[position] = c0 / 255.0;
    // input_buff_device[position + edge] = c1 / 255.0;
    // input_buff_device[position + edge * 2] = c2 / 255.0;
	float* dst_c0 = input_buff_device + position;
    float* dst_c1 = input_buff_device + position + edge;
    float* dst_c2 = input_buff_device + position + edge * 2;
    *dst_c0 = c0 / 255.0;
     *dst_c1 = c1 / 255.0;
     *dst_c2 = c2 / 255.0;
}

void preprocess_by_cuda(cv::Mat& img, float* device_input, int dst_h, int dst_w, float* d2s_matrix, YOLO::TASK::TRT::TaskFlowTRTContext* ctx){
    // 将图片上传至显存
    uint8_t* img_buffer_host = nullptr;
    uint8_t* img_buffer_device = nullptr;
	int img_size = img.total() * img.elemSize();
	CUDA_CHECK(cudaMallocHost((void**)&img_buffer_host, img_size));
	CUDA_CHECK(cudaMalloc((void**)&img_buffer_device, img_size));
    memcpy(img_buffer_host, img.data, img_size);
    CUDA_CHECK(cudaMemcpyAsync(img_buffer_device, img_buffer_host, img_size, cudaMemcpyHostToDevice, ctx->stream));
    TRACE("preprocess ---- update img", ctx);

    int src_w = img.cols;
    int src_h = img.rows;
    // 分配cuda线程
    int jobs = dst_h * dst_w;  // 每个像素给线程，负责完成该像素的生成、rgb拆解、归一化操作
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);

    preprocess_kernel<<<blocks, threads, 0, ctx->stream>>>(img_buffer_device, device_input, src_w, src_h, dst_w, dst_h, d2s_matrix, jobs);

    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    TRACE("preprocess ---- call kernel", ctx);

    CUDA_CHECK(cudaFree(img_buffer_device));
    CUDA_CHECK(cudaFreeHost(img_buffer_host));
}
