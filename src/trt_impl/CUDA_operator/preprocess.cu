﻿#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

#include "trt_impl/yolo_trt_utils.h"
#include "trt_impl/taskflow_trt.h"


/*
    预处理流程：

*/
__global__ void preprocess_kernel(uint8_t* img_buffer_device, float* input_buff_device, int src_w, int src_h, int dst_w, int dst_h, float* d2s_matrix,
		unsigned char* mask, int mask_h, int mask_w,	// 带掩膜预处理。分割+后续时，分割会输出一个mask，后续模型复用原图+mask，生成一个只保留mask部分的预处理图
		float* roi,	// 带roi预处理。目标检测+后续时，目标检测会输出一个roi，后续模型会基于roi更新d2s_matrix,并传入roi的起点坐标
		int edge){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if(position >= edge) return;    // 因为cuda并行计算的特性，可能有部分线程是多余的，不需要实际参与计算
    int dx = position % dst_w;
    int dy = position / dst_w;
    uint8_t default_value = 128;


	float sx = d2s_matrix[0] * (float)dx + d2s_matrix[1] * (float)dy + d2s_matrix[2];
	float sy = d2s_matrix[3] * (float)dx + d2s_matrix[4] * (float)dy + d2s_matrix[5];

    
    int channel = 3;    // cv::Mat应该都是3通道的
    float c0,c1,c2;
	bool done = false;

	int w_limit = src_w;
	int h_limit = src_h;
	int x_offset = 0;
	int y_offset = 0;
	if (roi != NULL) {
		w_limit = roi[2] - roi[0];
		h_limit = roi[3] - roi[1];
		x_offset = roi[0];
		y_offset = roi[1];
	}

    // 有效范围为实际范围外延一个像素。对于不在有效范围内的点，直接填充默认值（128）
    if(sx <= -1  || sx >= w_limit || sy <= -1 || sy >= h_limit){
	//if (sx <= 0 || (sx >= w_limit - 1) || sy <= 0 || sy >= (h_limit - 1)) {
        c0 = c1 = c2 = default_value;
		done = true;
	} else if (mask != nullptr) {
		// 如果有掩膜，那么只有掩膜值非0的位置才会计算实际颜色
		if (mask != nullptr) {
			int mx = dx * mask_w / dst_w;
			int my = dy * mask_h / dst_h;
			if (mask[my * mask_w + mx] == 0) {
				c0 = c1 = c2 = 255;
				done = true;
			}
		}
	}
	if(!done) {
        int y_low = sy;
        int x_low = sx;
        int y_high = y_low + 1;
        int x_high = x_low + 1;

        // 每个目标点的实际颜色是周围四个点的加权值之和
        float x_h_weight = sx - x_low;
        float y_h_weight = sy - y_low;
        float x_l_weight = 1. - x_h_weight;
        float y_l_weight = 1. - y_h_weight;

        uint8_t default_point[3] = {128, 128, 128};
        uint8_t* left_top = default_point;
        uint8_t* right_top = default_point;
        uint8_t* left_bottom = default_point;
        uint8_t* right_bottom = default_point;
        //if(y_low >= 0){
        //    if(x_low >= 0){
        //        left_top = img_buffer_device + ((y_low + y_offset) * src_w + x_low + x_offset) * channel;
        //    }
        //    if (x_high < w_limit){
        //        right_top = img_buffer_device + ((y_low + y_offset) * src_w + x_high + x_offset) * channel;
        //    }
        //}
        //if(y_high < h_limit){
        //    if(x_low >= 0){
        //        left_bottom = img_buffer_device + ((y_high + y_offset)* src_w + x_low + x_offset) * channel;
        //    }
        //    if(x_high < w_limit){
        //        right_bottom = img_buffer_device + ((y_high + y_offset) * src_w + x_high + x_offset) * channel;
        //    }
        //}
		if (y_low < 0) {
			// 第一行, top行不用处理
			if (x_low >= 0) {
				left_bottom = img_buffer_device + ((y_high + y_offset)* src_w + x_low + x_offset) * channel;
			}
			if (x_high < w_limit) {
				right_bottom = img_buffer_device + ((y_high + y_offset) * src_w + x_high + x_offset) * channel;
			}
		} else if (y_high >= h_limit) {
			// 最后一行，bottom不用处理
			if (x_low >= 0) {
				left_top = img_buffer_device + ((y_low + y_offset) * src_w + x_low + x_offset) * channel;
			}
			if (x_high < w_limit) {
				right_top = img_buffer_device + ((y_low + y_offset) * src_w + x_high + x_offset) * channel;
			}
		} else if (x_low < 0) {
			// 中间行，最左列，left不用处理
			right_top = img_buffer_device + ((y_low + y_offset) * src_w + x_high + x_offset) * channel;
			right_bottom = img_buffer_device + ((y_high + y_offset) * src_w + x_high + x_offset) * channel;
		} else if (x_high >= w_limit) {
			// 中间行，最右列，right不用处理
			left_top = img_buffer_device + ((y_low + y_offset) * src_w + x_low + x_offset) * channel;
			left_bottom = img_buffer_device + ((y_high + y_offset)* src_w + x_low + x_offset) * channel;
		} else {
			// 排除4边情况，剩下的都是中部点
			left_top = img_buffer_device + ((y_low + y_offset) * src_w + x_low + x_offset) * channel;
			left_bottom = img_buffer_device + ((y_high + y_offset)* src_w + x_low + x_offset) * channel;
			right_top = img_buffer_device + ((y_low + y_offset) * src_w + x_high + x_offset) * channel;
			right_bottom = img_buffer_device + ((y_high + y_offset) * src_w + x_high + x_offset) * channel;
		}
        // cv::Mat的通道顺序是BGR，而yolo的输入是RGB，所以需要交换
        c2 = left_top[0] * x_l_weight * y_l_weight + right_top[0] * x_h_weight * y_l_weight + left_bottom[0] * x_l_weight * y_h_weight + right_bottom[0] * x_h_weight * y_h_weight;
        c1 = left_top[1] * x_l_weight * y_l_weight + right_top[1] * x_h_weight * y_l_weight + left_bottom[1] * x_l_weight * y_h_weight + right_bottom[1] * x_h_weight * y_h_weight;
        c0 = left_top[2] * x_l_weight * y_l_weight + right_top[2] * x_h_weight * y_l_weight + left_bottom[2] * x_l_weight * y_h_weight + right_bottom[2] * x_h_weight * y_h_weight;
	}
    input_buff_device[position] = c0 / 255.0;
    input_buff_device[position + edge] = c1 / 255.0;
    input_buff_device[position + edge * 2] = c2 / 255.0;

	
	// float* dst_c0 = input_buff_device + position;
    // float* dst_c1 = input_buff_device + position + edge;
    // float* dst_c2 = input_buff_device + position + edge * 2;
    // *dst_c0 = c0 / 255.0;
    //  *dst_c1 = c1 / 255.0;
    //  *dst_c2 = c2 / 255.0;
}

void preprocess_by_cuda(cv::Mat& img, float* device_input, int dst_h, int dst_w, float* d2s_matrix, unsigned char* mask, int mask_h, int mask_w, float* roi, YOLO::TASK::TRT::TaskFlowTRTContext* ctx){
    // 将图片上传至显存
    if(ctx->img_data.len() == 0){
        int img_size = img.total() * img.elemSize();
        ctx->img_data.malloc(img_size);
        memcpy(ctx->img_data.cpu(), img.data, img_size);
        ctx->img_data.memcpy_to_gpu_sync();
    }
    
    TRACE("preprocess ---- update img", ctx);

    int src_w = img.cols;
    int src_h = img.rows;
    // 分配cuda线程
    int jobs = dst_h * dst_w;  // 每个像素给线程，负责完成该像素的生成、rgb拆解、归一化操作
    int threads = 256;
    int blocks = ceil(jobs / (float)threads);

    preprocess_kernel<<<blocks, threads, 0, ctx->stream>>>(ctx->img_data.gpu(), device_input, src_w, src_h, dst_w, dst_h, d2s_matrix, mask, mask_h, mask_w, roi, jobs);

    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    TRACE("preprocess ---- call kernel", ctx);
}