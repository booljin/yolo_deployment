#include <cuda_runtime_api.h>
#include <algorithm>
#include "trt_impl/yolo_trt_utils.h"
#include "trt_impl/taskflow_trt.h"


// NCA:batch|anchor|anchor_count [1,8400]


// NFB:batch|feature|box [1,37,8400]
// 这应该是yolo模型检测头默认输出形状
// 这种情况下，每个feature的信息并不连续，直觉上应该转置为NBF
// 但考虑到这样需要增加一倍的检测头显存占用，所以先写一个专用核函数，后面根据测试结果决定是否转置
static __global__ void decode_nfb_kernel(float* predict, int box_count, int class_count, float confidence_threshold, float* d2s_matrix, int mask_dim, float* temparray, int ret_limit){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    if(position >= box_count) return;
    
    // 计算每个anchor的置信度
    float* item = predict + (box_count * 4) + position;
    float confidence = *item;
    int label = 0;
    for(int i = 1; i < class_count; ++i){
        item += box_count;
        if(*item > confidence){
            confidence = *item;
            label = i;
        }
    }
    if(confidence < confidence_threshold) return;

    int index = atomicAdd(temparray, 1);
    if(index >= ret_limit) return;

    // 计算每个anchor的坐标
    // 1,获取推理结果
    float cx = predict[position];
    float cy = predict[position + box_count];
    float w = predict[position + box_count * 2];
    float h = predict[position + box_count * 3];
    // 2,计算rect坐标
    float left = cx - 0.5f * w;
    float top = cy - 0.5f * h;
    float right = cx + 0.5f * w;
    float bottom = cy + 0.5f * h;
    // 3,仿射变换回原图坐标
    left = left * d2s_matrix[0] + d2s_matrix[2];
    top = top * d2s_matrix[4] + d2s_matrix[5];
    right = right * d2s_matrix[0] + d2s_matrix[2];
    bottom = bottom * d2s_matrix[4] + d2s_matrix[5];
    // 4,将结果写入temparray
    float* dest = temparray + 1 + index * (YOLO::TASK::TRT::NUM_OF_BOX_ELEMENTS + mask_dim);
    *dest++ = left;         // left
    *dest++ = top;          // top
    *dest++ = right;            // rihgt
    *dest++ = bottom;            // bottom
    *dest++ = confidence;   // confidence
    *dest++ = label;        // class
    *dest++ = 1.0f;         // keep_flag
    for(int i = 0; i < mask_dim; ++i){
        // 将mask的权重写入temparray
        *dest++ = predict[position + box_count * (5 + i)];
    }
}

static __device__ float box_iou(float left_a, float top_a, float right_a, float bottom_a,
                                float left_b, float top_b, float right_b, float bottom_b){
    // 计算两个box的交集
    float left = left_a > left_b ? left_a : left_b;
    float top = top_a > top_b ? top_a : top_b;
    float right = right_a < right_b ? right_a : right_b;
    float bottom = bottom_a < bottom_b ? bottom_b : bottom_a;
    
    if(left >= right || top >= bottom) return 0.0f; // 没有交集

    // 计算交集面积
    float intersection_area = (right - left) * (bottom - top);
    // 计算并集面积
    float union_area = (right_a - left_a) * (bottom_a - top_a) + (right_b - left_b) * (bottom_b - top_b) - intersection_area;
    if(union_area <= 0.0f) return 0.0f; // 防止除以零

    return intersection_area / union_area;
}

static __global__ void nms_kernel(float* temparray, int bbox_len, int ret_limit, float nms_threshold){
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    int count = (int)temparray[0];
    count = count < ret_limit ? count : ret_limit;
    if(position >= count) return;

    float* cur_item = temparray + 1 + position * bbox_len;
    for(int i = 0; i < count; ++i){
        if(i == position) continue;
        float* other_item = temparray + 1 + i * bbox_len;
        if(other_item[5] != cur_item[5]) continue; // 只比较同类
        
        float delta_confidence = other_item[4] - cur_item[4];
        if(delta_confidence < 0.0f){
            // 其他格子比这个格子置信度低，不用处理，它在自己的核函数里会被抑制
            continue;
        } else if(delta_confidence < 1e-6f) {
            // 两个格子置信度一样
            // 倾向于保留前面的格子
            // 每个核函数都会从头到尾搜索所有格子，那么只要发现前面有重合的格子，就抑制自己
            //   自己身后的格子不用管，因为他们会因为我的存在而自己抑制自己
            if(i > position) continue;
        }
        {
            float iou = box_iou(
                cur_item[0], cur_item[1], cur_item[2], cur_item[3],
                other_item[0], other_item[1], other_item[2], other_item[3]
            );
            if(iou > nms_threshold){
                cur_item[6] = 0.0f; // 抑制自己
                return;
            }
        }
    }
}

static void decode_boxes(float* predict, int box_count, int class_count, float confidence_threshold, float nms_threshold, float* d2s_matrix, float* temparray, int ret_limit, int mask_dim, YOLO::TASK::TRT::TaskFlowTRTContext* ctx){
    int threads = std::min(256, box_count);
    int blocks = (box_count + threads - 1) / threads;
    decode_nfb_kernel<<<blocks, threads, 0, ctx->stream>>>(predict, box_count, class_count, confidence_threshold, d2s_matrix, mask_dim, temparray, ret_limit);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    threads = std::min(256, ret_limit);
    blocks = (ret_limit + threads - 1) / threads;
    nms_kernel<<<blocks, threads, 0, ctx->stream>>>(temparray, YOLO::TASK::TRT::NUM_OF_BOX_ELEMENTS + mask_dim, ret_limit, nms_threshold);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

static __global__ void decode_mask_kernel(
        float* mask_predict, int mask_width, int mask_height, int mask_dim, float mask_threshold,
        float left, float top, float* mask_weights,
        unsigned char* mask_out, int out_width, int out_height)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if(dx >= out_width || dy >= out_height) return;
    int sx = dx + left;
    int sy = dy + top;
    if(sx < 0 || sx >= mask_width || sy < 0 || sy >= mask_height){
        mask_out[dy * out_width + dx] = 0; // 超出范围,权重默认为0
        return;
    }
    float cumprod = 0;
    for(int c = 0; c < mask_dim; ++c){
        float value = mask_predict[c * mask_height * mask_width + sy * mask_width + sx];
        cumprod += value * mask_weights[c];
    }
    float alpha = 1.0f / (1.0f + exp(-cumprod)); // sigmoid
    // 这里假设mask_out是一个灰度图像，0表示无物体，255表示有物体
    if(alpha < mask_threshold){
        mask_out[dy * out_width + dx] = 0;      // 小于阈值，权重为0
    } else {
        mask_out[dy * out_width + dx] = 255;    // 大于阈值，权重为255
    }
}

static void decode_mask(
        float* mask_predict, int mask_width, int mask_height, int mask_dim, float mask_threshold,
        float left, float top, float* mask_weights,
        unsigned char* mask_out, int out_width, int out_height,
        YOLO::TASK::TRT::TaskFlowTRTContext* ctx)
{
    dim3 block(32, 32);
    dim3 grid(out_width + 31 / 32, out_height + 31 / 32);
    decode_mask_kernel<<<grid, block, 0, ctx->stream>>>(mask_predict, mask_width, mask_height, mask_dim, mask_threshold, left, top, mask_weights, mask_out, out_width, out_height);
}


struct bbox_buffer{
    unsigned char* data;
    unsigned char* data_device;
    int width;
    int height;
};

void postprocess_segment_by_cuda(
        // 检测头相关
        float* predict, int box_count, int class_count,
        // mask头相关
        float* mask_predict, int mask_w, int mask_h, int mask_dim,
        // 配置相关
        float confidence_threshold, float nms_threshold, float mask_threshold, int ret_limit,
        float* d2s_matrix, float* s2d_matrix,
        int input_w, int input_h,
        std::vector<YOLO::TASK::SegmentResult>& output,
        YOLO::TASK::TRT::TaskFlowTRTContext* ctx)
{
    YOLO::TRT::CudaMemory<float> temparray;
    temparray.malloc((YOLO::TASK::TRT::NUM_OF_BOX_ELEMENTS + mask_dim) * ret_limit + 1);
    // 解析检测头，将合适的box写入temparray，并进行nms操作。此时可以从temparray中提取所有有效的bbox
    decode_boxes(predict, box_count, class_count, confidence_threshold, nms_threshold, d2s_matrix, temparray.gpu(), ret_limit, mask_dim, ctx);
    TRACE("postprocess --- decode_boxes", ctx);
    // 将temparray从device拷贝到host，准备尽心下一步mask头解析
    temparray.memcpy_to_cpu_sync(ctx->stream);
    TRACE("postprocess --- download predict", ctx);
    
    int count = std::min((int)temparray.cpu()[0], ret_limit);
    output.reserve(count);
    std::vector<bbox_buffer> temp_buffer;
    temp_buffer.reserve(count);
    for(int i = 0; i < count; ++i){
        float* cur_item = temparray.cpu() + 1 + i * (YOLO::TASK::TRT::NUM_OF_BOX_ELEMENTS + mask_dim);
        if((int)(cur_item[6]) == 0){
            // 这个box被抑制了，不需要输出
            continue;
        }
        YOLO::TASK::SegmentResult box;
        box.left = cur_item[0];
        box.top = cur_item[1];
        box.right = cur_item[2];
        box.bottom = cur_item[3];
        box.confidence = cur_item[4];
        box.label = (int)(cur_item[5]);

        float left_t = box.left * s2d_matrix[0] + s2d_matrix[2];
        float top_t = box.top * s2d_matrix[4] + s2d_matrix[5];
        float right_t = box.right * s2d_matrix[0] + s2d_matrix[2];
        float bottom_t = box.bottom * s2d_matrix[4] + s2d_matrix[5];
        float box_w = right_t - left_t;
        float box_h = bottom_t - top_t;
        float scale_x = mask_w / (float)input_w;
        float scale_y = mask_h / (float)input_h;
        int mask_out_w = (int)(box_w * scale_x + 0.5f);
        int mask_out_h = (int)(box_h * scale_y + 0.5f);
    
        if(mask_out_w > 0 && mask_out_h > 0){
            temp_buffer.emplace_back(bbox_buffer{nullptr, nullptr, mask_out_w, mask_out_h});
            bbox_buffer& box_buf = temp_buffer.back();
            CUDA_CHECK(cudaMallocHost((void**)&box_buf.data, mask_out_w * mask_out_h));
            CUDA_CHECK(cudaMalloc((void**)&box_buf.data_device, mask_out_w * mask_out_h));
            decode_mask(mask_predict, mask_w, mask_h, mask_dim, mask_threshold,
                    left_t * scale_x, top_t * scale_y, cur_item + 7, box_buf.data_device, mask_out_w, mask_out_h, ctx);
            CUDA_CHECK(cudaMemcpyAsync(box_buf.data, box_buf.data_device, mask_out_w * mask_out_h, cudaMemcpyDeviceToHost, ctx->stream));
			output.emplace_back(box);
        }
        
    }

    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
    TRACE("postprocess --- decode mask", ctx);

    for(int i = 0; i < (int)temp_buffer.size(); ++i){
        bbox_buffer& box = temp_buffer[i];
        if(box.data != nullptr){
            output[i].mask = cv::Mat(box.height, box.width, CV_8UC1, box.data).clone();
            CUDA_CHECK(cudaFreeHost(box.data));
            CUDA_CHECK(cudaFree(box.data_device));
        }
    }
    TRACE("postprocess --- clear tempbuffer", ctx);
}