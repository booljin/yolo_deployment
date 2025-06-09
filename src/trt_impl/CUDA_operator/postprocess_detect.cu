#include <cuda_runtime_api.h>
#include <algorithm>
#include "trt_impl/yolo_trt_utils.h"
#include "trt_impl/taskflow_trt.h"


// NCA:batch|anchor|anchor_count [1,8400]


// NFB:batch|feature|box [1,37,8400]
// 这应该是yolo模型检测头默认输出形状
// 这种情况下，每个feature的信息并不连续，直觉上应该转置为NBF
// 但考虑到这样需要增加一倍的检测头显存占用，所以先写一个专用核函数，后面根据测试结果决定是否转置
static __global__ void decode_nfb_kernel(float* predict, int box_count, int class_count, float confidence_threshold, float* d2s_matrix, float* temparray, int ret_limit){
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
    float* dest = temparray + 1 + index * (YOLO::TASK::TRT::NUM_OF_BOX_ELEMENTS);
    *dest++ = left;         // left
    *dest++ = top;          // top
    *dest++ = right;            // rihgt
    *dest++ = bottom;            // bottom
    *dest++ = confidence;   // confidence
    *dest++ = label;        // class
    *dest++ = 1.0f;         // keep_flag
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

static void decode_boxes(float* predict, int box_count, int class_count, float confidence_threshold, float nms_threshold, float* d2s_matrix, float* temparray, int ret_limit, YOLO::TASK::TRT::TaskFlowTRTContext* ctx){
    int threads = std::min(256, box_count);
    int blocks = (box_count + threads - 1) / threads;
    decode_nfb_kernel<<<blocks, threads, 0, ctx->stream>>>(predict, box_count, class_count, confidence_threshold, d2s_matrix, temparray, ret_limit);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));

    threads = std::min(256, ret_limit);
    blocks = (ret_limit + threads - 1) / threads;
    nms_kernel<<<blocks, threads, 0, ctx->stream>>>(temparray, YOLO::TASK::TRT::NUM_OF_BOX_ELEMENTS, ret_limit, nms_threshold);
    CUDA_CHECK(cudaStreamSynchronize(ctx->stream));
}

void postprocess_detect_by_cuda(
        // 检测头相关
        float* predict, int box_count, int class_count,
        // 配置相关
        float confidence_threshold, float nms_threshold, int ret_limit,
        float* d2s_matrix,
        std::vector<YOLO::TASK::DetectResult>& output,
        YOLO::TASK::TRT::TaskFlowTRTContext* ctx)
{
    YOLO::TRT::CudaMemory<float> temparray;
    temparray.malloc(YOLO::TASK::TRT::NUM_OF_BOX_ELEMENTS * ret_limit + 1);
    // 解析检测头，将合适的box写入temparray，并进行nms操作。此时可以从temparray中提取所有有效的bbox
    decode_boxes(predict, box_count, class_count, confidence_threshold, nms_threshold, d2s_matrix, temparray.gpu(), ret_limit, ctx);
    // 将temparray从device拷贝到host，准备尽心下一步mask头解析
    temparray.memcpy_to_cpu_sync(ctx->stream);

    int count = std::min((int)temparray.cpu()[0], ret_limit);
    output.reserve(count);
    for(int i = 0; i < count; ++i){
        float* cur_item = temparray.cpu() + 1 + i * (YOLO::TASK::TRT::NUM_OF_BOX_ELEMENTS);
        if((int)(cur_item[6]) == 0){
            // 这个box被抑制了，不需要输出
            continue;
        }
        YOLO::TASK::DetectResult result;
        result.left = cur_item[0];
        result.top = cur_item[1];
        result.right = cur_item[2];
        result.bottom = cur_item[3];
        result.confidence = cur_item[4];
        result.label = (int)(cur_item[5]);

		output.emplace_back(result);        
    }
}