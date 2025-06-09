#include <cuda_runtime_api.h>
#include <algorithm>
#include "yolo_utils.h"
#include "yolo_struct_def.h"


// @brief 分类的后处理非常简单，暂时没用cuda加速
void postprocess_class_by_cuda(
        // 检测头相关
        float* predict, int class_count,
        std::vector<YOLO::ClassificationResult>& output,
        cudaStream_t& stream)
{
    float first_confidence = 0.0f;
    float second_confidence = 0.0f;
    int first_label = -1;
    int second_label = -1;
    for(int i = 0; i < class_count; ++i){
        if(predict[i] > first_confidence){
            second_confidence = first_confidence;
            second_label = first_label;
            first_confidence = predict[i];
            first_label = i;
        } else if(predict[i] > second_confidence){
            second_confidence = predict[i];
            second_label = i;
        }
    }
    output.emplace_back(YOLO::ClassificationResult{first_label, first_confidence});
    output.emplace_back(YOLO::ClassificationResult{second_label, second_confidence});
    return;
}