#ifndef __YOLO_SEGMENTATION_TRT_H__
#define __YOLO_SEGMENTATION_TRT_H__

#include "yolo_base.h"
#include "yolo_struct_def.h"
#include <chrono>

namespace YOLO{

//struct SegmentResult{
//    int id;
//    float score;
//    int target_flag;
//    cv::Rect box;
//    cv::Mat box_mask;
//    std::vector<std::vector<cv::Point>> contours;
//};

class YoloSegmentationTRT : public YoloBase{

public:
    YoloSegmentationTRT();
    YoloSegmentationTRT(const YoloSegmentationTRT&) = delete;
    YoloSegmentationTRT operator=(const YoloSegmentationTRT&) = delete;
    virtual ~YoloSegmentationTRT() override;

    bool load(const std::string& engine_file);

public:
    void detect_cpu(cv::Mat& img, WorkSpace* work_space = nullptr);
    std::vector<SegmentationResult> detect(cv::Mat& img, WorkSpace* work_space = nullptr);

    void draw_mask(cv::Mat& mask_img, const std::vector<SegmentationResult>& bboxes);

private:
    void post_processing_cpu(float* prob0, float* prob1, const ResizeInfo& resize_info, std::vector<std::pair<cv::Rect, cv::Mat>>& results);


private:
    int _class_count;
    int _box_count;

    int _mask_dim;
    int _mask_w;
    int _mask_h;

    int _input_size;
    int _output0_size;
    int _output1_size;
};


}

#endif //__YOLO_SEGMENTATION_TRT_H__