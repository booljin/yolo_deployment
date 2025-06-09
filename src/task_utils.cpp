#include "task_base.h"
#include <opencv2/opencv.hpp>
#include <sstream>

cv::Mat YOLO::TASK::draw_mask(const std::vector<YOLO::TASK::SegmentResult>& bboxes, cv::Mat& mask_img){
    for(auto& bbox: bboxes){
		cv::Mat mask;
		cv::resize(bbox.mask, mask, cv::Size(bbox.right - bbox.left, bbox.bottom - bbox.top), 0, 0, cv::INTER_NEAREST);
		cv::Mat roi = mask_img(cv::Rect(bbox.left, bbox.top, bbox.right - bbox.left, bbox.bottom - bbox.top));
        roi.setTo(cv::Scalar(0, 0, 255), mask);
    }
    return mask_img;
}

std::string dump_tracer(YOLO::TASK::TaskFlowContext* ctx){
    std::ostringstream ss;
    ss << std::endl << "  ------------------ trace ------------------" << std::endl;
    for (int i = 1; i < ctx->tracer.size(); i++) {
		ss << "\t" << ctx->tracer[i].first << " cost:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(ctx->tracer[i].second - ctx->tracer[i - 1].second).count() << " ms" << std::endl;
	}
	ss << "\t*** total cost:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(ctx->tracer[ctx->tracer.size() - 1].second - ctx->tracer[0].second).count() << " ms" << std::endl;
    ss << "  ------------------" << std::endl;
    return ss.str();
}