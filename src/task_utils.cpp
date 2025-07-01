#include "task_base.h"
#include <opencv2/opencv.hpp>
#include <sstream>

cv::Mat YOLO::TASK::draw_mask(const YOLO::TASK::SegmentResult& sr, cv::Mat& mask_img, bool origin){
	cv::Mat combin_mask = sr.masks[0].second.clone();
	for (int i = 1; i < sr.masks.size(); ++i) {
		combin_mask.setTo(sr.masks[i].second, sr.masks[i].second);
	}
	cv::Mat ret_img = mask_img;
	if(!origin)
		ret_img = cv::Mat(mask_img.rows, mask_img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
	float scale = std::min(combin_mask.rows / (float)mask_img.rows, combin_mask.cols / (float)mask_img.cols);
	
	cv::Mat combin_mask_n;
	cv::resize(combin_mask(cv::Rect(
			(combin_mask.cols - scale * mask_img.cols) / 2,
			(combin_mask.rows - scale * mask_img.rows) / 2,
			scale * mask_img.cols,
			scale * mask_img.rows
		)),
		combin_mask_n, cv::Size(mask_img.cols, mask_img.rows), 0, 0, cv::INTER_NEAREST
	);
	ret_img.setTo(cv::Scalar(0, 0, 255), combin_mask_n);

    return ret_img;
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