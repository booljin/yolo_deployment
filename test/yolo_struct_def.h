#ifndef __YOLO_STRUCT_DEF_H__
#define __YOLO_STRUCT_DEF_H__
#include <opencv2/opencv.hpp>

namespace YOLO {

	struct BoundingBox {
		float left;
		float top;
		float right;
		float bottom;
		float confidence;
		int label;
		cv::Mat mask;
	};

	struct SegmentationResult{
		float left;
		float top;
		float right;
		float bottom;
		float confidence;
		int label;
		cv::Mat mask;
	};

	struct ClassificationResult {
		int id;          // 分类ID
		float confidence; // 分类置信度
	};

	struct DetectionResult {
		float left;
		float top;
		float right;
		float bottom;
		float confidence;
		int label;
	};

}
#endif // __YOLO_STRUCT_DEF_H__