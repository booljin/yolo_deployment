
#include <opencv2/opencv.hpp>

#include "yolo_trt.h"

int main(int argc, char const* argv[])
{
	//onnx_runtime_test();
	try {
		cv::Mat img = cv::imread("D:/work/yolo1/seg2seg/ori.bmp");

		YOLO::YoloScheduler<YOLO::TRT> scheduler;
		scheduler.load_model("D:/work/yolo1/seg2seg/cdw.onnx", "cdw");
		auto m2 = scheduler.load_model("D:/work/yolo1/seg2seg/jdw.onnx", "jdw", "cdw");
		
		m2->set_confidence_threshold(0.4);
		m2->set_nms_threshold(0.3);

		auto workspace = scheduler.default_workspace();
		scheduler.binding_taskflow();

		std::vector<YOLO::TASK::TaskResult> results;

		auto start = std::chrono::system_clock::now();
		for(int i = 0; i < 10; i++)
			results = scheduler.execute(img, workspace);
		auto end = std::chrono::system_clock::now();
		std::cout << "avg cost" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 10 << std::endl;

		auto segment_ret = std::get_if<YOLO::TASK::SegmentResult>(&results.back());

		if (segment_ret && segment_ret->masks.size() > 0&& segment_ret->bboxes.size() > 0) {
			cv::imwrite("D:/work/yolo1/seg2seg/output.jpg", draw_mask(*segment_ret, img, true));
		}
		else {
			std::cout << "found nothing" << std::endl;
		}

	}
	catch (YOLO::YoloException& e) {
		std::cout << e.what() << std::endl;
	}
	return 0;
}

