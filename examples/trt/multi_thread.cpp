
#include <opencv2/opencv.hpp>

#include "yolo_trt.h"
#include <thread>

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

		scheduler.binding_taskflow();


		auto start = std::chrono::system_clock::now();
		std::vector<std::thread> threads;
		for (int i = 0; i < 10; i++) {
			threads.emplace_back(std::thread([&scheduler, &img]() {
				auto workspace = scheduler.get_new_workspace();
				scheduler.execute(img, workspace);
			}));
		}
		for (auto& t : threads) {
			t.join();
		}
		auto end = std::chrono::system_clock::now();
		std::cout << "total cost" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;


	}
	catch (YOLO::YoloException& e) {
		std::cout << e.what() << std::endl;
	}
	return 0;
}

