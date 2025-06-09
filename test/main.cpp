 #include "yolo_segmentation_trt.h"
 #include "yolo_classification_trt.h"
 #include "yolo_detection_trt.h"

#include <opencv2/opencv.hpp>

#include <cassert>
#include <thread>
#include <chrono>
#include <cuda_runtime_api.h>

 void classification_trt_test()
 {
 	YOLO::YoloClassificationTRT model;
 	model.load("D:/work/yolo1/yanye_fenlei/train2/weights/fenlei.engine");

 	cv::Mat img = cv::imread("D:/work/yolo1/yanye_fenlei/1/tcLeafImage_1_(工位3)_00005.bmp");
 	if (img.empty()) {
 		std::cerr << "Failed to load image." << std::endl;
 		return;
 	}

 	auto ret = model.detect(img);
 	std::cout << std::endl << std::endl;
	
 }

 void detection_trt_test()
 {
 	YOLO::YoloDetectionTRT model;
 	model.load("D:/work/yolo1/dabaoji_dect/detection.engine");

 	cv::Mat img = cv::imread("D:/work/yolo1/dabaoji_dect/c4b5e064a0ee6cb7a0a2192d8156401.bmp");
 	if (img.empty()) {
 		std::cerr << "Failed to load image." << std::endl;
 		return;
 	}

 	auto ret = model.detect(img);
 	std::cout << std::endl << std::endl;

 }

 void segmentation_trt_test()
 {
 	YOLO::YoloSegmentationTRT model;
 	model.load("D:/work/yolo1/thickness.engine");

 	cv::Mat img = cv::imread("D:/work/yolo1/t.bmp");
 	if (img.empty()) {
 		std::cerr << "Failed to load image." << std::endl;
 		return;
 	}

 	model.detect_cpu(img);
 	std::cout << std::endl << std::endl;

 	auto ret = model.detect(img);
 	model.draw_mask(img, ret);
 	std::cout << std::endl << std::endl;
 }

#include "yolo.h"



void test_new(){
	//cv::Mat img = cv::imread("D:/work/yolo1/yanye_fenlei/1/tcLeafImage_1_(工位3)_00005.bmp");
	//cv::Mat img = cv::imread("D:/work/yolo1/dabaoji_dect/c4b5e064a0ee6cb7a0a2192d8156401.bmp");
	cv::Mat img = cv::imread("D:/work/yolo1/t.bmp");

	YOLO::YoloScheduler scheduler;
	//scheduler.load_model(YOLO::ModelType::MODEL_TYPE_TRT, "D:/work/yolo1/yanye_fenlei/train2/weights/fenlei.engine");
	//scheduler.load_model(YOLO::ModelType::MODEL_TYPE_TRT, "D:/work/yolo1/dabaoji_dect/detection.engine");
	scheduler.load_model(YOLO::ModelType::MODEL_TYPE_TRT, "D:/work/yolo1/thickness.engine");

	auto ttt = scheduler.default_workspace();

	auto r1 = scheduler.binding_taskflow();

	auto start = std::chrono::system_clock::now();

	auto r2 = scheduler.execute(img, ttt);
	
	

	auto end = std::chrono::system_clock::now();
	std::cout << "total cost" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


	auto segment_ret = std::get_if<std::vector<YOLO::TASK::SegmentResult>>(&r2[0]);

	

	if(segment_ret){
		cv::imwrite("D:/mask_new.jpg", draw_mask(*segment_ret, img));
	}

	

	// auto t1 = YOLO::TRT::Model(std::move(model1));

	int i= 0;
	
}

int main(int argc, char const* argv[])
{
	
	try {
		throw YOLO::YoloException("hello");
	}
	catch (YOLO::YoloException& e) {
		std::cout << e.what() << std::endl;
	}

	int nGpuNumber = 0; //GPU数量
	cudaGetDeviceCount(&nGpuNumber);
	if (!(nGpuNumber > 0))
	{
		return false;
	}
	


	segmentation_trt_test();
	
	
	test_new();
	return 0;



	// YOLO::YoloSegment seg;
	// seg.load("D:/work/yolo1/thickness.engine");

	
	// cv::Mat img = cv::imread("D:/work/yolo1/t.bmp");
	//cv::Mat img = cv::imread("E:/work/print/flower/NewIPU2008/Bin/Debug/ipu4_20173-31264_11.bmp");
	//cv::Mat img = cv::imread("E:/work/print/flower/NewICW2008/IPU_Simulator/data/ipu3_20187-23226_4004.bmp");

	//float* device_input = nullptr;
	//preprocess_cuda(img, &device_input, 640, 640);

	// seg.detect_cpu(img);
	// std::cout << std::endl << std::endl;
	// seg.detect_gpu(img);

	//auto start = std::chrono::system_clock::now();
	//std::vector<std::thread> threads;
	//for (int i = 0; i < 4; i++) {
	//	threads.emplace_back(std::thread([&seg, &img]() {
	//		YOLO::WorkSpace ws = seg.getNewContext();
	//		seg.detect_gpu(img, &ws);
	//	}));
	//}
	//for (auto& t : threads) {
	//	t.join();
	//}
	//auto end = std::chrono::system_clock::now();
	//std::cout << "total cost" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	return 0;
}

