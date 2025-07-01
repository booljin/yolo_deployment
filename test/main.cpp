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



#include "yolo_trt.h"

void test_new(){
	//cv::Mat img = cv::imread("D:/work/yolo1/yanye_fenlei/1/tcLeafImage_1_(工位3)_00005.bmp");
	//cv::Mat img = cv::imread("D:/work/yolo1/dabaoji_dect/c4b5e064a0ee6cb7a0a2192d8156401.bmp");
	cv::Mat img = cv::imread("D:/work/yolo1/t.bmp");

	YOLO::YoloScheduler<YOLO::TRT> scheduler;
	//scheduler.load_model(YOLO::ModelType::MODEL_TYPE_TRT, "D:/work/yolo1/yanye_fenlei/train2/weights/fenlei.engine");
	//scheduler.load_model(YOLO::ModelType::MODEL_TYPE_TRT, "D:/work/yolo1/dabaoji_dect/detection.engine");
	scheduler.load_model("D:/work/yolo1/thickness.engine", "thickness");

	auto ttt = scheduler.default_workspace();

	auto r1 = scheduler.binding_taskflow();

	auto start = std::chrono::system_clock::now();

	auto r2 = scheduler.execute(img, ttt);
	
	

	auto end = std::chrono::system_clock::now();
	std::cout << "total cost" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();


	auto segment_ret = std::get_if<YOLO::TASK::SegmentResult>(&r2[0]);

	

	if(segment_ret){
		cv::imwrite("D:/mask_new.jpg", draw_mask(*segment_ret, img));
	}

	

	// auto t1 = YOLO::TRT::Model(std::move(model1));

	int i= 0;
	
}

//带增强的任务流
void test_new_new() {
	YOLO::YoloScheduler<YOLO::TRT> scheduler;
	//scheduler.load_model("D:/work/yolo1/yanye_fenlei/train2/weights/fenlei.engine", "L0-0");
	//scheduler.load_model("D:/work/yolo1/yanye_fenlei/train2/weights/fenlei.engine", "L1-0", "L0-0", 0);
	//scheduler.load_model("D:/work/yolo1/yanye_fenlei/train2/weights/fenlei.engine", "L1-1", "L0-0", 1);
	//scheduler.load_model("D:/work/yolo1/yanye_fenlei/train2/weights/fenlei.engine", "L1-2", "L0-0", 2);
	////scheduler.load_model("D:/work/yolo1/yanye_fenlei/train2/weights/fenlei.engine", "L1-3", "L0-0", 3);
	//scheduler.load_model("D:/work/yolo1/yanye_fenlei/train2/weights/fenlei.engine", "L2-1", "L1-1", 1);
	//scheduler.load_model("D:/work/yolo1/yanye_fenlei/train2/weights/fenlei.engine", "L3-1", "L2-1", 1);
	
	scheduler.load_model("D:/work/yolo1/seg2seg/cdw.onnx", "cdw");
	auto m2 = scheduler.load_model("D:/work/yolo1/seg2seg/jdw.onnx", "jdw", "cdw");
	//auto m2 = scheduler.load_model("D:/work/yolo1/seg2seg/jdw.onnx", "jdw");
	
	m2->set_confidence_threshold(0.4);
	m2->set_nms_threshold(0.3);

	auto ttt = scheduler.default_workspace();

	auto r1 = scheduler.binding_taskflow();
	//cv::Mat img = cv::imread("D:/work/yolo1/cududingwei/img/20250514124306457-工位3-ID48-序号124-粗度-12.30mm.bmp");
	//cv::Mat img = cv::imread("D:/work/yolo1/jdw/image/20250508124502419-工位3-ID5-序号87-粗度-6.69mm.bmp");
	cv::Mat img = cv::imread("D:/work/yolo1/seg2seg/ori.bmp");

	std::vector<YOLO::TASK::TaskResult> r2;

	auto start = std::chrono::system_clock::now();
	for(int i = 0; i < 10; i++)
		r2 = scheduler.execute(img, ttt);
	auto end = std::chrono::system_clock::now();
	std::cout << "avg cost" << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 10 << std::endl;

	auto segment_ret = std::get_if<YOLO::TASK::SegmentResult>(&r2.back());

	if (segment_ret && segment_ret->masks.size() > 0&& segment_ret->bboxes.size() > 0) {
		cv::imwrite("D:/work/yolo1/seg2seg/output.jpg", draw_mask(*segment_ret, img, true));
	}
	else {
		std::cout << "found nothing" << std::endl;
	}

	int i = 0;
}

#include <onnxruntime_cxx_api.h>
void preprocess_by_cpu(cv::Mat& img, float* blob, int dst_h, int dst_w);
void onnx_runtime_test() {
//https://blog.csdn.net/yangyu0515/article/details/142057357
	Ort::SessionOptions session_options;
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "best.onnx");
	//OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0);
	Ort::Session sess(env, L"D:/work/yolo1/yanye_fenlei/train2/weights/best.onnx", session_options);
	int input_nodes_num = sess.GetInputCount();
	int output_nums = sess.GetOutputCount();

	float* blob = new float[640 * 640 * sizeof(float)];
	cv::Mat img = cv::imread("D:/work/yolo1/yanye_fenlei/1/tcLeafImage_1_(工位3)_00005.bmp");
	preprocess_by_cpu(img, blob, 640, 640);

	std::array<int64_t, 4> input_shape_info{ 1, 3, 640, 640 };

	auto allocator_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::Value input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, blob, 640*640*3, input_shape_info.data(), input_shape_info.size());
	// 输入一个数据
	const std::array<const char*, 1> inputNames = { "images"};
	// 输出多个数据
	const std::array<const char*, 1> outNames = { "output0" };
	std::vector<Ort::Value> ort_outputs;
	try {
		ort_outputs = sess.Run(Ort::RunOptions{ nullptr }, inputNames.data(), &input_tensor_, inputNames.size(), outNames.data(), outNames.size());
	}
	catch (std::exception e) {
		std::cout << e.what() << std::endl;
	}

	// 选择最后一个输出作为最终的mask
	const float* output0 = ort_outputs[0].GetTensorMutableData<float>();
	auto outShape = ort_outputs[0].GetTensorTypeAndShapeInfo().GetShape();
	
	float first_confidence = 0.0f;
	float second_confidence = 0.0f;
	int first_label = -1;
	int second_label = -1;
	for (int i = 0; i < outShape[1]; ++i) {
		if (output0[i] > first_confidence) {
			second_confidence = first_confidence;
			second_label = first_label;
			first_confidence = output0[i];
			first_label = i;
		}
		else if (output0[i] > second_confidence) {
			second_confidence = output0[i];
			second_label = i;
		}
	}

	int i = 0;

}

int main(int argc, char const* argv[])
{
	//onnx_runtime_test();
	try {
		test_new_new();
	}
	catch (YOLO::YoloException& e) {
		std::cout << e.what() << std::endl;
	}
	return 0;
	struct TTT {
		YOLO::UTILS::TRT::CudaMemory<unsigned char> m;
		int l;
		int t;
		int w;
		int h;
		TTT(int i1,int i2, int i3, int i4) {
			l = i1;
			t = i2;
			w = i3;
			h = i4;
			m.malloc(w*h);
		}
	};
	std::vector<TTT> vt;
	vt.reserve(67);
	vt.emplace_back(1,2,3,4);
	TTT& last = vt.back();


	int nGpuNumber = 0; //GPU数量
	cudaGetDeviceCount(&nGpuNumber);
	if (!(nGpuNumber > 0))
	{
		return false;
	}
	


	//classification_trt_test();
	
	
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

