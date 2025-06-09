#include "yolo_classification_trt.h"
#include "trt_engine.h"
#include "yolo_struct_def.h"
#include "yolo_utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>


#define TRACE(point)\
tracer.emplace_back(std::make_pair(#point, std::chrono::system_clock::now()));

// 导入关键处理函数
void preprocess_by_cuda_t(cv::Mat& img, float* device_input, int dst_h, int dst_w, float* d2s_matrix, cudaStream_t& stream);
void postprocess_class_by_cuda(
	// 检测头相关
	float* predict, int class_count,
	std::vector<YOLO::ClassificationResult>& output,
	cudaStream_t& stream);

using namespace YOLO;
YoloClassificationTRT::YoloClassificationTRT() : YoloBase(){
}

YoloClassificationTRT::~YoloClassificationTRT() {
}

bool YoloClassificationTRT::load(const std::string& engine_file){
    if(!YoloBase::load(engine_file))
        return false;

	// 读取模型信息并初始化相关参数
	auto dims = engine()->engine()->getTensorShape("output0");
	if (dims.nbDims == -1) {	// 无法读取输入参数
		engine()->destroy();;
		return false;
	}

    _class_count = dims.d[1];

    _input_size = _batch_size * _input_c * _input_h * _input_w;
    _output0_size = _batch_size * _class_count;

    return true;
}


//置信度阈值
static const double CONF_THRESHOLD = 0.3;
std::vector<ClassificationResult> YoloClassificationTRT::detect(cv::Mat& img, WorkSpace* work_space){
	if (!work_space)
		work_space = &_default_workspace;
	std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> tracer;
	TRACE("classification_trt begin");

    // 计算仿射矩阵
    /*
    令缩放比例为s,原坐标表示为:{x, y},缩放后坐标变为{x', y'}，即 {x', y'} = {x*s, y*s}
    在缩放的同时，需要进行位移操作，假设x需要修正a，y需要修正b， 即 {x', y'} = {x*s + a, y*s + b} 
    此操作可以用仿射矩阵来表示：
    |x'|   |s 0 a|   |x|
    |y'| = |0 s b| * |y|
    |1 |   |0 0 1|   |1|  （齐次项）
    */
    int src_w = img.cols;
    int src_h = img.rows;
    float scale = std::min((float)_input_h / (float)src_h, (float)_input_w / (float)src_w);
    // 这个是从原图坐标到目标图的缩放矩阵A
    float s2d_matrix_t[6] = {scale, 0, (_input_w - (scale * src_w)) / 2, 0, scale, (_input_h - (scale * src_h)) / 2};
    cv::Mat s2d_mat(2, 3, CV_32F, s2d_matrix_t);
    cv::Mat d2s_mat(2, 3, CV_32F);
    // 获取逆变换矩阵，即目标图坐标到原图的变换矩阵A'
    cv::invertAffineTransform(s2d_mat, d2s_mat);
    CudaMemory<float> d2s_matrix, s2d_matrix;
    d2s_matrix.malloc(6);
    s2d_matrix.malloc(6);
    memcpy(d2s_matrix.cpu(), d2s_mat.ptr<float>(0), sizeof(float) * d2s_matrix.len());
    memcpy(s2d_matrix.cpu(), s2d_matrix_t, sizeof(float) * s2d_matrix.len());
	TRACE("create affine matrix");

	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));


    CudaMemory<float> buffer[2];
	buffer[0].malloc(_input_size);
	buffer[1].malloc(_output0_size);
	work_space->context()->setTensorAddress("images", buffer[0].gpu());
	work_space->context()->setTensorAddress("output0", buffer[1].gpu());
	

	preprocess_by_cuda_t(img, buffer[0].gpu(), _input_h, _input_w, d2s_matrix.cpu(), stream);
	TRACE("preprocess by cuda");

	work_space->context()->enqueueV3(stream);
    TRACE("inference");
    
    buffer[1].memcpy_to_cpu_sync(stream);
    
    std::vector<ClassificationResult> output;
    postprocess_class_by_cuda(buffer[1].cpu(), _class_count, output, stream);

	TRACE("postprocess by cuda");
    cudaStreamDestroy(stream);
    
	for (int i = 1; i < tracer.size(); i++) {
		std::cout << tracer[i].first << " cost:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(tracer[i].second - tracer[i - 1].second).count() << " ms" << std::endl;
	}
	std::cout << "classification_trt total cost:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(tracer[tracer.size() - 1].second - tracer[0].second).count() << " ms" << std::endl;
    return output;
}