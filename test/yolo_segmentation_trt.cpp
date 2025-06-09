#include "yolo_segmentation_trt.h"
#include "trt_engine.h"
#include "yolo_struct_def.h"
#include "yolo_utils.h"

#include <algorithm>
#include <chrono>
#include <iostream>


#define TRACE(point)\
tracer.emplace_back(std::make_pair(#point, std::chrono::system_clock::now()));

// 导入关键处理函数
void preprocess_by_cpu(cv::Mat& img, float* blob, int dst_h, int dst_w);
void preprocess_by_cuda_t(cv::Mat& img, float* device_input, int dst_h, int dst_w, float* d2s_matrix, cudaStream_t& stream);
void postprocess_seg_by_cuda(
        // 检测头相关
        float* predict, int box_count, int class_count,
        // mask头相关
        float* mask_predict, int mask_width, int mask_height, int mask_dim,
        // 配置相关
        float confidence_threshold, float nms_threshold, float mask_threshold, int ret_limit,
        float* d2s_matrix, float* s2d_matrix,
        int input_w, int input_h,
        std::vector<YOLO::SegmentationResult>& output,
        cudaStream_t& stream);


using namespace YOLO;
YoloSegmentationTRT::YoloSegmentationTRT() : YoloBase(){

}

YoloSegmentationTRT::~YoloSegmentationTRT() {
}

const int RECT_LEN = 4; // 检测框的长度为4个float
bool YoloSegmentationTRT::load(const std::string& engine_file){
    if(!YoloBase::load(engine_file))
        return false;

	// 读取模型信息并初始化相关参数
	auto dims_0 = engine()->engine()->getTensorShape("output0");
	if (dims_0.nbDims == -1) {	// 无法读取输入参数
		engine()->destroy();;
		return false;
	}
    // dims_0.d[1] = RECT + class_count + weights
    _box_count = dims_0.d[2];
    
    auto dims_1 = engine()->engine()->getTensorShape("output1");
	if (dims_1.nbDims == -1) {	// 无法读取输入参数
		engine()->destroy();;
		return false;
    }
    _mask_dim = dims_1.d[1];
    _mask_h = dims_1.d[2];
    _mask_w = dims_1.d[3];
    _class_count = dims_0.d[1] -  _mask_dim - RECT_LEN;;

    _input_size = _batch_size * _input_c * _input_h * _input_w;
    _output0_size = _batch_size * dims_0.d[1] * _box_count;
    _output1_size = _batch_size * _mask_dim * _mask_h * _mask_w;

    return true;
}


//置信度阈值
static const double CONF_THRESHOLD = 0.3;
//nms阈值
static const float NMS_THRESHOLD = 0.5;
//mask阈值
static const float MASK_THRESHOLD = 0.5;
void YoloSegmentationTRT::post_processing_cpu(float* prob0, float* prob1, const ResizeInfo& resize_info, std::vector<std::pair<cv::Rect, cv::Mat>>& results, std::vector<std::pair<std::string, std::chrono::system_clock::time_point>>& tracer){
    int net_length = _class_count + _box_count + RECT_LEN;
    // x轴8400个box，y轴net_length
    cv::Mat prob0_mat = cv::Mat(net_length, _box_count, CV_32F, prob0);

    std::vector<int> class_ids;
    std::vector<float> scores;
    std::vector<cv::Rect> boxes;
	std::vector<int> box_ids;
    
	double max_score = 0.0;
    // decode boxes
    for(int b = 0; b < _box_count; ++b){
        cv::Mat score = prob0_mat(cv::Rect(b, 4, 1, 1)).clone();
        cv::Point class_id_point;
        double max_class_score;
        cv::minMaxLoc(score, 0, &max_class_score, 0, &class_id_point);
		if (max_class_score > max_score) max_score = max_class_score;
        if(max_class_score >= CONF_THRESHOLD){
            float x_in_ori = (prob0_mat.at<float>(0, b) - resize_info.real_x) / resize_info.scale;
            float y_in_ori = (prob0_mat.at<float>(1, b) - resize_info.real_y) / resize_info.scale;
            float w = prob0_mat.at<float>(2, b) / resize_info.scale;
            float h = prob0_mat.at<float>(3, b) / resize_info.scale;
            int left = std::max((int)(x_in_ori - 0.5 * w), 0);
            int top = std::max((int)(y_in_ori - 0.5 * h), 0);
            if(w <= 0  || h <= 0)
                continue;
            class_ids.emplace_back(class_id_point.y);
            scores.emplace_back((float)max_class_score);
            boxes.emplace_back(cv::Rect(left, top, w, h));
			box_ids.emplace_back(b);
        }
    }
	TRACE("post_processing - decode boxes");

    // nms
    std::vector<int> rows_after_nms;
    cv::dnn::NMSBoxes(boxes, scores, CONF_THRESHOLD, NMS_THRESHOLD, rows_after_nms);
	TRACE("post_processing - nms");

    // docode_mask
    cv::Mat mask_proposals;
    for(int row: rows_after_nms){
        mask_proposals.push_back(prob0_mat(cv::Rect(box_ids[row], RECT_LEN + _class_count, 1, _mask_dim)).t());
    }
    cv::Mat prob1_mat = cv::Mat(_mask_dim, _mask_h * _mask_w, CV_32F, prob1);
    cv::Mat mul_res = (mask_proposals * prob1_mat).t();
    cv::Mat masks = mul_res.reshape(rows_after_nms.size(), { _mask_h, _mask_w });

    std::vector<cv::Mat> mask_channels;
    cv::split(masks, mask_channels);
    
    int nms_idx = 0;
    for(auto& mask_: mask_channels){
        cv::Mat dest, mask;
        //sigmoid
        cv::exp(-mask_, dest);
        dest = 1.0 / (1.0 + dest);
        cv::Rect roi(int((float)resize_info.real_x / _input_w * _mask_w), int((float)resize_info.real_y / _input_h * _mask_h),
            int(_mask_w - resize_info.real_x / 2), int(_mask_h - resize_info.real_y / 2));
        dest = dest(roi);
        cv::resize(dest, mask, cv::Size(resize_info.ori_w, resize_info.ori_h), cv::INTER_NEAREST);
        cv::Rect temp_rect = boxes[rows_after_nms[nms_idx++]];
        mask = mask(temp_rect) > MASK_THRESHOLD;
		results.emplace_back(std::make_pair(temp_rect, mask));
    }
	TRACE("post_processing - decode mask");
}


void YoloSegmentationTRT::detect_cpu(cv::Mat& img, WorkSpace* work_space){
	if (!work_space)
		work_space = &_default_workspace;

	YoloBase::ResizeInfo resize_info;
	std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> tracer;
	cudaStream_t stream;
	CHECK(cudaStreamCreate(&stream));


	TRACE("begin");
	pre_resize_img(img, resize_info, tracer);
    TRACE("pre resize");

	CudaMemory<float> buffer[3];
	buffer[0].malloc(_input_size);
	buffer[1].malloc(_output0_size);
	buffer[2].malloc(_output1_size);

    preprocess_by_cpu(img, buffer[0].cpu(), _input_h, _input_w);
    TRACE("preprocess by cpu");
    
	buffer[0].memcpy_async(MemcpyHostToDevice, stream);
    
	work_space->context()->setTensorAddress("images", buffer[0].gpu());
	work_space->context()->setTensorAddress("output0", buffer[1].gpu());
	work_space->context()->setTensorAddress("output1", buffer[2].gpu());
	work_space->context()->enqueueV3(stream);

	buffer[1].memcpy_async(MemcpyDeviceToHost, stream);
	buffer[2].memcpy_async(MemcpyDeviceToHost, stream);
	cudaStreamSynchronize(stream);
	TRACE("inference");

	std::vector<std::pair<cv::Rect, cv::Mat>> result;
	post_processing_cpu(buffer[1].cpu(), buffer[2].cpu(), resize_info, result, tracer);


	cudaStreamDestroy(stream);
	// 绘制mask
	cv::Mat mask_img = cv::Mat(cv::Size(resize_info.ori_w, resize_info.ori_h), CV_8UC3, cv::Scalar(255, 255, 255));
	for (int i = 0; i < result.size(); ++i) {
		mask_img(result[i].first).setTo(cv::Scalar(0, 0, 255), result[i].second);
	}
	cv::imwrite("d:/mask.bmp", mask_img);

	for (int i = 1; i < tracer.size(); i++) {
		std::cout << tracer[i].first << " cost:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(tracer[i].second - tracer[i - 1].second).count() << " ms" << std::endl;
	}
	std::cout << "total cost:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(tracer[tracer.size() - 1].second - tracer[0].second).count() << " ms" << std::endl;
}


std::vector<SegmentationResult> YoloSegmentationTRT::detect(cv::Mat& img, WorkSpace* work_space){
	if (!work_space)
		work_space = &_default_workspace;
	std::vector<std::pair<std::string, std::chrono::system_clock::time_point>> tracer;
	TRACE("cuda begin");

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

    CudaMemory<float> buffer[3];
	buffer[0].malloc(_input_size);
	buffer[1].malloc(_output0_size);
    buffer[2].malloc(_output1_size);
	work_space->context()->setTensorAddress("images", buffer[0].gpu());
	work_space->context()->setTensorAddress("output0", buffer[1].gpu());
    work_space->context()->setTensorAddress("output1", buffer[2].gpu());

	preprocess_by_cuda_t(img, buffer[0].gpu(), _input_h, _input_w, d2s_matrix.cpu(), stream);
	TRACE("preprocess by cuda");

	work_space->context()->enqueueV3(stream);
    TRACE("inference");
    
    std::vector<SegmentationResult> output;
    postprocess_seg_by_cuda(buffer[1].gpu(), _box_count, _class_count,
        buffer[2].gpu(), _mask_w, _mask_h, _mask_dim,
        CONF_THRESHOLD, NMS_THRESHOLD, MASK_THRESHOLD, 1024,
        d2s_matrix.cpu(), s2d_matrix.cpu(),
        _input_w, _input_h, output, stream);

    cudaStreamSynchronize(stream);
	
    cudaStreamDestroy(stream);
    TRACE("postprocess by cuda");
    for (int i = 1; i < tracer.size(); i++) {
		std::cout << tracer[i].first << " cost:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(tracer[i].second - tracer[i - 1].second).count() << " ms" << std::endl;
	}
	std::cout << "total cost:\t" << std::chrono::duration_cast<std::chrono::milliseconds>(tracer[tracer.size() - 1].second - tracer[0].second).count() << " ms" << std::endl;

    return output;
    

	//cv::Mat mask_img = cv::Mat(cv::Size(img.cols, img.rows), CV_8UC3, cv::Scalar(255, 255, 255));
    // for(auto& bbox: output){
	// 	cv::Mat mask;
	// 	cv::resize(bbox.mask, mask, cv::Size(bbox.right - bbox.left, bbox.bottom - bbox.top), 0, 0, cv::INTER_NEAREST);
	// 	cv::Mat roi = mask_img(cv::Rect(bbox.left, bbox.top, bbox.right - bbox.left, bbox.bottom - bbox.top));
    //     roi.setTo(cv::Scalar(0, 0, 255),
    //         mask
    //     );
    // }
    // TRACE("draw mask");
	
	// cv::imwrite("d:/mask1.bmp", mask_img);
	// TRACE("imwrite mask");
}

void YoloSegmentationTRT::draw_mask(cv::Mat& mask_img, const std::vector<SegmentationResult>& bboxes){
    for(auto& bbox: bboxes){
		cv::Mat mask;
		cv::resize(bbox.mask, mask, cv::Size(bbox.right - bbox.left, bbox.bottom - bbox.top), 0, 0, cv::INTER_NEAREST);
		cv::Mat roi = mask_img(cv::Rect(bbox.left, bbox.top, bbox.right - bbox.left, bbox.bottom - bbox.top));
        roi.setTo(cv::Scalar(0, 0, 255),
            mask
        );
    }
    cv::imwrite("d:/mask1.bmp", mask_img);
}