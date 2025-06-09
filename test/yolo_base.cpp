#include "yolo_base.h"
#include "trt_engine.h"

using namespace YOLO;
YoloBase::YoloBase() {
    _engine = new TrtEngine();
}

YoloBase::~YoloBase() {
    delete _engine;
}

bool YoloBase::load(const std::string& engine_file){
    //TODO: 理论上需要提供重新加载其他模型的功能
    if(_default_workspace.context()!= nullptr)
        return true;
    if(!_engine->load(engine_file)){
        return false;
    }

	// 读取模型信息并初始化相关参数
	auto dims = _engine->engine()->getTensorShape("images");
	if (dims.nbDims == -1) {	// 无法读取输入参数
		_engine->destroy();
		return false;
	}
    _batch_size = dims.d[0];
    _input_c = dims.d[1];
	_input_h = dims.d[2];
	_input_w = dims.d[3];

	_default_workspace.setContext(_engine->get_context());
    if(_default_workspace.context() == nullptr)
        return false;
    return true;
}

void YoloBase::pre_resize_img(const cv::Mat& input, ResizeInfo& resize_info, std::vector<std::pair<std::string, std::chrono::system_clock::time_point>>& tracer){
    float w_scale = _input_w / (float)input.cols;
    float h_scale = _input_h / (float)input.rows;
    // 锁定缩放比，确保长边能够缩放到输入尺寸
    if(w_scale < h_scale){
        resize_info.real_w = _input_w;
        resize_info.real_h = w_scale * input.rows;
        resize_info.real_x = 0;
        resize_info.real_y = (_input_h - resize_info.real_h) / 2;
        resize_info.scale = w_scale;
    } else {
        resize_info.real_w = h_scale * input.cols;
        resize_info.real_h = _input_h;
        resize_info.real_x = (_input_w - resize_info.real_w) / 2;
        resize_info.real_y = 0;
        resize_info.scale = h_scale;
    }
    resize_info.ori_w = input.cols;
    resize_info.ori_h = input.rows;
}

WorkSpace YoloBase::getNewContext() {
	auto ctx = _engine->get_context();
	return WorkSpace{ctx};
}