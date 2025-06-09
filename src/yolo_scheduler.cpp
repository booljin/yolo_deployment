#include "yolo_scheduler.h"
#include "trt_impl/model_trt.h"
#include "trt_impl/workspace_trt.h"
#include "trt_impl/taskflow_trt.h"

using namespace YOLO;

YoloScheduler::~YoloScheduler(){
    relase_all_workspaces();
}

std::shared_ptr<ModelAny> YoloScheduler::load_model(ModelType type, const std::string& model_file){
    if(_type != ModelType::MODEL_TYPE_UNKNOWN && _type != type){
        return nullptr; // 不支持的模型类型
    }
    if(type == ModelType::MODEL_TYPE_TRT){
        YOLO::TRT::Model model;
        bool ret = model.load(model_file);
        if(ret){
            _models.emplace_back(std::make_shared<YOLO::TRT::Model>(std::move(model)));
			_type = type;
            return _models.back();
        }
        return nullptr;
    }
    return nullptr;
}

WorkSpaceAny* YoloScheduler::get_new_workspace(){
    if(_type == ModelType::MODEL_TYPE_TRT){
        YOLO::TRT::WorkSpace* workspace = new YOLO::TRT::WorkSpace();
        if(workspace->create_by(_models)){
            _workspaces.emplace_back(workspace);
            return workspace;
        } else {
            delete workspace;
            return nullptr;
        }
    }
    return nullptr;
}

WorkSpaceAny* YoloScheduler::default_workspace(){
    if(_workspaces.size() > 0){
        return _workspaces[0];
    } else {
        return get_new_workspace();
    }
}

void YoloScheduler::relase_all_workspaces(){
    for(auto& workspace : _workspaces){
        delete workspace;
    }
}


bool YoloScheduler::binding_taskflow(){
    if(_taskflow){
        //  already binded
        return false;
    }
    if(_type == ModelType::MODEL_TYPE_TRT){
        _taskflow = std::make_shared<YOLO::TASK::TRT::TaskFlow>();
        if(!_taskflow->binding(_models)){
            // 不知道为什么，反正绑定失败了
            return false;
        }
        return true;
    }
    return false;
}

std::vector<YOLO::TASK::TaskResult> YoloScheduler::execute(cv::Mat& img, WorkSpaceAny* wsa){
    return _taskflow->execute(img, wsa);
}