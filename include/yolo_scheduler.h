#ifndef __YOLO_SCHEDULER_H__
#define __YOLO_SCHEDULER_H__

#include <vector>
#include <memory>
#include <tuple>
#include "yolo_defines.h"
#include "model_base.h"
#include "workspace_base.h"
#include "task_base.h"


namespace YOLO{

template<typename Backend>
struct BackendTraits;

template <typename Backend>
class YoloScheduler {
    public:
    // 通过 Traits 获取后端专属类型
    using ModelT = typename YOLO::BackendTraits<Backend>::Model;
    using TaskFlowT = typename YOLO::BackendTraits<Backend>::TaskFlow;
    using WorkSpaceT = typename YOLO::BackendTraits<Backend>::WorkSpace;
public:
    YoloScheduler() = default;
    ~YoloScheduler() {
        relase_all_workspaces();
    }
public:
// model管理相关
    // @brief 加载模型
    // @param model_path    模型路径
    // @param alias         模型别名，用于串联模型
    // @param parent        上级模型别名
    // @param branch        上级模型如果是分类，则指示其为哪个类别的后续
    ModelT* load_model(const std::string& model_file, const std::string& alias, const std::string& parent = "", unsigned int branch = 0){
        auto model = std::make_unique<ModelT>(alias);
        bool ret = model->load(model_file);
        if(ret){
            if(_type == ModelType::MODEL_TYPE_UNKNOWN){
                _type = model->type();
            } else if(_type != model->type()){
                throw YOLO::YoloException("model type error");
            }
        } else {
            throw YOLO::YoloException("load model error");
        }

        bool found_parent = false;
        for(auto& model:_models){
            std::string cur_alias = std::get<0>(model)->alias();
            if(cur_alias == alias){
                throw YOLO::YoloException("model alias already exists");
            }
            if(cur_alias == parent){
                found_parent = true;
                if(std::get<0>(model)->task_type() == YOLO::TASK_CLASSIFY){
                    if(branch >= std::get<0>(model)->engine()->getTensorShape("output0").d[1])
                        throw YOLO::YoloException("model parent's branch error");
                } else {
                    if(branch != 0)
                        throw YOLO::YoloException("model parent's is not classify but branch is not 0");
                }
            }
            if(std::get<1>(model) == parent && std::get<2>(model) == branch){
                throw YOLO::YoloException("model parent's branch already exists");
            }
        }
        _models.emplace_back(std::make_tuple(std::move(model), parent, branch));

        return std::get<0>(_models.back()).get();
    }

// workspace管理相关。工作空间是线程独立的资源管理器，比如在tensorrt中就是context
    // @brief 创建一个工作空间
    // @return 返回工作空间指针
	WorkSpaceT* get_new_workspace(){
		WorkSpaceT* workspace = new WorkSpaceT();
        std::vector<ModelT*> models;
        for(auto& model : _models){
            models.emplace_back(std::get<0>(model).get());
        }
        if(models.size() == 0)
            return nullptr;
        if(workspace->create_by(models)){
            _workspaces.emplace_back(workspace);
            return workspace;
        } else {
            delete workspace;
            throw YOLO::YoloException("workspace create by models failed");
            return nullptr;
        }
    }
    // @brief 获取默认工作空间
    // @details  默认工作空间就是第一个。如果工作空间数量为0，则创建一个工作空间
    // @return 默认工作空间指针
    WorkSpaceT* default_workspace(){
        if(_workspaces.size() > 0){
            return _workspaces[0];
        } else {
            return get_new_workspace();
        }
    }
    void relase_all_workspaces(){
        for(auto& workspace : _workspaces){
            delete workspace;
        }
    }
// taskflow管理相关
    bool binding_taskflow(){
        if(_taskflow){
            //  already binded
            return false;
        }
        auto taskflow = std::make_unique<TaskFlowT>();
        if(!taskflow->binding(_models)){
            // 不知道为什么，反正绑定失败了
            throw YOLO::YoloException("taskflow binding by models failed");
            return false;
        }
        _taskflow = std::move(taskflow);
        return true;
    }
    std::vector<YOLO::TASK::TaskResult> execute(cv::Mat& img, WorkSpaceT* ws){
        return _taskflow->execute(img, ws);
    }
private:
    std::unique_ptr<TaskFlowT> _taskflow = nullptr;
    std::vector<std::tuple<std::unique_ptr<ModelT>, std::string, int>> _models;
    std::vector<WorkSpaceT*> _workspaces;
    ModelType _type = ModelType::MODEL_TYPE_UNKNOWN;   // 所有模型的推理运行时必须是一致的
};





}
#endif // __YOLO_SCHEDULER_H__