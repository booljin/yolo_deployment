#ifndef __YOLO_SCHEDULER_H__
#define __YOLO_SCHEDULER_H__

#include <vector>
#include <memory>
#include "yolo_defines.h"
#include "model_base.h"
#include "workspace_base.h"
#include "task_base.h"

namespace YOLO{

class YoloScheduler {
public:
    YoloScheduler() = default;
    ~YoloScheduler();
public:
// model管理相关
    std::shared_ptr<ModelAny> load_model(ModelType type, const std::string& model_file);
// workspace管理相关
    WorkSpaceAny* get_new_workspace();
    WorkSpaceAny* default_workspace();
    void relase_all_workspaces();
// taskflow管理相关
    bool binding_taskflow();
    std::vector<YOLO::TASK::TaskResult> execute(cv::Mat& img, WorkSpaceAny*);
private:
    std::shared_ptr<YOLO::TASK::TaskFlowAny> _taskflow = nullptr;
    std::vector<std::shared_ptr<ModelAny>> _models;
    std::vector<WorkSpaceAny*> _workspaces;
    ModelType _type = ModelType::MODEL_TYPE_UNKNOWN;   // 所有模型的推理运行时必须是一致的
};

}
#endif // __YOLO_SCHEDULER_H__