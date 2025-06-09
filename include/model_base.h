#ifndef __YOLO_MODEL_ANY_H__
#define __YOLO_MODEL_ANY_H__
#include "yolo_defines.h"

namespace YOLO{

class ModelAny{
public:
    virtual void* create_context() = 0;
    virtual ~ModelAny() = default;
public:
    enum Status{
        // 模型会先于任务加载，并因此知道自己属于哪个推理任务，用于创建推理任务实例
        // 在创建对应的推理任务之前，暂不允许其创建推理上下文
        STATUS_NotReady = 0,
        STATUS_Ready = 1,
    };
    enum TaskType{
        TASK_UNKNOWN = 0,
        TASK_CLASSIFY = 1,
        TASK_DETECT = 2,
        TASK_SEGMENT = 3,
    };
public:
    inline ModelType type() const{ return _type; }
    inline TaskType task_type() const{ return _task_type; }
    inline Status status() const{ return _status; }

    inline double confidence_threshold() const {return _confidence_threshold;}
    inline double nms_threshold() const {return _nms_threshold;}
    inline double mask_threshold() const {return _mask_threshold;}

    inline void set_confidence_threshold(double threshold){ _confidence_threshold = threshold; }
    inline void set_nms_threshold(double threshold) {_nms_threshold = threshold;}
    inline void set_mask_threshold(double threshold) {_mask_threshold = threshold;}

protected:
    ModelType _type = MODEL_TYPE_UNKNOWN;
    Status _status = STATUS_NotReady;
    TaskType _task_type = TASK_UNKNOWN;
protected:
    double _confidence_threshold = 0.3;
    double _nms_threshold = 0.5;
    double _mask_threshold = 0.5;
};


}
#endif // __YOLO_MODEL_LOADER_ANY_H__