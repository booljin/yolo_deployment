#ifndef __YOLO_MODEL_ANY_H__
#define __YOLO_MODEL_ANY_H__
#include "yolo_defines.h"

namespace YOLO{
namespace MODEL{

enum Status{
    // 模型会先于任务加载，并因此知道自己属于哪个推理任务，用于创建推理任务实例
    // 在创建对应的推理任务之前，暂不允许其创建推理上下文
    STATUS_NotReady = 0,
    STATUS_Ready = 1,
};


class Any{
public:
    Any(const std::string& alias):_alias(alias) {}
    virtual void* create_context() = 0;
    virtual ~Any() = default;

public:
    inline YOLO::ModelType type() const{ return _type; }
    virtual inline YOLO::TaskType task_type() { return _task_type; }
    inline Status status() const{ return _status; }

    inline double confidence_threshold() const {return _confidence_threshold;}
    inline double nms_threshold() const {return _nms_threshold;}
    inline double mask_threshold() const {return _mask_threshold;}

    inline void set_confidence_threshold(double threshold){ _confidence_threshold = threshold; }
    inline void set_nms_threshold(double threshold) {_nms_threshold = threshold;}
    inline void set_mask_threshold(double threshold) {_mask_threshold = threshold;}

    inline void set_alias(const std::string& alias) {_alias = alias;}
    inline std::string alias() {return _alias;}
protected:
    YOLO::ModelType _type = MODEL_TYPE_UNKNOWN;
    YOLO::MODEL::Status _status = STATUS_NotReady;
    TaskType _task_type = TASK_UNKNOWN;
protected:
    double _confidence_threshold = 0.3;
    double _nms_threshold = 0.5;
    double _mask_threshold = 0.5;
protected:
    std::string _alias;
};


}}
#endif // __YOLO_MODEL_LOADER_ANY_H__