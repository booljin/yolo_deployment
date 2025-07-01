#ifndef __YOLO_DEFINES_H__
#define __YOLO_DEFINES_H__
#include <exception>
#include <string>

namespace YOLO{

enum ModelType {
    MODEL_TYPE_UNKNOWN = 0,
    MODEL_TYPE_TRT = 1, // TensorRT
    MODEL_TYPE_INTEL = 2, // Intel OpenVINO
    MODEL_TYPE_ONNX = 3, // ONNX Runtime
    MODEL_TYPE_TENSORFLOW = 4, // TensorFlow
    MODEL_TYPE_PYTORCH = 5, // PyTorch
};

enum TaskType{
    TASK_UNKNOWN = 0,
    TASK_CLASSIFY = 1,
    TASK_DETECT = 2,
    TASK_SEGMENT = 3,
};

class YoloException : public std::exception {
public:
	YoloException(const std::string& w) : exception(w.c_str()){
	}
};



}

namespace YOLO{namespace IMPL{
template<typename Backend>
struct BackendTraits;
}}
#endif // __YOLO_DEFINES_H__