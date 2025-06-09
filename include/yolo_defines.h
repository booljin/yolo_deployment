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

class YoloException : public std::exception {
public:
	YoloException(const std::string& w) : exception(w.c_str()){
	}
};

}
#endif // __YOLO_DEFINES_H__