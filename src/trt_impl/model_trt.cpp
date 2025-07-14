#include "model_trt.h"

#include <fstream>
#include <sstream>
#include <filesystem>

#include "yolo_defines.h"
#include "yolo_trt_utils.h"

#include "NvOnnxParser.h"

static YOLO::UTILS::TRT::Logger gLogger;

using namespace YOLO::MODEL::TRT;

class TRT_Runtime{
public:
    ~TRT_Runtime(){
        if(_runtime){
            delete _runtime;
        }
    }
    static nvinfer1::IRuntime* getRuntime(){
        static TRT_Runtime instance;
        if(!instance._runtime){
            instance._runtime = nvinfer1::createInferRuntime(gLogger);
        }
        return instance._runtime;
    }
private:
    TRT_Runtime() = default;
    nvinfer1::IRuntime* _runtime = nullptr;
};


Model::Model(const std::string& alias) : YOLO::MODEL::Any(alias){
    _engine = nullptr;
}

Model::~Model(){
    destroy();    
}

bool Model::load(const std::string& file_name){
	std::string engine;
	std::string onnx;
	{
		std::filesystem::path file = std::filesystem::path(file_name);
		std::filesystem::path ext = file.extension();
		std::filesystem::path onnx_path;
		std::filesystem::path engine_path;
        bool load_direct = false;
        if(ext == ".onnx"){
			onnx_path = file;
            engine_path = file.replace_extension(std::filesystem::path(".engine"));
        } else if (ext == ".engine") {
			engine_path = file;
			onnx_path = file.replace_extension(std::filesystem::path(".onnx"));
        } else {
            // 不支持的文件类型
            return false;
        }
		engine = engine_path.string();
		load_direct = is_engine_file_valid(onnx_path, engine_path);
		if (!load_direct) {
			_type = YOLO::ModelType::MODEL_TYPE_TRT;
            _convert_thread = std::thread(&Model::convert_from_onnx, this, onnx_path, engine_path);
            return true;
		}
    }


    if(_type != YOLO::ModelType::MODEL_TYPE_UNKNOWN) return true;
    std::string engine_buff;
    std::ifstream engine_file(engine, std::ios::binary);
    if(!engine_file)
        return false;
    std::ostringstream oss;
    oss << engine_file.rdbuf();
    engine_buff = oss.str();
    engine_file.close();

    return build_engine(engine_buff.data(), engine_buff.size());
}

void Model::destroy(){
	if (_engine) {
		delete _engine;
        _engine = nullptr;
	}
}

nvinfer1::IExecutionContext* Model::create_context_trt(){
    if(_convert_thread.joinable()){
        // 等待转换线程完成
        _convert_thread.join();
    }
    if(!_engine){
        return nullptr;
    }
    if(_status != YOLO::MODEL::Status::STATUS_Ready){
        // 模型未准备好，不能创建上下文
        return nullptr;
    }
    auto context = _engine->createExecutionContext();
    return context;
}

void* Model::create_context(){
    return create_context_trt();
}

bool Model::is_engine_file_valid(const std::filesystem::path& onnx, const std::filesystem::path& engine) {
    if(std::filesystem::exists(engine)){
        if(!std::filesystem::exists(onnx)) {
            // 如果引擎文件存在，但ONNX文件不存在，则认为引擎文件是有效的
            return true;
        } else {
            auto onnx_time = std::filesystem::last_write_time(onnx);
            auto engine_time = std::filesystem::last_write_time(engine);
            return onnx_time <= engine_time;
        }
    } else {
        if(!std::filesystem::exists(onnx)) {
            // 如果ONNX文件和引擎文件都不存在，则认为无效
            throw YOLO::YoloException("Both ONNX and engine files do not exist.");
        } else {
            return false;
        }
    }
}


void Model::convert_from_onnx(const std::filesystem::path& onnx, const std::filesystem::path& engine){
	std::unique_lock<std::mutex> lock(_task_type_mtx);
	nvinfer1::IBuilder* builder = nvinfer1::createInferBuilder(gLogger);


	uint32_t flag = 1U << static_cast<uint32_t>(nvinfer1::NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
	nvinfer1::INetworkDefinition* network = builder->createNetworkV2(flag);

	nvonnxparser::IParser* parser = nvonnxparser::createParser(*network, gLogger);
	/*if (!parser) {
		std::cerr << "创建 ONNX 解析器失败。" << std::endl;
		return;
	}*/

	parser->parseFromFile(onnx.string().c_str(), static_cast<int32_t>(nvinfer1::ILogger::Severity::kVERBOSE));
	//for (int32_t i = 0; i < parser->getNbErrors(); ++i)
	//{
	//	std::cout << parser->getError(i)->desc() << std::endl;
	//}

	// 读取模型维度，判断任务类型
	auto output1 = network->getOutput(1);
	if(output1){
		// 分割任务有两个输出，一个是检测框，一个是mask。其中mask有4个维度：batch,maskdim,h,w
		_task_type = YOLO::TaskType::TASK_SEGMENT;
	} else {
		auto output0 = network->getOutput(0);
		auto dims_0 = output0->getDimensions();
		if (dims_0.nbDims == 3) {
			// 检测任务有3个维度：batch，bbox， box_count
			_task_type = YOLO::TaskType::TASK_DETECT;
		}
		else {
			// 分类任务有2个维度：batch， class_count
			_task_type = YOLO::TaskType::TASK_CLASSIFY;
		}
	}
	lock.unlock();

	nvinfer1::IBuilderConfig* config = builder->createBuilderConfig();

	config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, 16 * (1 << 20));

	if (builder->platformHasFastFp16())
	{
		config->setFlag(nvinfer1::BuilderFlag::kFP16);
	}


	nvinfer1::IHostMemory* serializedModel = builder->buildSerializedNetwork(*network, *config);

	std::ofstream p(engine.string(), std::ios::binary);
	p.write(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

    // 直接构建引擎，不用再读一遍文件
    build_engine(reinterpret_cast<const char*>(serializedModel->data()), serializedModel->size());

	delete parser;
	delete network;
	delete config;
	delete builder;

	delete serializedModel;

}

bool Model::build_engine(const char* buffer, size_t size){
    auto runtime = TRT_Runtime::getRuntime();
    
    _engine = runtime->deserializeCudaEngine(buffer, size);
    if(_engine == nullptr){
        return false;
    }
    _type = YOLO::ModelType::MODEL_TYPE_TRT;

    // 读取模型维度，判断任务类型
    auto dims = _engine->getTensorShape("output1");
    if(dims.nbDims == 4){
        // 分割任务有两个输出，一个是检测框，一个是mask。其中mask有4个维度：batch,maskdim,h,w
        _task_type = YOLO::TaskType::TASK_SEGMENT;
    } else {
        auto dims_0 = _engine->getTensorShape("output0");
        if(dims_0.nbDims == 3){
            // 检测任务有3个维度：batch，bbox， box_count
            _task_type = YOLO::TaskType::TASK_DETECT;
        } else {
            // 分类任务有2个维度：batch， class_count
            _task_type = YOLO::TaskType::TASK_CLASSIFY;
        }
    }
	_status = YOLO::MODEL::Status::STATUS_Ready;
    return true;
}