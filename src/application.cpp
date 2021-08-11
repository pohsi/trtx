#include "application.h"
#include "common.h"
#include "NvInfer.h"
#include <fstream>
#include <unordered_map>
#include <vector>


namespace {

using namespace trtx;

const char weightPath[] { "../data/model.wts" };
const char enginePath[] { "./model.engine" };
const char inputBlobName[] { "data" };
const char ouputBlobName[] { "prob" };
constexpr size_t inputHeight{ 32 };
constexpr size_t inputWidth{ 32 };
constexpr int32_t outputSize{ 10 };

// Seralize to model
class SerializeApplication : public Application {
    
public:
    explicit SerializeApplication(int batchSize)
        : m_engine{ this->CreateEngine(batchSize) }
    {}

    virtual int Run() override {

        if (nullptr == this->m_engine) {
            gLogError << "SerializeApplication::Run() error, engine is not create" << std::endl;
            return -1;
        }
        //TrtxUniquePtr<nvinfer1::IHostMemory> engine_plan{ trt_engine->serialize() };
        //std::ofstream engine_file(engine_filename.c_str(), std::ios::binary);
        //engine_file.write(reinterpret_cast<const char*>(engine_plan->data()), engine_plan->size());
        //engine_file.close();

        return 0;
    }

private:

    using WeightMap = std::unordered_map<std::string, nvinfer1::Weights>;

    using HostMemoryPtr = std::unique_ptr<HostMemory>;

    using EnginePtr = std::shared_ptr<nvinfer1::ICudaEngine>;

    WeightMap LoadWeights(std::istream &is) {

        WeightMap weightMap;
        int32_t count{ 0 };
        is >> count;
        if (count <= 0 ) {
            ASSERT(false && "Invalid weight map file.");
            return WeightMap{};
        }

        while (count--) {
            nvinfer1::Weights wt{ nvinfer1::DataType::kFLOAT, nullptr, 0 };
            uint32_t size{ 0 };

            // Read name and type of blob
            std::string name;
            is >> name >> std::dec >> size;
            // Load blob
            auto mem{ new TypedHostMemory<uint32_t, nvinfer1::DataType::kFLOAT>(size) };
            this->m_weightsMemory.emplace_back(mem);
            auto val{ mem->Raw() };
            for (uint32_t x = 0; x < size; ++x) {
                is >> std::hex >> val[x];
            }
            wt.type = nvinfer1::DataType::kFLOAT;
            wt.values = val;
            wt.count = size;
            weightMap[name] = wt;
            
        }
        return weightMap;
    }


    std::shared_ptr<nvinfer1::INetworkDefinition> BuildNetwork(nvinfer1::IBuilder &builder, const WeightMap &weightMap) {
        auto network{ InferObject(builder.createNetworkV2(0)) };
        ASSERT(nullptr != network);

        auto tensor{ network->addInput(inputBlobName, nvinfer1::DataType::kFLOAT, nvinfer1::Dims{ 1, { inputHeight, inputWidth } }) };
        ASSERT(nullptr != tensor && "network addInput failed");

        auto iterConv1Weight{ weightMap.find("conv1.weight") };
        auto iterConv1Bias{ weightMap.find("conv1.bias") };
        ASSERT(weightMap.end() != iterConv1Weight && weightMap.end() != iterConv1Bias);

        auto conv1 = network->addConvolution(*tensor, 6, nvinfer1::DimsHW{ 5, 5 }, iterConv1Weight->second, iterConv1Bias->second);
        ASSERT(nullptr != conv1 && "network addConvolution failed");
        conv1->setStride(nvinfer1::DimsHW{ 1, 1 });

        auto relu1{ network->addActivation(*conv1->getOutput(0), nvinfer1::ActivationType::kRELU) };
        ASSERT(nullptr != relu1 && "network addActivation failed");

        auto pool1{ network->addPooling(*relu1->getOutput(0), nvinfer1::PoolingType::kAVERAGE, nvinfer1::DimsHW{ 2, 2 }) };
        ASSERT(nullptr != pool1 && "network addPooling failed");
        pool1->setStride(nvinfer1::DimsHW{ 2, 2 });


        auto iterConv2Weight{ weightMap.find("conv2.weight") };
        auto iterConv2Bias{ weightMap.find("conv2.bias") };
        ASSERT(weightMap.end() != iterConv1Weight && weightMap.end() != iterConv1Bias);
        auto conv2 = network->addConvolution(*pool1->getOutput(0), 16, nvinfer1::DimsHW{ 5, 5 }, iterConv2Weight->second, iterConv2Bias->second);
        ASSERT(nullptr != conv2 && "network addConvolution failed");
        conv2->setStride(nvinfer1::DimsHW{ 1, 1 });

        auto relu2{ network->addActivation(*conv2->getOutput(0), nvinfer1::ActivationType::kRELU) };
        ASSERT(nullptr != relu2 && "network addActivation failed");

        auto pool2{ network->addPooling(*relu2->getOutput(0), nvinfer1::PoolingType::kAVERAGE, nvinfer1::DimsHW{ 2, 2 }) };
        ASSERT(nullptr != pool2 && "network addPooling failed");
        pool2->setStride(nvinfer1::DimsHW{ 2, 2 });

        auto iterFc1Weight{ weightMap.find("fc1.weight") };
        auto iterFc1Bias{ weightMap.find("fc1.bias") };
        ASSERT(weightMap.end() != iterFc1Weight && weightMap.end() != iterFc1Bias);
        auto fc1{ network->addFullyConnected(*pool2->getOutput(0), 120, iterFc1Weight->second, iterFc1Bias->second) };
        ASSERT(nullptr != fc1 && "network addFullyConnected failed");


        auto relu3{ network->addActivation(*fc1->getOutput(0), nvinfer1::ActivationType::kRELU) };
        ASSERT(nullptr != relu3 && "network addActivation failed");

        auto iterFc2Weight{ weightMap.find("fc2.weight") };
        auto iterFc2Bias{ weightMap.find("fc2.bias") };
        ASSERT(weightMap.end() != iterFc2Weight && weightMap.end() != iterFc2Bias);
        auto fc2{ network->addFullyConnected(*relu3->getOutput(0), 84, iterFc2Weight->second, iterFc2Bias->second) };
        ASSERT(nullptr != fc2 && "network addFullyConnected failed");

        auto relu4{ network->addActivation(*fc2->getOutput(0), nvinfer1::ActivationType::kRELU) };
        ASSERT(nullptr != relu4 && "network addActivation failed");


        auto iterFc3Weight{ weightMap.find("fc3.weight") };
        auto iterFc3Bias{ weightMap.find("fc3.bias") };
        ASSERT(weightMap.end() != iterFc3Weight && weightMap.end() != iterFc3Bias);
        auto fc3{ network->addFullyConnected(*relu4->getOutput(0), outputSize, iterFc3Weight->second, iterFc3Bias->second) };
        ASSERT(nullptr != fc3 && "network addFullyConnected failed");

        auto prob{ network->addSoftMax(*fc3->getOutput(0)) };
        ASSERT(nullptr != prob && "network addSoftMax failed");
        prob->getOutput(0)->setName(ouputBlobName);
        network->markOutput(*prob->getOutput(0));

        return network;
    }

    EnginePtr CreateEngine(int batchSize) {
        gLogInfo << "Enter createEngine" << std::endl;
        auto builder{ InferObject(nvinfer1::createInferBuilder(gLogger.getTRTLogger())) };
        if (nullptr == builder) {
            ASSERT(false && "CreateInferBuilder failed.");
            gLogError << "CreateInferBuilder failed" << std::endl;
            return nullptr;
        }

        auto config{ InferObject(builder->createBuilderConfig()) };
        if (nullptr == config) {
            ASSERT(false && "CreateBuilderConfig failed.");
            gLogError << "CreateBuilderConfig failed" << std::endl;
            return nullptr;
        }

        std::ifstream is{ weightPath, std::ios_base::binary };
        gLogInfo << "Loading weights from file: " << weightPath << std::endl;
        if (false == is.is_open()) {
            ASSERT(false && "Unable to load weight file.");
            gLogError << "Unable to load weight file: " << weightPath << std::endl;
            return nullptr;
        }

        const auto weightMap{ this->LoadWeights(is) };
        is.close();
        gLogInfo << "Leave createEngine" << std::endl;


        auto network{ this->BuildNetwork(*builder, weightMap) };
        if (nullptr == network) {
            ASSERT(false && "BuildNetwork failed.");
            gLogError << "BuildNetwork failed" << std::endl;
            return nullptr;
        }

        
        builder->setMaxBatchSize(batchSize);
        config->setMaxWorkspaceSize(1 << 20);


        return  EnginePtr{ builder->buildEngineWithConfig(*network, *config), InferDeleter{} };
    }

    std::vector<HostMemoryPtr> m_weightsMemory;

    EnginePtr m_engine{ nullptr };
};

class DeserializeApplication : public Application {
public:

    virtual int Run() override {
       return 0;
    }
};

}

namespace trtx {

Application::~Application() = default;

ApplicationPtr ApplicationFactory::Create(const ApplicationBuildOption &option) const {
    return ApplicationPtr{ new SerializeApplication{ option.m_batchSize } };
}

}