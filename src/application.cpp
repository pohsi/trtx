#include "application.h"
#include "common.h"
#include "NvInfer.h"
#include "buffers.h"
#include <fstream>
#include <unordered_map>
#include <vector>


namespace {

using namespace trtx;

const char weightPath[] { "../data/model.wts" };
const char enginePath[] { "../data/model.engine" };
const char inputBlobName[] { "data" };
const char ouputBlobName[] { "prob" };
constexpr size_t inputHeight{ 32 };
constexpr size_t inputWidth{ 32 };
constexpr int32_t outputSize{ 10 };

using EnginePtr = std::shared_ptr<nvinfer1::ICudaEngine>;

bool SaveEngine(const ICudaEngine& engine, const std::string_view &filePath, std::ostream &err) {
    std::ofstream engineFile(filePath.data(), std::ios::binary);
    if (!engineFile) {
        err << "Cannot open engine file: " << filePath << std::endl;
        return false;
    }

    auto serializedEngine{ infer_object(engine.serialize()) };
    if (serializedEngine == nullptr) {
        err << "Engine serialization failed" << std::endl;
        return false;
    }

    engineFile.write(static_cast<char*>(serializedEngine->data()), serializedEngine->size());
    return !engineFile.fail();
}

bool ReadReferenceFile(const std::string &fileName, std::vector<std::string> &refVector)
{
    std::ifstream infile(fileName);
    if (!infile.is_open()) {
        gLogError << "ERROR: readReferenceFile: Attempting to read from a file that is not open." << std::endl;
        return false;
    }
    std::string line;
    while (std::getline(infile, line)) {
        if (line.empty()) {
            continue;
        }
        refVector.push_back(line);
    }
    infile.close();
    return true;
}

// Seralize to model
class APIToModelApplication : public Application {
    
public:
    explicit APIToModelApplication(int batchSize)
        : m_engine{ this->CreateEngine(batchSize) }
    {}

    virtual int Run() override {

        if (nullptr == this->m_engine) {
            gLogError << "APIToModelApplication::Run() error, engine is not created" << std::endl;
            return -1;
        }
        auto enginePlan{ infer_object(this->m_engine->serialize()) };
        bool result{ SaveEngine(*this->m_engine, enginePath, std::cerr) };
        ASSERT(true == result);
        return true == result ? 0 : -1;
    }

private:

    using WeightMap = std::unordered_map<std::string, nvinfer1::Weights>;

    using HostMemoryPtr = std::unique_ptr<HostMemory>;

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
            auto val{ mem->raw() };
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
        auto network{ infer_object(builder.createNetworkV2(0)) };
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
        auto builder{ infer_object(nvinfer1::createInferBuilder(gLogger.getTRTLogger())) };
        if (nullptr == builder) {
            ASSERT(false && "CreateInferBuilder failed.");
            gLogError << "CreateInferBuilder failed" << std::endl;
            return nullptr;
        }

        auto config{ infer_object(builder->createBuilderConfig()) };
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


        return EnginePtr{ builder->buildEngineWithConfig(*network, *config), InferDeleter{} };
    }

    std::vector<HostMemoryPtr> m_weightsMemory;

    EnginePtr m_engine{ nullptr };
};

class InfererenceApplication : public Application {
public:

    explicit InfererenceApplication(int batchSize)
        : m_batchSize{ batchSize }
        , m_engine{ this->CreateEngine(batchSize) }
    {}

    virtual int Run() override {

        if (nullptr == this->m_engine) {
            gLogError << "InfererenceApplication::Run() error, engine is not created" << std::endl;
            return -1;
        }

        if (false == GetInputOutputNames()) {
            gLogError << "GetInputOutputNames failed" << std::endl;
            return -1;
        }

        auto context{ infer_object(this->m_engine->createExecutionContext()) };
        ASSERT(nullptr != context);
        if (nullptr == context) {
            gLogError << "CreateExecutionContext failed" << std::endl;
            return -1;
        }

        BufferManager buffers{ this->m_engine, this->m_batchSize };

        if (false == this->ProcessInput(buffers, inputHeight, inputWidth)) {
            gLogError << "ProcessInput failed" << std::endl;
            return -1;
        }

        cudaStream_t stream;
        CHECK(cudaStreamCreate(&stream));


        buffers.copyInputToDeviceAsync(stream);
        if (!context->enqueueV2(buffers.getDeviceBindings().data(), stream, nullptr)) {
            return -1;
        }
        buffers.copyOutputToHostAsync(stream);
        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
        PrintOutput(buffers);
        return 0;

    }

    EnginePtr CreateEngine(int) {

        std::ifstream engineFile(enginePath, std::ios::binary);
        if (engineFile.fail()) {
            gLogError << "Load engine file failed, path: " << enginePath << std::endl;
            return nullptr;
        }

        engineFile.seekg(0, std::ifstream::end);
        auto fsize{ engineFile.tellg() };
        engineFile.seekg(0, std::ifstream::beg);

        std::vector<char> engineData(fsize);
        engineFile.read(engineData.data(), fsize);

        auto runtime{ infer_object(nvinfer1::createInferRuntime(gLogger.getTRTLogger())) };
        EnginePtr engine{ runtime->deserializeCudaEngine(engineData.data(), fsize, nullptr) };
        ASSERT(nullptr != engine && "DeserializeCudaEngine failed");
        return engine;
    }

    bool GetInputOutputNames()
    {
        auto engine{ this->m_engine.get() };
        const int numberOfBindings{ engine->getNbBindings() };
        ASSERT(numberOfBindings == 2);
        if (numberOfBindings != 2) {
            return false;
        }

        for (int b = 0; b < numberOfBindings; ++b) {
            const nvinfer1::Dims dims{ engine->getBindingDimensions(b) };
            if (engine->bindingIsInput(b)) {
                if (this->m_verbose) {
                    gLogInfo << "Found input: " << engine->getBindingName(b) << " shape=" << dims
                                    << " dtype=" << (int) engine->getBindingDataType(b) << std::endl;
                }
                this->m_inOut["input"] = engine->getBindingName(b);
            }
            else {
                if (this->m_verbose) {
                    gLogInfo << "Found output: " << engine->getBindingName(b) << " shape=" << dims
                                    << " dtype=" << (int) engine->getBindingDataType(b) << std::endl;
                }
                this->m_inOut["output"] = engine->getBindingName(b);
            }
        }
    }


    bool ProcessInput(const trtx::BufferManager &buffers, int inputH, int inputW) {
        //const std::vector<uint8_t> inputData(inputH * inputW, 1);
        // Fill input buffer with all 1's which size is H * W
        float *hostDataBuffer = static_cast<float*>(buffers.getHostBuffer(this->m_inOut["input"]));
        for (int i = 0; i < inputH * inputW; i++) {
            hostDataBuffer[i] = 1;
        }
        return true;
    }

    void PrintOutput(const BufferManager &buffers) const {
        const float *probPtr{ static_cast<const float*>(buffers.getHostBuffer(this->m_inOut.at("output"))) };
        std::cout << "Outout: ";
        for (const auto &iter : std::vector<float>{ probPtr, probPtr + outputSize }) {
            std::cout << iter << " ";
        }
        std::cout << std::endl;
    }

    bool m_verbose{ true };

    int m_batchSize{ 1 };

    std::unordered_map<std::string, std::string> m_inOut;

    EnginePtr m_engine{ nullptr };
};

class NullApplication : public Application {
public:
    virtual int Run() override {
        std::cerr << "Option is required, try 'trtx --help' for more information" << std::endl;
        return 0;
    }
};

}

namespace trtx {

Application::~Application() = default;

ApplicationPtr ApplicationFactory::Create(const ApplicationBuildOption &option) const {
    if (true == option.m_serailize) {
        return ApplicationPtr{ new APIToModelApplication{ option.m_batchSize } };
    }
    else if (true == option.m_deserailize) {
        return ApplicationPtr{ new InfererenceApplication{ option.m_batchSize } };
    }
    return ApplicationPtr{ new NullApplication{} };
}

}