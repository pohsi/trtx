#pragma once

#include <memory>

namespace trtx {

class Application {
public:

    virtual ~Application();
    
    virtual int Run() = 0;
};


using ApplicationPtr = std::unique_ptr<Application>;


struct ApplicationBuildOption {
    bool m_serailize{ false };
    bool m_deserailize{ false };
    int m_batchSize{ 1 };
};

class ApplicationFactory {
public:

    ApplicationPtr Create(const ApplicationBuildOption &option) const;

};

}