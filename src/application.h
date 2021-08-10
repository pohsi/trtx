#pragma once

#include <memory>

namespace trtx {

class Appliction {
    
};


using ApplictionPtr = std::unique_ptr<Appliction>;


struct ApplicationBuildOption {
    bool m_serailize{ false };
    bool m_deserailize{ false };
    int m_batchSize{ 1 };
};

class ApplicationFactory {
public:

    ApplictionPtr Create(const ApplicationBuildOption &option) const;

};

}