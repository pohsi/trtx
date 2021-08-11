#include <getopt.h>
#include <vector>
#include <string>
#include <iostream>
#include "common.h"
#include "application.h"

namespace {

using namespace trtx;

struct Args {
    bool m_help{false};
    ApplicationBuildOption m_option;
};

bool ParseArgs(Args &args, int argc, char **argv) {
    constexpr static struct option long_options[] = {
        { "help", no_argument, 0, 'h' },
        { "serialize", no_argument, 0, 's' },
        { "deserialize", no_argument, 0, 'd' },
        { "batch_size", required_argument, 0, 'b' },
        { nullptr, 0, nullptr,  0 }
    };
    int option_index = 0;
    auto getOpt = [&] () {
        return getopt_long(argc, argv, "hsdb:", long_options, &option_index);
    };
    auto &option{ args.m_option };
    for(int arg{ getOpt() }; -1 != arg; arg = getOpt()) {
        switch (arg) {
            case 'h':
                args.m_help = true;
                return false;
            case 's':
                option.m_serailize = true;
                break;
            case 'd':
                option.m_deserailize = true;
                break;
            case 'b':
                if (optarg) {
                    if (auto size{ std::stoi(optarg) }; size > 1) {                    
                        option.m_batchSize = size;
                    }
                }
                else {
                    std::cerr << "ERROR: --batch_size requires option argument" << std::endl;
                    return false;
                }
                break;
            default:
                return false;
        }
    }
    return true;
}

void PrintHelpInfo() {
    std::cout << "Usage: ./trtx [-h or --help]\n" << 
                 "--help[-h]          Display help information\n" <<
                 "--serialize[-s]     Serialize model to plan file\n" <<
                 "--deserialize[-d]   Deserialize plan file and run inference\n" <<
                 "--batchSize[-b]     Specify the batch size for inference." << std::endl;
}

}

int main(int argc, char **argv) {
    Args args;

    if (false == ParseArgs(args, argc, argv) || true == args.m_help) {
        PrintHelpInfo();
        return 0;
    }

    return ApplicationFactory().Create(args.m_option)->Run();
}