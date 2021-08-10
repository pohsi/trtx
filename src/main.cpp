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

bool ParseArgs(Args &options, int argc, char **argv) {
    constexpr static struct option long_options[] = {
        {"help", no_argument, 0, 'h'},
        {"serialize", no_argument, 0, 's'},
        {"deserialize", no_argument, 0, 'd'},
        {"batch_size", required_argument, 0, 'b'},
        {nullptr, 0, nullptr, 0}
    };
    for(; true; ) {
        int option_index = 0;
        int arg = getopt_long(argc, argv, "hsdb:", long_options, &option_index);
        if (-1 == arg) {
            break;
        }
        switch (arg)
        {
            case 'h':
                options.m_help = true;
                return false;
            case 's':
                break;
            case 'd':
                break;
            case 'b':
                if (optarg) {
                    if (auto size{ std::stoi(optarg) }; size > 1) {                    
                        options.m_option.m_batchSize = size;
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

void PrintHelpInfo()
{
    std::cout << "Usage: ./trtx [-h or --help]\n" << 
                 "--help[-h]          Display help information\n" <<
                 "--serialize[-s]     Serialize model to plan file\n" <<
                 "--deserialize[-d]   Deserialize plan file and run inference\n" <<
                 "--batchSize[-B]     Specify the batch size for inference." << std::endl;
}

}

int main(int argc, char **argv) {
    Args args;
    bool argsResult{ ParseArgs(args, argc, argv) };
    if (false == argsResult || true == args.m_help) {
        PrintHelpInfo();
    }
    return 0;
}