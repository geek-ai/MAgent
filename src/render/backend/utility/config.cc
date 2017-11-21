#include <cstring>
#include <iostream>
#include "config.h"

namespace magent {
namespace render {

const char MATRIX_ARGP_ARGS_DOCUMENT[] = "arg1 arg2 ...";
const char MATRIX_ARGP_DOCUMENT[] = "render: The backend server of magnet platform.";
const argp_option MATRIX_ARGP_OPTIONS[] ={
        {"port"                 , 'P', "PORT" , 0, "Specify the port to be used by the server(the default port is 9030)."                                , 1},
        {"quiet"                , 'Q', nullptr, 0, "Quiet mode will be used and almost all warning, diagnostic and exception message will be suppressed.", 2},
        {nullptr}
};

error_t __parser(int key, char *arg, struct argp_state *state) {
    RenderConfig &config = *static_cast<RenderConfig *>(state->input);
    switch (key) {
        case 'P': {
            int port = 0;
            for (size_t i = 0, size = strlen(arg); i < size; i++) {
                if (isdigit(arg[i]) == 0) {
                    std::cout << "port must be positive integer between (0..65535)." << std::endl;
                    return ARGP_ERR_UNKNOWN;
                }
                port = port * 10 + (arg[i] - '0');
                if (port < 0 || port >= 65536) {
                    std::cout << "port must be positive integer between (0..65535)." << std::endl;
                    return ARGP_ERR_UNKNOWN;
                }
            }
            config.port = static_cast<uint16_t>(port);
            break;
        }
        case 'Q': {
            config.quiet = true;
            break;
        }
        default:
            return ARGP_ERR_UNKNOWN;
    }
    return 0;
}

const argp MATRIX_ARGP = {
        MATRIX_ARGP_OPTIONS,
        __parser,
        MATRIX_ARGP_ARGS_DOCUMENT,
        MATRIX_ARGP_DOCUMENT
};

void parse(int argc, char *argv[], RenderConfig &config) {
    argp_parse(&MATRIX_ARGP, argc, argv, ARGP_NO_ARGS | ARGP_IN_ORDER, 0, &config);
}

} // namespace render
} // namespace magnet