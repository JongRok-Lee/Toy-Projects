#include "detection.h"

int main(int argc, char** argv)
{   
    std::string image(argv[1]);
    double resize_param = atof(argv[2]);
    double T1 = atof(argv[3]),
           T2 = atof(argv[4]);
    std::string method(argv[5]);
    
    std::cout << "Interpixel Approximation method: " << method << "\n";
    std::cout << "T1 : " << T1 << ",\t T2: " << T2 << "\n";

    Corner Trajkovic(image, resize_param, T1, T2, method);
    Trajkovic.show();

    return 0;
}
