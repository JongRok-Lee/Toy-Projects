#include "detection.h"

int main(int argc, char** argv)
{   
    std::string image(argv[1]);
    double T1 = atof(argv[2]), T2 = atof(argv[3]);
    std::string method(argv[4]);
    std::cout << "Interpixel Approximation method: " << method << "\n";
    std::cout << "T1 : " << T1 << ",\t T2: " << T2 << "\n";
    Corner Trajkovic(image, T1, T2, method);
    Trajkovic.show();


    return 0;
}