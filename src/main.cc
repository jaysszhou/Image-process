#include "XReader.h"
#include <iostream>

int main(int argc, char **argv)
{
    if(argc != 2){
        std::cout << "Usage: ./read_pic path_to_local_pic !" << std::endl;
        return 0;
    }
    XReader xReader;
    std::string picture_dir = argv[1];
    xReader.ShowLocalPicture(picture_dir);
}