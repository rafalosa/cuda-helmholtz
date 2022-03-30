#ifndef HELMHOLTZCUDA_MESHUTILS_H
#define HELMHOLTZCUDA_MESHUTILS_H

namespace MeshUtils{
    enum class Dim : short {
        D1 = 0,
        D2 = 1,
        D3 = 2
    };

    enum class Units : size_t { // Don't know if I'll use it, checking things out.

        MILLIMETERS = 1000,
        CENTIMETERS = 100,
        METERS = 1,
    };
}

#endif //HELMHOLTZCUDA_MESHUTILS_H
