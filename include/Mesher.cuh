//
// Created by rafal on 28.03.2022.
//

#ifndef HELMHOLTZCUDA_MESHER_CUH
#define HELMHOLTZCUDA_MESHER_CUH

namespace MeshUtils{
    enum class Dim : short {
        D1 = 0,
        D2 = 1,
        D3 = 2
    };

    enum class Units : short { // Don't know if I'll use it, checking things out.

        MILLIMETERS = 1000,
        CENTIMETERS = 100,
        METERS = 1,
    };
}

// todo: Maybe add mesh base class that contains mutual behaviour if there is any.

class MeshBase{
protected:
    float meshUnitMultiplier;
    explicit MeshBase(MeshUtils::Units unitMultiplier) : meshUnitMultiplier((float)unitMultiplier){};
    virtual ~MeshBase();
    virtual float operator[](const MeshBase& mesh);
    virtual size_t size();



};


template<MeshUtils::Dim>
class Mesh;


template<>
class Mesh<MeshUtils::Dim::D1>{

private:

    float* pointsArray{nullptr};
    float meshUnitMultiplier;

public:
    Mesh(MeshUtils::Units meshUnits, size_t steps, float x1Boundary, float x2Boundary);
    ~Mesh();


};

template<>
class Mesh<MeshUtils::Dim::D2>{

private:
    float** pointsArray{nullptr};
    float meshUnitMultiplier;

public:
    Mesh(MeshUtils::Units meshUnits, size_t steps, float x1Boundary, float x2Boundary,
         float y1Boundary, float y2Boundary);
    ~Mesh();

};

template<>
class Mesh<MeshUtils::Dim::D3>{

private:
    float*** pointsArray{nullptr};
    float meshUnitMultiplier;

public:
    Mesh(MeshUtils::Units meshUnits, size_t steps, float x1Boundary, float x2Boundary,
         float y1Boundary, float y2Boundary, float z1Boundary, float z2Boundary);
    ~Mesh();
};


#endif //HELMHOLTZCUDA_MESHER_CUH
