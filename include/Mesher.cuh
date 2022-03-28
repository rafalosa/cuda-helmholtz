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
// ---------- Loud thoughts ----------

// todo: Add steps as a template parameter, so the size of the object is known at compile time. Then I can make the
//   size() method as static and return it in bytes for allocation on GPU. Now it is possible to do it with #define.

// todo: I can probably make the base function as a template and just inherit with a template too. But It just moves the
//  different float definitions to the base class, so not much point in doing it I guess. Also It will allow for the
//  get() method to be virtual.

// ---------- END ----------

class MeshBase{
protected:
    float meshUnitMultiplier;
    size_t steps_;
    explicit MeshBase(MeshUtils::Units unitMultiplier, size_t steps) : meshUnitMultiplier((float)unitMultiplier), steps_(steps){};
    virtual ~MeshBase() = default;
    virtual size_t size();

};

template<MeshUtils::Dim>
class Mesh;


template<>
class Mesh<MeshUtils::Dim::D1> : protected MeshBase{

private:

    float1* pointsArray{nullptr};

public:
    Mesh(MeshUtils::Units meshUnits, size_t steps, float x1Boundary, float x2Boundary);
    ~Mesh() override;
    size_t size() override;
    float1 get(size_t index);
};

template<>
class Mesh<MeshUtils::Dim::D2> : protected MeshBase{

private:
    float2** pointsArray{nullptr};

public:
    Mesh(MeshUtils::Units meshUnits, size_t steps, float x1Boundary, float x2Boundary,
         float y1Boundary, float y2Boundary);
    ~Mesh() override;
    size_t size() override;
    float2 get(size_t idX, size_t idY);
};

template<>
class Mesh<MeshUtils::Dim::D3> : protected MeshBase{

private:
    float3*** pointsArray{nullptr};

public:
    Mesh(MeshUtils::Units meshUnits, size_t steps, float x1Boundary, float x2Boundary,
         float y1Boundary, float y2Boundary, float z1Boundary, float z2Boundary);
    ~Mesh() override;
    size_t size() override;
    float3 get(size_t idX, size_t idY, size_t idZ);
};


#endif //HELMHOLTZCUDA_MESHER_CUH
