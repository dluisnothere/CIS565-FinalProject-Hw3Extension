#pragma once

#include <string>
#include <vector>
#include <cuda_runtime.h>
#include "glm/glm.hpp"
#include "image.h"

#define BACKGROUND_COLOR (glm::vec3(0.0f))

#define LOAD_GLTF 1
#define LOAD_OBJ 1
#define USE_KD_VEC 0;

enum GeomType {
    SPHERE,
    CUBE,
    TRIANGLE, // Triangle only exists in the scene if USE_BOUND_BOX 0
    OBJ, // Obj only exists in the scene if USE_BOUND_BOX 1
    GLTF
};

enum SARMode {
    SPOTLIGHT, 
    CIRCULAR_SPOTLIGHT,
    SCANNING,
    STRIPMAP
};

enum KDSPLIT {
    X,
    Y,
    Z
};

enum parentRelation {
    LEFT,
    RIGHT,
    ROOT
};

struct BoundBox {
    glm::vec3 minCorner;
    glm::vec3 maxCorner;
};

struct Ray {
    glm::vec3 origin;
    glm::vec3 direction;
};

struct Vertex {
    glm::vec4 pos;
    glm::vec4 nor;
    std::vector<int> host_texCoords;
    std::vector<glm::vec2> host_uvs;

    glm::vec4 tan = glm::vec4(0.f, 0.f, 0.f, 0.f); // dummy vals;'

    int* dev_texCoords;
    glm::vec2* dev_uvs;
    //int materialId;
};

struct Triangle {
    Vertex pointA;
    Vertex pointB;
    Vertex pointC;
};

struct KDNode {
    KDSPLIT split;
    parentRelation relation;
    int parent_node;
    int near_node;
    int far_node;
    //Refers to the triangle buffer 
    int trisIndex; 
    BoundBox bound; //should be max and min values of geom and subtrees. They are computed after tree is constructed

    //For debugging only
#if USE_KD_VEC
    int* device_trisIndices;
    std::vector<int> tempBuffer;
    int numIndices;
#endif
    int depth;

};

struct Geom {
    enum GeomType type;
    int materialid;
    glm::vec3 translation;
    glm::vec3 rotation;
    glm::vec3 scale;
    glm::mat4 transform;
    glm::mat4 inverseTransform;
    glm::mat4 invTranspose;

    Triangle* host_tris;
    Triangle* device_tris;
    BoundBox bound;
    int root;
    int numTris = 0;

#if LOAD_OBJ
    int objTexId = -1; // texture associated with geom // convert to array of textures
#endif
#if LOAD_GLTF
    int materialOffset = -1;
#endif
};

struct Texture {
    int width = 0;
    int height = 0;

    unsigned char* host_texData;
    int channels;
};

struct Material {
    glm::vec3 color;
    struct {
        float exponent;
        glm::vec3 color;
    } specular;
    float hasReflective;
    float hasRefractive;
    float indexOfRefraction;
    float emittance;

    //std::vector<int> texIdxs; // for GLTF, corresponding to roughness and etc.
    //std::vector<int> texCoords; // for GLTF

#if LOAD_GLTF
    struct {
        int baseColorIdx = -1;
        int baseColorOffset = -1;
        int baseColorTexCoord = -1;

        int metallicRoughnessIdx = -1;
        int metallicRoughnessOffset = -1;
        int metallicRoughnessTexCoord = -1;

        glm::vec3 baseColorFactor = glm::vec3(1.f, 1.f, 1.f);
        float metallicFactor = 1.0f;
        float roughnessFactor = 1.0f;
    } pbrMetallicRoughness;

    struct {
        int index = -1;
        int texOffset = -1;

        int texCoord = -1;
    } normalTexture;

    struct {
        int index = -1;
        int texOffset = -1;

        int texCoord = -1;
    }occlusionTexture;

    struct {
        int index = -1;
        int texOffset = -1;

        int texCoord = -1;
    } emissiveTexture;

    glm::vec3 emissiveFactor = glm::vec3(1.f, 1.f, 1.f);

#endif

};

struct Camera {
    glm::ivec2 resolution;
    glm::vec3 position;
    glm::vec3 lookAt;
    glm::vec3 view;
    glm::vec3 up;
    glm::vec3 right;
    glm::vec2 fov;
    glm::vec2 pixelLength;

    glm::mat4 inverseTransform;
    glm::mat4 transform;
};

struct RenderState {
    Camera camera;
    unsigned int iterations;
    int traceDepth;
    std::vector<glm::vec3> image;
    std::string imageName;

    bool usingSar;
    //Only for SAR
    int receiverIndex;
    bool moveReceiver;
};

struct Trajectory {
    SARMode mode; //Defines how antenna moves
    int snapshotCount; //How many different positions along the trajectory we trace from
    glm::vec3 closestApproach; //The mid point of the SAR mode movement
    glm::vec3 targetLocationAvg; //Average location of the target

    //These three vectors are used to manipulate a camera object to generate phase history
    std::vector<glm::vec3> vehicleTraj;
    std::vector<glm::vec3> lookPositions;
    std::vector<glm::vec3> rightVecs;

    //antenna receiver
    Camera antenna;
    glm::vec2 receiverScale;

    int iterations;
    //This comes in as an angle in radians
    float travelDistance;
};

struct PathSegment {
    Ray ray;
    glm::vec3 color;
    glm::vec3 color2;
    glm::vec3 color3;
    glm::vec3 outColor;
    glm::vec3 negPriRay;
    int pixelIndex;
    int pixelIndex2;
    int remainingBounces;
    float length;
    float length1;
    float length2;
    float length3;
    int depth;
    glm::vec2 ae1;
    int pixelIndexX;
    int pixelIndexX2;
    int pixelIndexY;
    int pixelIndexY2;
    
    glm::vec3 realRayDir;
    bool checkCameraBlock;
    
    bool fordebug;
};

// Use with a corresponding PathSegment to do:
// 1) color contribution computation
// 2) BSDF evaluation: generate a new ray
struct ShadeableIntersection {
    float t;
    glm::vec4 tangent = glm::vec4(0, 0, 0, 0);
    glm::vec3 surfaceNormal;
    int materialId;
    int textureId = -1;
    glm::vec2 uv = glm::vec2(-1.f, -1.f); // set to -1 -1 by default
    bool hasHitObj = false;
};
