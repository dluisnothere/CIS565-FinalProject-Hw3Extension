#pragma once

#include <vector>
#include <set>
#include <sstream>
#include <fstream>
#include <iostream>
#include "glm/glm.hpp"
#include "utilities.h"
#include "sceneStructs.h"
#include "tiny_gltf.h"

using namespace std;

class Scene {
private:
    ifstream fp_in;
    void processGLTFNode(const tinygltf::Model& model, const tinygltf::Node& gltf_node, const glm::mat4& parent_matrix, std::vector<Geom>* geom);
    int loadGltf(std::string filename, Geom* transformGeom, Material* sceneMat,/*std::vector<Triangle>* triangleArray, std::vector<Geom>* geom,*/ const char* basepath, bool triangulate);
    int loadObj(const char* filename, std::vector<Triangle>* triangleArray, const char* basepath, bool triangulate);
    int loadMaterial(string materialid);
    int loadTexture(string textureid);
    int loadGeom(string objectid);
    int loadCamera();
public:
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;

    // for textures
    std::vector<Texture> textures;
    // std::vector<int> textureChannels;

    RenderState state;

    std::vector<Geom> lights;
    int numLights = 0;
};
