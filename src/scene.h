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

    //KDTree construction
    bool triCompare(Triangle t1, Triangle t2, int index); //true if t2 > t1 false if t2 < t1
    BoundBox buildBound(BoundBox box, Triangle t1, Triangle t2, int index, bool useNear);
    void createNode(int node_idx, int tri_idx, int parent_idx, BoundBox bound, KDSPLIT split, parentRelation rel); //called to fill a node in the kd node vector
    void pushdown(Triangle* tri_arr, int parent, BoundBox bound, int tri_idx); //called to push a triangel down the tree
public:
    void constructKDTrees();
    Scene(string filename);
    ~Scene();

    std::vector<Geom> geoms;
    std::vector<Material> materials;
    std::vector<int> kdtree_indices; //THIS REFERS TO THE INDEX OF GEOMS THAT HAVE KDTREES
    std::vector<KDNode> vec_kdnode;

    // for textures
    std::vector<Texture> textures;
    // std::vector<int> textureChannels;

    RenderState state;
    Trajectory traj;

    std::vector<Geom> lights;
    int numLights = 0;
    int numGeoms = 0;
};
