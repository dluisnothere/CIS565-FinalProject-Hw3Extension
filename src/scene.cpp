#include <iostream>
#include "scene.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "stb_image.h"
#include "stb_image_write.h"

int MAXDEPTH = 2;
#define TINYGLTF_IMPLEMENTATION
//#define STB_IMAGE_IMPLEMENTATION
//#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_MSC_SECURE_CRT
#define SAR 1

#include "tiny_gltf.h"

#include "tiny_obj_loader.h"

using namespace std;

Scene::Scene(string filename) {
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    char* fname = (char*)filename.c_str();
    fp_in.open(fname);
    if (!fp_in.is_open()) {
        cout << "Error reading from file - aborting!" << endl;
        throw;
    }
    while (fp_in.good()) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "MATERIAL") == 0) {
                loadMaterial(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "OBJECT") == 0) {
                loadGeom(tokens[1]);
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "CAMERA") == 0) {
                loadCamera();
                cout << " " << endl;
            }
            else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
                loadTexture(tokens[1]);
                cout << " " << endl;
            }
        }
    }
    constructKDTrees();
}

void Scene::processGLTFNode(const tinygltf::Model& model, const tinygltf::Node& gltf_node, const glm::mat4& parent_matrix, std::vector<Geom>* geoms)
{
    glm::mat4 node_xform;
    if (gltf_node.matrix.empty()) {
        glm::vec3 t = gltf_node.translation.empty() ? glm::vec3(0.f) : glm::make_vec3(gltf_node.translation.data());
        glm::quat r = gltf_node.rotation.empty() ? glm::quat(1, 0, 0, 0) : glm::make_quat(gltf_node.rotation.data());
        glm::vec3 s = gltf_node.scale.empty() ? glm::vec3(1.f) : glm::make_vec3(gltf_node.scale.data());
        node_xform = utilityCore::buildTransformationMatrix(t, r, s);
    }
    else {
        node_xform = glm::make_mat4(gltf_node.matrix.data());
    }
    node_xform = parent_matrix * node_xform;

    if (!gltf_node.children.empty())
    {
        for (int32_t child : gltf_node.children)
        {
            processGLTFNode(model, model.nodes[child], node_xform, geoms);
        }
    }

    //Geom newGeom;
    if (gltf_node.mesh > -1) {
        int meshNum = gltf_node.mesh;

        Geom *nodeGeom = &(*geoms)[meshNum];
        nodeGeom->transform = node_xform;
        nodeGeom->inverseTransform = glm::inverse(nodeGeom->transform);
        nodeGeom->invTranspose = glm::inverseTranspose(nodeGeom->transform);
    }
}

int Scene::loadGltf(std::string filename, Geom* transformGeom,/*std::vector<Triangle>* triangleArray, std::vector<Geom>* geoms,*/ Material* sceneMat, const char* basepath = NULL, bool triangulate = true) {
    tinygltf::Model model;
    tinygltf::TinyGLTF loader;
    std::string err;
    std::string warn;

    bool ret;
    if (filename.size() >= 4 && strncmp(filename.c_str() + filename.size() - 4, ".glb", 4) == 0)
        ret = loader.LoadBinaryFromFile(&model, &err, &warn, filename);
    else
        ret = loader.LoadASCIIFromFile(&model, &err, &warn, filename);
    if (!warn.empty())
        std::cerr << "glTF WARNING: " << warn << std::endl;
    if (!ret)
    {
        std::cerr << "Failed to load GLTF scene '" << filename << "': " << err << std::endl;
        return -1;
        //throw Exception(err.c_str());
    }

    //
    // Textures
    //
    std::vector<Texture> gltfTextures = std::vector<Texture>();
    int texIdx = 0;
    int texOffset = textures.size();
    for (const tinygltf::Texture& gltfTexture : model.textures) {
        Texture loadedTexture;
        const tinygltf::Image& image = model.images[gltfTexture.source];
        loadedTexture.channels = image.component;  // number of components
        if (loadedTexture.channels != 3 && loadedTexture.channels != 4) {
            std::cerr << "Some error : channels are weird\n" << std::endl;
        }
        loadedTexture.width = image.width;
        loadedTexture.height = image.height;
        int size = image.component * image.width * image.height * sizeof(unsigned char);
        loadedTexture.host_texData = new unsigned char[size];
        memcpy(loadedTexture.host_texData, image.image.data(), size);

        textures.push_back(loadedTexture);
    }

    //
    // Materials
    //
    int gltfMatIdx = 0;
    int matOffset = materials.size();
    std::vector<Material> gltfMaterials = std::vector<Material>();
    for (auto& gltf_material : model.materials)
    {
        std::cerr << "Processing glTF material: '" << gltf_material.name << "'\n";
        Material newMaterial;
        newMaterial.specular = sceneMat->specular;
        newMaterial.hasReflective = sceneMat->hasReflective;
        newMaterial.hasRefractive = sceneMat->hasRefractive;
        newMaterial.indexOfRefraction = sceneMat->indexOfRefraction;
        newMaterial.emittance = sceneMat->emittance;

        {
            const auto roughness_it = gltf_material.values.find("roughnessFactor");
            if (roughness_it != gltf_material.values.end())
            {
                newMaterial.pbrMetallicRoughness.roughnessFactor = static_cast<float>(roughness_it->second.Factor());
                std::cerr << "\tRougness:  " << newMaterial.pbrMetallicRoughness.roughnessFactor << "\n";
            }
            else
            {
                std::cerr << "\tUsing default roughness factor\n";
            }
        }

        {
            const auto metallic_it = gltf_material.values.find("metallicFactor");
            if (metallic_it != gltf_material.values.end())
            {
                newMaterial.pbrMetallicRoughness.metallicFactor = static_cast<float>(metallic_it->second.Factor());
                std::cerr << "\tMetallic:  " << newMaterial.pbrMetallicRoughness.metallicFactor << "\n";
            }
            else
            {
                std::cerr << "\tUsing default metallic factor\n";
            }
        }

        {
            const auto emissive_factor_it = gltf_material.additionalValues.find("emissiveFactor");
            if (emissive_factor_it != gltf_material.additionalValues.end())
            {
                const tinygltf::ColorValue c = emissive_factor_it->second.ColorFactor();
                newMaterial.emissiveFactor = glm::vec3(c[0], c[1], c[2]);
                std::cerr
                    << "\tEmissive factor: ("
                    << newMaterial.emissiveFactor.x << ", "
                    << newMaterial.emissiveFactor.y << ", "
                    << newMaterial.emissiveFactor.z << ")\n";
            }
            else
            {
                std::cerr << "\tUsing default base color factor\n";
            }
        }

        const auto baseColorTexture = gltf_material.values.find("baseColorTexture");
        if (baseColorTexture != gltf_material.values.end()) {
            std::cout << "found pbrMetallicRoughness " << std::endl;
            newMaterial.pbrMetallicRoughness.baseColorIdx = gltf_material.pbrMetallicRoughness.baseColorTexture.index;
            newMaterial.pbrMetallicRoughness.baseColorOffset = texOffset;
            newMaterial.pbrMetallicRoughness.baseColorTexCoord = gltf_material.pbrMetallicRoughness.baseColorTexture.texCoord;
        }

        const auto metallicRoughness = gltf_material.values.find("metallicRoughnessTexture");
        if (metallicRoughness != gltf_material.values.end()) {
            std::cout << "found pbrMetallicRoughness " << std::endl;
            newMaterial.pbrMetallicRoughness.metallicRoughnessIdx = gltf_material.pbrMetallicRoughness.metallicRoughnessTexture.index;
            newMaterial.pbrMetallicRoughness.metallicRoughnessOffset = texOffset;
            newMaterial.pbrMetallicRoughness.metallicRoughnessTexCoord = gltf_material.pbrMetallicRoughness.metallicRoughnessTexture.texCoord;
        }

        const auto emission = gltf_material.additionalValues.find("emissiveTexture");
        if (emission != gltf_material.additionalValues.end()) {
            std::cout << "found emissiveTexture " << std::endl;
            newMaterial.emissiveTexture.index = gltf_material.emissiveTexture.index;
            newMaterial.emissiveTexture.texCoord = gltf_material.emissiveTexture.texCoord;
            newMaterial.emissiveTexture.texOffset = texOffset;
        }

        const auto normal = gltf_material.additionalValues.find("normalTexure");
        if (normal != gltf_material.additionalValues.end()) {
            std::cout << "found normalTexture " << std::endl;
            newMaterial.normalTexture.index = gltf_material.normalTexture.index;
            newMaterial.normalTexture.texCoord = gltf_material.normalTexture.texCoord;
            newMaterial.normalTexture.texOffset = texOffset;
        }

        const auto occlude = gltf_material.additionalValues.find("occlusionTexture");
        if (occlude != gltf_material.additionalValues.end()) {
            std::cout << "found occlusionTexture " << std::endl;
            newMaterial.occlusionTexture.index = gltf_material.occlusionTexture.index;
            newMaterial.occlusionTexture.texCoord = gltf_material.occlusionTexture.texCoord;
            newMaterial.occlusionTexture.texOffset = texOffset;
        }

        gltfMaterials.push_back(newMaterial);

        //gltfMatIdx++;
    }
    //
    // Meshes
    //

    int maxTexCoord = -1;

    std::vector<Geom> gltfGeoms = std::vector<Geom>();
    int gltfGeomIdx = 0;
    int geomOffset = geoms.size();

    for (int midx = 0; midx < model.meshes.size(); midx++)
    {
        std::vector<Triangle> triangleArray = std::vector<Triangle>();
        auto& gltf_mesh = model.meshes[midx];
        std::cerr << "Processing glTF mesh: '" << gltf_mesh.name << "'\n";
        std::cerr << "\tNum mesh primitive groups: " << gltf_mesh.primitives.size() << std::endl;

        for (const tinygltf::Primitive& gltf_primitive : gltf_mesh.primitives)
        {
            std::cout << "primitive encountered" << std::endl;
            if (gltf_primitive.mode != TINYGLTF_MODE_TRIANGLES) // Ignore non-triangle meshes
            {
                std::cerr << "\tNon-triangle primitive: skipping\n";
                continue;
            }

            std::vector<int> tmpIndices;
            std::vector<float4> tmpTangents;
            std::vector<float3> tmpNormals, tmpVertices;
            std::map<int, std::vector<float2>> tmpUvs; // map uvs to texcoords

            const tinygltf::Accessor& idxAccessor = model.accessors[gltf_primitive.indices];
            const tinygltf::BufferView& idxBufferView = model.bufferViews[idxAccessor.bufferView];
            const tinygltf::Buffer& idxBuf = model.buffers[idxBufferView.buffer];

            // idxBuf.data.data() returns an unsigned char
            const uint8_t* a = idxBuf.data.data() + idxBufferView.byteOffset + idxAccessor.byteOffset;
            const int byteStride = idxAccessor.ByteStride(idxBufferView);
            const size_t count = idxAccessor.count;

            // Debugging purposes
            const float scalingFactor = 3;

            switch (idxAccessor.componentType)
            {
            case TINYGLTF_COMPONENT_TYPE_BYTE:
                std::cout << "TINYGLTF_COMPONENT_TYPE_BYTE " << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((char*)a));
                }
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE:
                std::cout << "TINYGLTF_COMPONENT_TYPE_UNSIGNED_BYTE " << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((uint8_t*)a));
                }
                break;
            case TINYGLTF_COMPONENT_TYPE_SHORT:
                std::cout << "TINYGLTF_COMPONENT_TYPE_SHORT " << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((short*)a));
                }
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT:
                std::cout << "TINYGLTF_COMPONENT_TYPE_UNSIGNED_SHORT " << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((uint16_t*)a));
                }
                break;
            case TINYGLTF_COMPONENT_TYPE_INT:
                std::cout << "TINYGLTF_COMPONENT_TYPE_INT " << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((int*)a));
                }
                break;
            case TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT:
                std::cout << "TINYGLTF_COMPONENT_TYPE_UNSIGNED_INT " << tmpIndices.size() / 3 << std::endl;
                for (int k = 0; k < count; k++, a += byteStride) {
                    tmpIndices.push_back(*((uint32_t*)a));
                }
                break;
            default: break;
            }

            for (const auto& attribute : gltf_primitive.attributes)
            {
                //std::cout << "Encountered attribute" << std::endl;
                const tinygltf::Accessor attribAccessor = model.accessors[attribute.second];
                const tinygltf::BufferView& bufferView = model.bufferViews[attribAccessor.bufferView];
                const tinygltf::Buffer& buffer = model.buffers[bufferView.buffer];

                const uint8_t* a = buffer.data.data() + bufferView.byteOffset + attribAccessor.byteOffset;
                const int byte_stride = attribAccessor.ByteStride(bufferView);

                const size_t count = attribAccessor.count;
                if (attribute.first == "POSITION")
                {
                    //std::cout << "Encountered position" << count << std::endl;
                    if (attribAccessor.type == TINYGLTF_TYPE_VEC3) {
                        if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                            for (size_t i = 0; i < count; i++, a += byte_stride) {
                                //std::cout << "push position" << std::endl;
                                tmpVertices.push_back(*((float3*)a));
                            }
                        }
                        else {
                            std::cerr << "double precision positions not supported in gltf file" << std::endl;
                        }
                    }
                    else
                    {
                        std::cerr << "unsupported position definition in gltf file" << std::endl;
                    }
                }
                else if (attribute.first == "NORMAL") {
                    //std::cout << "Encountered normal" << count << std::endl;
                    if (attribAccessor.type == TINYGLTF_TYPE_VEC3) {
                        if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                            for (size_t i = 0; i < count; i++, a += byte_stride) {
                                //std::cout << "push normal" << std::endl;
                                tmpNormals.push_back(*((float3*)a));
                            }
                        }
                        else {
                            std::cerr << "double precision normals not supported in gltf file" << std::endl;
                        }
                    }
                    else
                    {
                        std::cerr << "unsupported position definition in gltf file" << std::endl;
                    }
                }
                else if (attribute.first == "TEXCOORD_0") {
                    //std::cout << "Encountered texcoord:"  << count << std::endl;
                    if (attribAccessor.type == TINYGLTF_TYPE_VEC2) {
                        if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                            //std::string attribName = attribute.first;
                            //char* coordNumStr = strtok(const_cast<char*>(attribName.c_str()), "_");
                            //int coordNum = atoi(coordNumStr);
                            for (size_t i = 0; i < count; i++, a += byte_stride) {
                                //std::cout << "push texcoord" << std::endl;
                                tmpUvs[0].push_back(*((float2*)a)); //texture a texture assigned texcoord 1 with this uv.
                                //std::cout << "coordNUm: " << coordNum << "contains: " << ((float2*)a)->x << " , " << ((float2*)a)->y << std::endl;
                                //maxTexCoord = std::max(coordNum, maxTexCoord);
                            }
                        }
                        else {
                            std::cerr << "double precision texcoords not supported in gltf file" << std::endl;
                        }
                    }
                    else
                    {
                        std::cerr << "unsupported position definition in gltf file" << std::endl;
                    }
                }
                //else if (strstr(attribute.first.c_str(), "TEXCOORD_")) {
                //    if (attribAccessor.type == TINYGLTF_TYPE_VEC2) {
                //        if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                //            std::string attribName = attribute.first;
                //            char* coordNumStr = strtok(const_cast<char*>(attribName.c_str()), "_");
                //            int coordNum = atoi(coordNumStr);
                //            for (size_t i = 0; i < count; i++, a += byte_stride) {
                //                tmpUvs[coordNum].push_back(*((float2*)a)); //texture a texture assigned texcoord 1 with this uv.
                //                std::cout << "coordNUm: " << coordNum << "contains: " << ((float2*)a)->x << " , " << ((float2*)a)->y << std::endl;
                //                maxTexCoord = std::max(coordNum, maxTexCoord);
                //            }
                //        }
                //        else {
                //            std::cerr << "double precision texcoords not supported in gltf file" << std::endl;
                //        }
                //    }
                //    else
                //    {
                //        std::cerr << "unsupported position definition in gltf file" << std::endl;
                //    }
                //}
                /*else if (attribute.first == "TANGENT") {
                    std::cout << "Encountered tangent: " << count << std::endl;
                    if (attribAccessor.type == TINYGLTF_TYPE_VEC4) {
                        if (attribAccessor.componentType == TINYGLTF_COMPONENT_TYPE_FLOAT) {
                            for (size_t i = 0; i < count; i++, a += byte_stride) {
                                std::cout << "push tangent" << std::endl;
                                tmpTangents.push_back(*((float4*)a));
                            }
                        }
                        else {
                            std::cerr << "double precision texcoords not supported in gltf file" << std::endl;
                        }
                    }
                    else
                    {
                        std::cerr << "unsupported position definition in gltf file" << std::endl;
                    }
                }*/

            }

            std::cerr << "\t\tNum triangles: " << tmpIndices.size() / 3 << std::endl;
            std::cerr << "\t\tNum verts: " << tmpVertices.size() << std::endl;

            const size_t triangleCount = tmpIndices.size() / 3;
            for (size_t i = 0; i < triangleCount; i++) {
                //std::cout << i << std::endl;
                const uint32_t aIdx = tmpIndices[i * 3 + 0];
                const uint32_t bIdx = tmpIndices[i * 3 + 1];
                const uint32_t cIdx = tmpIndices[i * 3 + 2];

                float3 pa = tmpVertices[aIdx];
                float3 pb = tmpVertices[bIdx];
                float3 pc = tmpVertices[cIdx];

                float3 na = tmpNormals[aIdx];
                float3 nb = tmpNormals[bIdx];
                float3 nc = tmpNormals[cIdx];

                //float4 ta = tmpTangents[aIdx];
                //float4 tb = tmpTangents[bIdx];
                //float4 tc = tmpTangents[cIdx];

                float2 ua = tmpUvs[0][aIdx];
                float2 ub = tmpUvs[0][bIdx];
                float2 uc = tmpUvs[0][cIdx];

                const glm::vec4 aPos = glm::vec4( pa.x, pa.y, pa.z, 1 );
                const glm::vec4 bPos = glm::vec4( pb.x, pb.y, pb.z, 1 );
                const glm::vec4 cPos = glm::vec4( pc.x, pc.y, pc.z, 1 );

                const glm::vec4 aNor = glm::vec4(na.x, na.y, na.z, 0);
                const glm::vec4 bNor = glm::vec4(nb.x, nb.y, nb.z, 0);
                const glm::vec4 cNor = glm::vec4(nc.x, nc.y, nc.z, 0);

                //const glm::vec4 aTan = glm::vec4(ta.x, ta.y, ta.z, ta.w);
                //const glm::vec4 bTan = glm::vec4(tb.x, tb.y, tb.z, tb.w);
                //const glm::vec4 cTan = glm::vec4(tc.x, tc.y, tc.z, tc.w);

                const glm::vec2 aUv = glm::vec2(ua.x, ua.y);
                const glm::vec2 bUv = glm::vec2(ub.x, ub.y);
                const glm::vec2 cUv = glm::vec2(uc.x, uc.y);

                //// separate uvs from everything else bc it's a different process.

                std::vector<glm::vec2> aUvList = std::vector<glm::vec2>();
                std::vector<int> aTexCoord = std::vector<int>();
                
                aUvList.push_back(aUv);
                aTexCoord.push_back(0);

                std::vector<glm::vec2> bUvList = std::vector<glm::vec2>();
                std::vector<int> bTexCoord = std::vector<int>();
                bUvList.push_back(bUv);
                bTexCoord.push_back(0);

                std::vector<glm::vec2> cUvList = std::vector<glm::vec2>();
                std::vector<int> cTexCoord = std::vector<int>();

                cUvList.push_back(cUv);
                cTexCoord.push_back(0);

                //// separate Uvs from everything else

                Vertex vertA = Vertex{ aPos, aNor, aTexCoord, aUvList/*, aTan*/};
                Vertex vertB = Vertex{ bPos, bNor, bTexCoord, bUvList/* bTan*/ };
                Vertex vertC = Vertex{ cPos, cNor, cTexCoord, cUvList/*, cTan*/};

                Triangle triangle = {
                    vertA,
                    vertB,
                    vertC
                };

                triangleArray.push_back(triangle);

            }

            std::cout << "triangles pushed" << std::endl;

            //std::cerr << "\t\tNum triangles: " << sceneMeshPositions.size() / 3 << std::endl;
            // worry about bounding box performance later

            //// i have a funny feeling in normal buffer that the normals are calculated dynamically through the cuda.
            //auto normal_accessor_iter = gltf_primitive.attributes.find("NORMAL");
            //if (normal_accessor_iter != gltf_primitive.attributes.end())
            //{
            //    std::cerr << "\t\tHas vertex normals: true\n";
            //    normalBuffer.push_back(bufferViewFromGLTF<float3>(model, normal_accessor_iter->second));
            //}
            //else
            //{
            //    std::cerr << "\t\tHas vertex normals: false\n";
            //    normalBuffer.push_back(bufferViewFromGLTF<float3>(model, -1));
            //}

            //for (size_t j = 0; j < GeometryData::num_textcoords; ++j)
            ////{
            //    char texcoord_str[128];
            //    //snprintf(texcoord_str, 128, "TEXCOORD_0", (int)j);
            //    auto texcoord_accessor_iter = gltf_primitive.attributes.find(texcoord_str);
            //    if (texcoord_accessor_iter != gltf_primitive.attributes.end())
            //    {
            //        std::cerr << "\t\tHas texcoords_" << j << ": true\n";
            //        mesh->texcoords[j].push_back(bufferViewFromGLTF<Vec2f>(model, scene, texcoord_accessor_iter->second));
            //    }
            //    else
            //    {
            //        std::cerr << "\t\tHas texcoords_" << j << ": false\n";
            //        mesh->texcoords[j].push_back(bufferViewFromGLTF<Vec2f>(model, scene, -1));
            //    }
            //}

            /*auto color_accessor_iter = gltf_primitive.attributes.find("COLOR_0");
            if (color_accessor_iter != gltf_primitive.attributes.end())
            {
                std::cerr << "\t\tHas color_0: true\n";
                mesh->colors.push_back(bufferViewFromGLTF<Vec4f>(model, scene, color_accessor_iter->second));
            }
            else
            {
                std::cerr << "\t\tHas color_0: false\n";
                mesh->colors.push_back(bufferViewFromGLTF<Vec4f>(model, scene, -1));
            }*/
            float xMin = FLT_MAX;
            float yMin = FLT_MAX;
            float zMin = FLT_MAX;
            float xMax = FLT_MIN;
            float yMax = FLT_MIN;
            float zMax = FLT_MIN;

            for (int i = 0; i < triangleArray.size(); i++) {
                // jank code to find the min and max of the box
                Triangle tri = triangleArray[i];

                xMin = fmin(fmin(tri.pointA.pos[0], tri.pointB.pos[0]), fmin(tri.pointC.pos[0], xMin));
                xMax = fmax(fmax(tri.pointA.pos[0], tri.pointB.pos[0]), fmax(tri.pointC.pos[0], xMax));

                yMin = fmin(fmin(tri.pointA.pos[1], tri.pointB.pos[1]), fmin(tri.pointC.pos[1], yMin));
                yMax = fmax(fmax(tri.pointA.pos[1], tri.pointB.pos[1]), fmax(tri.pointC.pos[1], yMax));

                zMin = fmin(fmin(tri.pointA.pos[2], tri.pointB.pos[2]), fmin(tri.pointC.pos[2], zMin));
                zMax = fmax(fmax(tri.pointA.pos[2], tri.pointB.pos[2]), fmax(tri.pointC.pos[2], zMax));
            }

            BoundBox box = {
                glm::vec3(xMin, yMin, zMin),
                glm::vec3(xMax, yMax, zMax)
            };

            cout << "xMin: " << xMin << " , " << yMin << " , " << zMin << endl;
            cout << "xMax: " << xMax << " , " << yMax << " , " << zMax << endl;

            Geom newGeom;
            newGeom.type = GLTF;
            newGeom.materialid = gltf_primitive.material;
            newGeom.materialOffset = matOffset;
            newGeom.host_tris = new Triangle[triangleArray.size()];
            newGeom.device_tris = NULL;
            newGeom.numTris = triangleArray.size();

            newGeom.bound = box;
            for (int i = 0; i < triangleArray.size(); i++) {
                newGeom.host_tris[i] = triangleArray[i];
            }

            gltfGeoms.push_back(newGeom);
        }
        
    }

    printf("MATRIX %i\n", transformGeom->transform.length());
    for (int i = 0; i < 4; i++) {
        printf("%f, %f, %f, %f \n", transformGeom->transform[i][0], transformGeom->transform[i][1], transformGeom->transform[i][2], transformGeom->transform[i][3]);
    }
    ////
    //// Process nodes's transforms
    ////
    std::vector<int32_t> root_nodes(model.nodes.size(), 1);
    for (auto& gltf_node : model.nodes)
        for (int32_t child : gltf_node.children)
            root_nodes[child] = 0;

    for (size_t i = 0; i < root_nodes.size(); ++i)
    {
        if (!root_nodes[i])
            continue;
        auto& gltf_node = model.nodes[i];

        processGLTFNode(model, gltf_node, transformGeom->transform, &gltfGeoms);
    }

    for (int i = 0; i < gltfGeoms.size(); i++) {
        kdtree_indices.push_back(geoms.size());
        geoms.push_back(gltfGeoms[i]);       
    }

    for (int i = 0; i < gltfMaterials.size(); i++) {
        materials.push_back(gltfMaterials[i]);
    }

    for (int i = 0; i < gltfTextures.size(); i++) {
        textures.push_back(gltfTextures[i]);
    }

}

#if LOAD_OBJ || LOAD_GLTF
// example code taken from https://github.com/tinyobjloader/tinyobjloader
int Scene::loadObj(const char* filename, 
    std::vector<Triangle>* triangleArray,
    const char* basepath = NULL,
    bool triangulate = true)
{
    tinyobj::attrib_t attrib;
    std::vector<tinyobj::shape_t> shapes;
    std::vector<tinyobj::material_t> materials;
    std::string warn;
    std::string err;

    tinyobj::ObjReaderConfig reader_config;
    reader_config.mtl_search_path = "../Project3-CUDA-Path-Tracer/objs/"; // Path to material files
    tinyobj::ObjReader reader;

    if (!reader.ParseFromFile(filename, reader_config)) {
        if (!reader.Error().empty()) {
            std::cerr << "TinyObjReader error: " << reader.Error();
        }
        exit(1);
    }

    if (!reader.Warning().empty()) {
        std::cout << "TinyObjReader warning: " << reader.Warning();
    }

    attrib = reader.GetAttrib();
    shapes = reader.GetShapes();
    // materials = reader.GetMaterials();

    //cout << "material name: " << materials[0].name << std::endl;
    // cout << materials.size() << std::endl;

    // Loop over shapes and load each attrib
    for (size_t s = 0; s < shapes.size(); s++) {

        // Loop over faces(polygon)
        size_t index_offset = 0;
        for (size_t f = 0; f < shapes[s].mesh.num_face_vertices.size(); f++) {
            size_t fv = size_t(shapes[s].mesh.num_face_vertices[f]);
            // Loop over vertices in the face.
            for (size_t v = 1; v < fv - 1; v++) {
                // access to vertex

                // idxa is the primary vertex's idx
                tinyobj::index_t idxa = shapes[s].mesh.indices[index_offset];
                tinyobj::index_t idxb = shapes[s].mesh.indices[index_offset + v];
                tinyobj::index_t idxc = shapes[s].mesh.indices[index_offset + v + 1];

                tinyobj::real_t vxa = attrib.vertices[3 * size_t(idxa.vertex_index) + 0];
                tinyobj::real_t vya = attrib.vertices[3 * size_t(idxa.vertex_index) + 1];
                tinyobj::real_t vza = attrib.vertices[3 * size_t(idxa.vertex_index) + 2];

                tinyobj::real_t vxb = attrib.vertices[3 * size_t(idxb.vertex_index) + 0];
                tinyobj::real_t vyb = attrib.vertices[3 * size_t(idxb.vertex_index) + 1];
                tinyobj::real_t vzb = attrib.vertices[3 * size_t(idxb.vertex_index) + 2];

                tinyobj::real_t vxc = attrib.vertices[3 * size_t(idxc.vertex_index) + 0];
                tinyobj::real_t vyc = attrib.vertices[3 * size_t(idxc.vertex_index) + 1];
                tinyobj::real_t vzc = attrib.vertices[3 * size_t(idxc.vertex_index) + 2];


                tinyobj::real_t nxa = attrib.normals[3 * size_t(idxa.normal_index) + 0];
                tinyobj::real_t nya = attrib.normals[3 * size_t(idxa.normal_index) + 1];
                tinyobj::real_t nza = attrib.normals[3 * size_t(idxa.normal_index) + 2];

                tinyobj::real_t nxb = attrib.normals[3 * size_t(idxb.normal_index) + 0];
                tinyobj::real_t nyb = attrib.normals[3 * size_t(idxb.normal_index) + 1];
                tinyobj::real_t nzb = attrib.normals[3 * size_t(idxb.normal_index) + 2];

                tinyobj::real_t nxc = attrib.normals[3 * size_t(idxc.normal_index) + 0];
                tinyobj::real_t nyc = attrib.normals[3 * size_t(idxc.normal_index) + 1];
                tinyobj::real_t nzc = attrib.normals[3 * size_t(idxc.normal_index) + 2];

                // construct triangle object
                Vertex vertA = {
                    glm::vec4(vxa, vya, vza, 1),
                    glm::vec4(nxa, nya, nza, 0)
                };

                Vertex vertB = {
                    glm::vec4(vxb, vyb, vzb, 1),
                    glm::vec4(nxb, nyb, nzb, 0)
                };

                Vertex vertC = {
                    glm::vec4(vxc, vyc, vzc, 1),
                    glm::vec4(nxc, nyc, nzc, 0)
                };

#if USE_UV
                // Check if `texcoord_index` is zero or positive. negative = no texcoord data
                // don't texture yet
                if (idxa.texcoord_index >= 0) {
                    tinyobj::real_t txa = attrib.texcoords[2 * size_t(idxa.texcoord_index) + 0];
                    tinyobj::real_t tya = attrib.texcoords[2 * size_t(idxa.texcoord_index) + 1];
                    // vertA.hasUv = true;
                    vertA.host_uvs.push_back(glm::vec2(txa, tya));
                }
                if (idxb.texcoord_index >= 0) {
                    tinyobj::real_t txb = attrib.texcoords[2 * size_t(idxb.texcoord_index) + 0];
                    tinyobj::real_t tyb = attrib.texcoords[2 * size_t(idxb.texcoord_index) + 1];
                    // vertB.hasUv = true;
                    vertB.host_uvs.push_back(glm::vec2(txb, tyb));
                }
                if (idxc.texcoord_index >= 0) {
                    tinyobj::real_t txc = attrib.texcoords[2 * size_t(idxc.texcoord_index) + 0];
                    tinyobj::real_t tyc = attrib.texcoords[2 * size_t(idxc.texcoord_index) + 1];
                    // vertC.hasUv = true;
                    vertC.host_uvs.push_back(glm::vec2(txc, tyc));
                }

#endif

                Triangle triangle = {
                    vertA,
                    vertB,
                    vertC
                };

                triangleArray->push_back(triangle);

            }
            index_offset += fv;

            // shapes[s].mesh.material_ids[f];
        }
    }

    return true;
}

int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        bool hasObj = false;
        bool hasGltf = false;
        string objFileName;

        //load object from obj file
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
            else if (strstr(line.c_str(), ".obj") != NULL) {
                cout << "Creating some obj..." << endl;
#if USE_BOUND_BOX
                newGeom.type = OBJ;
#else
                newGeom.type = TRIANGLE;
#endif

                hasObj = true;
                objFileName = line;
            }
            else if (strstr(line.c_str(), ".gltf") != NULL) {
                cout << "creating some gltf... " << endl;
#if USE_BOUND_BOX
                // USELESS VARS BC ALL GEOMS ARE CREATED INISIDE OTHER FUNCTIONS FOR GLTF
                newGeom.type = GLTF;
#else
                newGeom.type = TRIANGLE;
#endif
                hasGltf = true;
                objFileName = line;
            }
            else {
                cout << "wtf is this??" << std::endl;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations and texture
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "TEXTURE") == 0) {
                cout << "finding Texture... " << endl;
#if LOAD_OBJ
                newGeom.objTexId = atof(tokens[1].c_str());
#endif
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (hasObj) {
            std::vector<Triangle> triangleArray;
            loadObj(objFileName.c_str(), &triangleArray);
#if USE_BOUND_BOX
            float xMin = FLT_MAX;
            float yMin = FLT_MAX;
            float zMin = FLT_MAX;
            float xMax = FLT_MIN;
            float yMax = FLT_MIN;
            float zMax = FLT_MIN;

            for (int i = 0; i < triangleArray.size(); i++) {
                // jank code to find the min and max of the box
                Triangle tri = triangleArray[i];

                xMin = fmin(fmin(tri.pointA.pos[0], tri.pointB.pos[0]), fmin(tri.pointC.pos[0], xMin));
                xMax = fmax(fmax(tri.pointA.pos[0], tri.pointB.pos[0]), fmax(tri.pointC.pos[0], xMax));

                yMin = fmin(fmin(tri.pointA.pos[1], tri.pointB.pos[1]), fmin(tri.pointC.pos[1], yMin));
                yMax = fmax(fmax(tri.pointA.pos[1], tri.pointB.pos[1]), fmax(tri.pointC.pos[1], yMax));

                zMin = fmin(fmin(tri.pointA.pos[2], tri.pointB.pos[2]), fmin(tri.pointC.pos[2], zMin));
                zMax = fmax(fmax(tri.pointA.pos[2], tri.pointB.pos[2]), fmax(tri.pointC.pos[2], zMax));
            }

            BoundBox box = {
                glm::vec3(xMin, yMin, zMin),
                glm::vec3(xMax, yMax, zMax)
            };

            cout << "xMin: " << xMin << " , " << yMin << " , " << zMin << endl;
            cout << "xMax: " << xMax << " , " << yMax << " , " << zMax << endl;


            newGeom.host_tris = new Triangle[triangleArray.size()];//triangleArray.size()];
            newGeom.device_tris = NULL;
            newGeom.numTris = triangleArray.size();

            newGeom.bound = box;
            for (int i = 0; i < triangleArray.size(); i++) {
                newGeom.host_tris[i] = triangleArray[i];
            }

            int gltf_idx = geoms.size();
            geoms.push_back(newGeom);
            kdtree_indices.push_back(gltf_idx);

#else 
            // create geoms from triangles using newGeom properties
            // load triangles into the geoms scene.
            for (int i = 0; i < triangleArray.size(); i ++) {
                // there should only be 1 triangle
                Triangle* trisInGeom = new Triangle(triangleArray[i]);

                // just a single triangle
                Geom newTriGeom = {
                    TRIANGLE,
                    newGeom.materialid,
                    newGeom.translation,
                    newGeom.rotation,
                    newGeom.scale,
                    newGeom.transform,
                    newGeom.inverseTransform,
                    newGeom.invTranspose,
                    trisInGeom,
                    NULL, // device pointer is not yet allocated
                    BoundBox {
                        },
                    1,
                };
                geoms.push_back(newTriGeom);
            }
#endif
        }
        else if (hasGltf) {
            //std::vector<Triangle> triangleArray;
            loadGltf(objFileName, &newGeom, &materials[newGeom.materialid]/*, &triangleArray, &geoms*/ ); // transforms are set in here.

#if USE_BOUND_BOX
            /*float xMin = FLT_MAX;
            float yMin = FLT_MAX;
            float zMin = FLT_MAX;
            float xMax = FLT_MIN;
            float yMax = FLT_MIN;
            float zMax = FLT_MIN;*/

            /*for (int i = 0; i < triangleArray.size(); i++) {
                // jank code to find the min and max of the box
                Triangle tri = triangleArray[i];

                xMin = fmin(fmin(tri.pointA.pos[0], tri.pointB.pos[0]), fmin(tri.pointC.pos[0], xMin));
                xMax = fmax(fmax(tri.pointA.pos[0], tri.pointB.pos[0]), fmax(tri.pointC.pos[0], xMax));

                yMin = fmin(fmin(tri.pointA.pos[1], tri.pointB.pos[1]), fmin(tri.pointC.pos[1], yMin));
                yMax = fmax(fmax(tri.pointA.pos[1], tri.pointB.pos[1]), fmax(tri.pointC.pos[1], yMax));

                zMin = fmin(fmin(tri.pointA.pos[2], tri.pointB.pos[2]), fmin(tri.pointC.pos[2], zMin));
                zMax = fmax(fmax(tri.pointA.pos[2], tri.pointB.pos[2]), fmax(tri.pointC.pos[2], zMax));
            }

            BoundBox box = {
                glm::vec3(xMin, yMin, zMin),
                glm::vec3(xMax, yMax, zMax)
            };*/

            //cout << "xMin: " << xMin << " , " << yMin << " , " << zMin << endl;
            //cout << "xMax: " << xMax << " , " << yMax << " , " << zMax << endl;


            //newGeom.host_tris = new Triangle[triangleArray.size()];//triangleArray.size()];
            //newGeom.device_tris = NULL;
            //newGeom.numTris = triangleArray.size();

            //newGeom.bound = box;
            /*for (int i = 0; i < triangleArray.size(); i++) {
                newGeom.host_tris[i] = triangleArray[i];
            }*/

            /*geoms.push_back(newGeom);*/

#else 
            // create geoms from triangles using newGeom properties
            // load triangles into the geoms scene.
            for (int i = 0; i < triangleArray.size(); i++) {
                // there should only be 1 triangle
                Triangle* trisInGeom = new Triangle(triangleArray[i]);

                // just a single triangle
                Geom newTriGeom = {
                    TRIANGLE,
                    newGeom.materialid,
                    newGeom.translation,
                    newGeom.rotation,
                    newGeom.scale,
                    newGeom.transform,
                    newGeom.inverseTransform,
                    newGeom.invTranspose,
                    trisInGeom,
                    NULL, // device pointer is not yet allocated
                    BoundBox {
                        },
                    1,
                };
                geoms.push_back(newTriGeom);
            }
#endif
        }
        else {
            if (newGeom.materialid == 0)
            {
                // materialid == 0 should always be a light
                lights.push_back(newGeom);
                numLights++;
            }
            geoms.push_back(newGeom);
        }
        return 1;
    }
}

#else
int Scene::loadGeom(string objectid) {
    int id = atoi(objectid.c_str());
    if (id != geoms.size()) {
        cout << "ERROR: OBJECT ID does not match expected number of geoms" << endl;
        return -1;
    }
    else {
        cout << "Loading Geom " << id << "..." << endl;
        Geom newGeom;
        string line;

        //load object type
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            if (strcmp(line.c_str(), "sphere") == 0) {
                cout << "Creating new sphere..." << endl;
                newGeom.type = SPHERE;
            }
            else if (strcmp(line.c_str(), "cube") == 0) {
                cout << "Creating new cube..." << endl;
                newGeom.type = CUBE;
            }
        }

        //link material
        utilityCore::safeGetline(fp_in, line);
        if (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);
            newGeom.materialid = atoi(tokens[1].c_str());
            cout << "Connecting Geom " << objectid << " to Material " << newGeom.materialid << "..." << endl;
        }

        //load transformations
        utilityCore::safeGetline(fp_in, line);
        while (!line.empty() && fp_in.good()) {
            vector<string> tokens = utilityCore::tokenizeString(line);

            //load tranformations
            if (strcmp(tokens[0].c_str(), "TRANS") == 0) {
                newGeom.translation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "ROTAT") == 0) {
                newGeom.rotation = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }
            else if (strcmp(tokens[0].c_str(), "SCALE") == 0) {
                newGeom.scale = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
            }

            utilityCore::safeGetline(fp_in, line);
        }

        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        if (newGeom.materialid == 0)
        {
            // materialid == 0 should always be a light
            lights.push_back(newGeom);
            numLights++;
        }

        geoms.push_back(newGeom);
        return 1;
    }
}
#endif

int Scene::loadCamera() {
    cout << "Loading Camera ..." << endl;
    RenderState& state = this->state;
    Camera& camera = state.camera;
    float fovy;

    //load static properties
    for (int i = 0; i < 5; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "RES") == 0) {
            camera.resolution.x = atoi(tokens[1].c_str());
            camera.resolution.y = atoi(tokens[2].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FOVY") == 0) {
            fovy = atof(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "ITERATIONS") == 0) {
            state.iterations = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "DEPTH") == 0) {
            state.traceDepth = atoi(tokens[1].c_str());
        }
        else if (strcmp(tokens[0].c_str(), "FILE") == 0) {
            state.imageName = tokens[1];
        }
    }

    string line;
    utilityCore::safeGetline(fp_in, line);
    while (!line.empty() && fp_in.good()) {
        vector<string> tokens = utilityCore::tokenizeString(line);
        if (strcmp(tokens[0].c_str(), "EYE") == 0) {
            camera.position = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "LOOKAT") == 0) {
            camera.lookAt = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }
        else if (strcmp(tokens[0].c_str(), "UP") == 0) {
            camera.up = glm::vec3(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
        }

        utilityCore::safeGetline(fp_in, line);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());

    cout << "Loaded camera!" << endl;
    return 1;
}

int Scene::loadMaterial(string materialid) {
    int id = atoi(materialid.c_str());
    if (id != materials.size()) {
        cout << "ERROR: MATERIAL ID does not match expected number of materials" << endl;
        return -1;
    }
    else {
        cout << "Loading Material " << id << "..." << endl;
        Material newMaterial;

        //load static properties
        for (int i = 0; i < 7; i++) {
            string line;
            utilityCore::safeGetline(fp_in, line);
            vector<string> tokens = utilityCore::tokenizeString(line);
            if (strcmp(tokens[0].c_str(), "RGB") == 0) {
                glm::vec3 color(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.color = color;
            }
            else if (strcmp(tokens[0].c_str(), "SPECEX") == 0) {
                newMaterial.specular.exponent = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "SPECRGB") == 0) {
                glm::vec3 specColor(atof(tokens[1].c_str()), atof(tokens[2].c_str()), atof(tokens[3].c_str()));
                newMaterial.specular.color = specColor;
            }
            else if (strcmp(tokens[0].c_str(), "REFL") == 0) {
                newMaterial.hasReflective = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFR") == 0) {
                newMaterial.hasRefractive = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "REFRIOR") == 0) {
                newMaterial.indexOfRefraction = atof(tokens[1].c_str());
            }
            else if (strcmp(tokens[0].c_str(), "EMITTANCE") == 0) {
                newMaterial.emittance = atof(tokens[1].c_str());
            }
        }
        materials.push_back(newMaterial);
        return 1;
    }
}

int Scene::loadTexture(string textureid) {
    int id = atoi(textureid.c_str());
    cout << "Loading Texture " << id << "..." << endl;
    Texture newTexture;
    unsigned char* pixelData;
    int width, height, channels;

    if (id != textures.size()) {
        cout << "ERROR: TEXTURE ID does not match expected number of textures" << endl;
        return -1;
    }
    //load static properties
    for (int i = 0; i < 1; i++) {
        string line;
        utilityCore::safeGetline(fp_in, line);
        vector<string> tokens = utilityCore::tokenizeString(line);

        if (strcmp(tokens[0].c_str(), "PATH") == 0) {
            const char* filepath = tokens[1].c_str();
            unsigned char* data = stbi_load(filepath, &width, &height, &channels, 0);

            // pixelData = new glm::vec3[width * height];
            pixelData = new unsigned char[width * height * channels * sizeof(unsigned char)];
            // pixelData = new unsigned char[width * height * 4 * sizeof(unsigned char)]

            // ... process data if not NULL ..
            if (data != nullptr && width > 0 && height > 0)
            {
                cout << "channels: " << channels << endl;
                if (channels == 3)
                {
                    // int pixelDataIdx = 0;
                    // iterate over every pixel
                    // total number of data points is width * height * channels (should be 3)
                    // for (int p = 0; p < (width * height) * channels - 2; p += 3) {
                        /*glm::vec3 currPix = glm::vec3(static_cast<float>(data[p]) / 256.f,
                            static_cast<float>(data[p + 1]) / 256.f,
                            static_cast<float>(data[p + 2]) / 256.f);
                        pixelData[pixelDataIdx] = currPix;
                        
                        pixelDataIdx++;*/
                    // }
                    memcpy(pixelData, data, width * height * channels * sizeof(unsigned char));

                    newTexture.width = width;
                    newTexture.height = height;
                    newTexture.channels = channels;

                    newTexture.host_texData = pixelData;
                    // newTexture.host_texImage = pixelData;

                    cout << "Loaded all Texture Points" << endl;
                    cout << "width: " << newTexture.width; // looks good
                    cout << "height: " << newTexture.height; // looks good
                    // cout << "last pixelIdx: " << pixelDataIdx; // looks correct
                }
                else if (channels == 4) {
                    // rgba
                    //int pixelDataIdx = 0;
                    // iterate over every pixel
                    // total number of data points is width * height * channels (should be 3)
                    //for (int p = 0; p < (width * height) * channels - 3; p += 4) {
                        /*glm::vec3 currPix = glm::vec3(static_cast<float>(data[p]) / 256.f,
                            static_cast<float>(data[p + 1]) / 256.f,
                            static_cast<float>(data[p + 2]) / 256.f);
                        pixelData[pixelDataIdx] = currPix;

                        pixelDataIdx++;*/

                    //}

                    memcpy(pixelData, data, width * height * channels * sizeof(unsigned char));

                    newTexture.width = width;
                    newTexture.height = height;
                    newTexture.channels = channels;

                    newTexture.host_texData = pixelData;

                    // newTexture.host_texImage = pixelData;

                    cout << "Loaded all Texture Points" << endl;
                    cout << "width: " << newTexture.width; // looks good
                    cout << "height: " << newTexture.height; // looks good
                    // cout << "last pixelIdx: " << pixelDataIdx; // looks correct
                }
            }
            else
            {
                std::cout << "Some error: channels are weird\n";
                return -1;
            }

            stbi_image_free(data);
        }
    }

    textures.push_back(newTexture);
    return 1;
}

static int counterForTri = 0;
bool Scene::triCompare(Triangle t1, Triangle t2, int index) {
    float t1_min = fmin(fmin(t1.pointA.pos[index], t1.pointB.pos[index]), t1.pointC.pos[index]);
    float t2_min = fmin(fmin(t2.pointA.pos[index], t2.pointB.pos[index]), t2.pointC.pos[index]);

    float t1_max = fmax(fmax(t1.pointA.pos[index], t1.pointB.pos[index]), t1.pointC.pos[index]);
    float t2_max = fmax(fmax(t2.pointA.pos[index], t2.pointB.pos[index]), t2.pointC.pos[index]);

    //printf("min %f max %f count: %d\n", t2_min, t2_max, counterForTri);
    //counterForTri++;
    if (t2_min > t1_min) {
        return true;
    }
    if (t2_max <= t2_max) {
        return false;
    }

    /*if (0 > t1_min) {
        return true;
    }
    if (0 <= t2_max) {
        return false;
    }*/

    

    //If neither are true then just take an avg
    float t1_avg = (t1.pointA.pos[index] + t1.pointB.pos[index] + t1.pointC.pos[index]) / 3.f;
    float t2_avg = (t2.pointA.pos[index] + t2.pointB.pos[index] + t2.pointC.pos[index]) / 3.f;

    if (t2_avg > t1_avg) {
        return true;
    }
    return false;
}

glm::vec3 Scene::triMin(Triangle t) {
    return glm::vec3(fmin(t.pointA.pos.x, fmin(t.pointB.pos.x, t.pointC.pos.x)),
                     fmin(t.pointA.pos.y, fmin(t.pointB.pos.y, t.pointC.pos.y)),
                     fmin(t.pointA.pos.z, fmin(t.pointB.pos.z, t.pointC.pos.z)));
}
glm::vec3 Scene::triMax(Triangle t) {
    return glm::vec3(fmax(t.pointA.pos.x, fmax(t.pointB.pos.x, t.pointC.pos.x)),
                     fmax(t.pointA.pos.y, fmax(t.pointB.pos.y, t.pointC.pos.y)),
                     fmax(t.pointA.pos.z, fmax(t.pointB.pos.z, t.pointC.pos.z)));
}

BoundBox Scene::buildBound(BoundBox box, Triangle t1, Triangle t2, int index, bool useNear) {
    BoundBox new_box = box;
    if (useNear) {
        float t1_max = fmax(fmax(t1.pointA.pos[index], t1.pointB.pos[index]), t1.pointC.pos[index]);
        float t2_max = fmax(fmax(t2.pointA.pos[index], t2.pointB.pos[index]), t2.pointC.pos[index]);
        float max = fmax(t1_max, t2_max);
        new_box.maxCorner[index] = max;
    }
    else {
        float t1_min = fmin(fmin(t1.pointA.pos[index], t1.pointB.pos[index]), t1.pointC.pos[index]);
        float t2_min = fmin(fmin(t2.pointA.pos[index], t2.pointB.pos[index]), t2.pointC.pos[index]);
        float min = fmin(t1_min, t2_min);
        new_box.minCorner[index] = min;
    }
    return new_box;
}

void Scene::buildBounds(Triangle* tri_arr, int parent, glm::vec3& min, glm::vec3& max) {
    KDNode& node = vec_kdnode[parent];

    glm::vec3 local_min = triMin(tri_arr[node.trisIndex]);
    glm::vec3 local_max = triMax(tri_arr[node.trisIndex]);

#if USE_KD_VEC
    for (int i = 0; i < vec_kdnode[parent].tempBuffer.size(); i++) { //KD_DEBUG
        glm::vec3 vec_min = triMin(tri_arr[vec_kdnode[parent].tempBuffer[i]]);
        glm::vec3 vec_max = triMax(tri_arr[vec_kdnode[parent].tempBuffer[i]]);
        local_min = glm::vec3(fmin(local_min.x, vec_min.x),
            fmin(local_min.y, vec_min.y),
            fmin(local_min.z, vec_min.z));
        local_max = glm::vec3(fmax(local_max.x, vec_max.x),
            fmax(local_max.y, vec_max.y),
            fmax(local_max.z, vec_max.z));
        //std::cout << local_min.x << " " << std::endl;
        //std::cout << local_max.x << std::endl;
    }
#endif
    //minCorner	{x=-0.0212809108 y=-4.77385511e-05 z=-0.0138090001 ...}	
    //maxCorner{ x = 0.0212809108 y = 0.0628480613 z = 0.0138090011 ... }	
            //printf("min-------------\n");
        //printf("x %f y %f z %f\n", local_min.x, local_min.y, local_min.z);
        //printf("x %f y %f z %f\n", child_min.x, child_min.y, child_min.z);
        //printf("x %f y %f z %f\n", fmin(local_min.x, child_min.x), fmin(local_min.y, child_min.y), fmin(local_min.z, child_min.z));
        //printf("max-------------\n");
        //printf("x %f y %f z %f\n", local_max.x, local_max.y, local_max.z);
        //printf("x %f y %f z %f\n", child_max.x, child_max.y, child_max.z);
        //printf("x %f y %f z %f\n", fmax(local_max.x, child_max.x), fmax(local_max.y, child_max.y), fmax(local_max.z, child_max.z));
    glm::vec3 child_min = local_min;
    glm::vec3 child_max = local_max;

    if (node.near_node != -1) {
        buildBounds(tri_arr, node.near_node, child_min, child_max);
        local_min = glm::vec3(fmin(local_min.x, child_min.x), 
            fmin(local_min.y, child_min.y), 
            fmin(local_min.z, child_min.z));
        local_max = glm::vec3(fmax(local_max.x, child_max.x),
            fmax(local_max.y, child_max.y),
            fmax(local_max.z, child_max.z));
    }
    if (node.far_node != -1) {
        buildBounds(tri_arr, node.far_node, child_min, child_max);

        local_min = glm::vec3(fmin(local_min.x, child_min.x),
            fmin(local_min.y, child_min.y),
            fmin(local_min.z, child_min.z));
        local_max = glm::vec3(fmax(local_max.x, child_max.x),
            fmax(local_max.y, child_max.y),
            fmax(local_max.z, child_max.z));
    }

    vec_kdnode[parent].bound.minCorner = local_min;
    vec_kdnode[parent].bound.maxCorner = local_max;

    //minCorner	{x=-0.0212809108 y=-4.77385511e-05 z=-0.0138090001 ...}	
    //maxCorner{ x = 0.0212809108 y = 0.0628480613 z = 0.0138090011 ... }
    min = local_min;
    max = local_max;
}

void Scene::createNode(int node_idx, int tri_idx, int parent_idx, BoundBox bound, KDSPLIT split, parentRelation rel, int depth) {
    vec_kdnode[node_idx].bound = bound;
    vec_kdnode[node_idx].trisIndex = tri_idx;
    vec_kdnode[node_idx].near_node = -1;
    vec_kdnode[node_idx].far_node = -1;
    vec_kdnode[node_idx].split = split;
    vec_kdnode[node_idx].parent_node = parent_idx;
    vec_kdnode[node_idx].relation = rel;

    vec_kdnode[node_idx].depth = depth;
#if USE_KD_VEC
    vec_kdnode[node_idx].tempBuffer.push_back(tri_idx); //KD_DEBUG
#endif
}

void Scene::pushdown(Triangle* tri_arr, int parent, BoundBox bound, int tri_idx) {


    KDNode& node = vec_kdnode[parent];
    KDSPLIT parent_split = node.split;
    int parent_tri_idx = vec_kdnode[parent].trisIndex;

#if USE_KD_VEC
    if (node.depth == MAXDEPTH) {
        vec_kdnode[parent].tempBuffer.push_back(tri_idx); //KD_DEBUG
        return;
    }
#endif

    Triangle& parent_tri = tri_arr[parent_tri_idx];
    Triangle& new_tri = tri_arr[tri_idx];
    //std::cout << tri_arr[tri_idx].pointA.pos[0] << std::endl;

    //Comparing the two triangles, deciding which subtree to go down
    bool useNear;
    if (parent_split == X) {
        useNear = triCompare(new_tri, parent_tri, 0);
    } 
    else if (parent_split == Y) {
        useNear = triCompare(new_tri, parent_tri, 1);
    }
    else {
        useNear = triCompare(new_tri, parent_tri, 2);
    }

    //Actually pushing down the correct subtree, if it exists
    if (useNear) {
        if (node.near_node >= 0) {
            pushdown(tri_arr, node.near_node, vec_kdnode[node.near_node].bound, tri_idx);
            return;
        }
    }
    else {
        if (node.far_node >= 0) {
            pushdown(tri_arr, node.far_node, vec_kdnode[node.far_node].bound, tri_idx);
            return;
        }
    }

    //If we are here then we need to create a new node
    //This determines the new split X-->Y-->Z-->-->X etc.
    //This is also used to determine how new bounding boxes are constructed/partitioned
    KDSPLIT child_split;
    BoundBox new_bound = bound;
    if (parent_split == X) {
        child_split = Y;
        new_bound = buildBound(bound, new_tri, parent_tri, 0, useNear);
    }
    else if (parent_split == Y) {
        child_split = Z;
        new_bound = buildBound(bound, new_tri, parent_tri, 1, useNear);
    }
    else {
        child_split = X;
        new_bound = buildBound(bound, new_tri, parent_tri, 2, useNear);
    }

    new_bound = bound; //KD_DEBUG

    //New Node pushed to vec
    int node_idx = vec_kdnode.size();
    int currentDepth = node.depth;
    vec_kdnode.push_back(KDNode());
    
    if (useNear) {
        createNode(node_idx, tri_idx, parent, new_bound, child_split, LEFT, currentDepth + 1);
        vec_kdnode[parent].near_node = node_idx;
    }
    else {
        createNode(node_idx, tri_idx, parent, new_bound, child_split, RIGHT, currentDepth + 1);
        vec_kdnode[parent].far_node = node_idx;
    }
}

void Scene::constructKDTrees() {
    vec_kdnode.clear();
    for (int i = 0; i < kdtree_indices.size(); i++) {
        int idx = kdtree_indices[i];

        Geom* ref = &geoms[idx];
        
        ref->root = vec_kdnode.size();
        vec_kdnode.push_back(KDNode());
        Triangle* tri_arr = ref->host_tris;
        if (ref->numTris >= 1) {
            createNode(ref->root, 0, -1, ref->bound, X, ROOT, 0);
        }

        for (int j = 1; j < ref->numTris; j++) {
            pushdown(tri_arr, ref->root, ref->bound, j);
            /*if (vec_kdnode[j].trisIndex != j) {
                printf("BUG: %d %d\n", j, vec_kdnode[j].trisIndex);
            }*/
        }
        glm::vec3 minCorn;
        glm::vec3 maxCorn;
        buildBounds(tri_arr, ref->root, minCorn, maxCorn);
    }
#if USE_KD_VEC
    for (int i = 0; i < vec_kdnode.size(); i++) {
        vec_kdnode[i].numIndices = vec_kdnode[i].tempBuffer.size();
    }
#endif
}
