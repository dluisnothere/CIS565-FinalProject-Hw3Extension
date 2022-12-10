#include <cstdio>
#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/sort.h>
#include <thrust/distance.h>
#include <thrust/partition.h>
#include <thrust/extrema.h>
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <iostream>
#include <glm/gtc/matrix_inverse.hpp>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "pathtrace.h"
#include "intersections.h"
#include "interactions.h"

#include <device_launch_parameters.h>

#define ERRORCHECK 1

//SAR
#define ORTHOGRAPHIC 1
#define SARNAIVE 1
#define PERF_ANALYSIS 0

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

PerformanceTimer& timer()
{
	static PerformanceTimer timer;
	return timer;
}

void checkCUDAErrorFn(const char* msg, const char* file, int line) {
#if ERRORCHECK
	cudaDeviceSynchronize();
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess == err) {
		return;
	}

	fprintf(stderr, "CUDA error");
	if (file) {
		fprintf(stderr, " (%s:%d)", file, line);
	}
	fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#  ifdef _WIN32
	getchar();
#  endif
	exit(EXIT_FAILURE);
#endif
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth) {
	int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
	return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution,
	int iter, glm::vec3* image) {
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < resolution.x && y < resolution.y) {
		int index = x + (y * resolution.x);
		glm::vec3 pix = image[index];

		glm::ivec3 color;
		color.x = glm::clamp((int)(pix.x / iter * 255.0), 0, 255);
		color.y = glm::clamp((int)(pix.y / iter * 255.0), 0, 255);
		color.z = glm::clamp((int)(pix.z / iter * 255.0), 0, 255);

		// Each thread writes one pixel location in the texture (textel)

		pbo[index].w = 0;
		pbo[index].x = color.x;
		pbo[index].y = color.y;
		pbo[index].z = color.z;
	}
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
static Camera* dev_camera = NULL;

static float* dev_maxRange;

// TODO: static variables for device memory, any extra info you need, etc
// ...
#if CACHE_FIRST_BOUNCE
static ShadeableIntersection* dev_cached_intersections = NULL;
#endif

static Texture* dev_textures = NULL;
static std::vector<cudaArray_t> dev_texData;
static std::vector<cudaTextureObject_t> host_textureObjs;
static cudaTextureObject_t* dev_textureObjs;
static int* dev_textureChannels;

static int numTextures;
static int numGeoms; //cursed variables to cudaFree nested pointers;
static int numMaterials;

static Geom* dev_lights = NULL;

#if PERF_ANALYSIS
static cudaEvent_t beginEvent = NULL;
static cudaEvent_t endEvent = NULL;
#endif

void InitDataContainer(GuiDataContainer* imGuiData)
{
	guiData = imGuiData;
}

// specialized function just to cudaMalloc textures
void createTexture(const Texture &t, int numChannels, int texIdx) {
	// create a channel desc
	cudaChannelFormatDesc channelDesc;

	if (numChannels == 4) {
		channelDesc = cudaCreateChannelDesc<uchar4>();
		checkCUDAError("cudaCreateChannelDesc 4 failed");
	}
	else if (numChannels == 3) {
		channelDesc = cudaCreateChannelDesc<uchar3>();
		checkCUDAError("cudaCreateChannelDesc 3 failed");
	}

	cudaMallocArray(&dev_texData[texIdx], &channelDesc, t.width, t.height);
	checkCUDAError("CudaMallocArray textures failed");

	cudaMemcpyToArray(dev_texData[texIdx], 0, 0, t.host_texData, t.height * t.width * numChannels * sizeof(unsigned char), cudaMemcpyHostToDevice);
	checkCUDAError("CudaMemcpyToArray textures failed");

	struct cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	resDesc.res.array.array = dev_texData[texIdx];

	struct cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeWrap;
	texDesc.addressMode[1] = cudaAddressModeWrap;
	texDesc.filterMode = cudaFilterModeLinear;
	texDesc.readMode = cudaReadModeNormalizedFloat;
	texDesc.normalizedCoords = 1;

	cudaTextureObject_t texObj = 0;
	cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);
	checkCUDAError("Cuda create texture object failed");
	cudaMemcpy(&dev_textureObjs[texIdx], &texObj, sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice);
	checkCUDAError("CudaMemcpy dev_textureObjs failed");
}

void pathtraceInit(Scene* scene) {
	hst_scene = scene;
	numTextures = hst_scene->textures.size();

	numGeoms = hst_scene->geoms.size();
	numMaterials = hst_scene->materials.size();

	const Camera& cam = hst_scene->state.camera;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
	checkCUDAError("cudaMalloc dev_image failed");
	cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));
	checkCUDAError("cudaMemsetd dev_image failed");

	cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));
	checkCUDAError("cudaMalloc dev_paths failed");

	for (int i = 0; i < scene->geoms.size(); i++) {
		if (scene->geoms[i].numTris > 0) {
			for (int j = 0; j < scene->geoms[i].numTris; j++) {
				Vertex pointA = scene->geoms[i].host_tris[j].pointA;
				Vertex pointB = scene->geoms[i].host_tris[j].pointB;
				Vertex pointC = scene->geoms[i].host_tris[j].pointC;

				cudaMalloc(&(scene->geoms[i].host_tris[j].pointA.dev_uvs), scene->geoms[i].host_tris[j].pointA.host_uvs.size() * sizeof(glm::vec2));
				checkCUDAError("cudaMalloc device_uvs A failed");

				cudaMalloc(&(scene->geoms[i].host_tris[j].pointB.dev_uvs), scene->geoms[i].host_tris[j].pointB.host_uvs.size() * sizeof(glm::vec2));
				checkCUDAError("cudaMalloc device_uvs B failed");

				cudaMalloc(&(scene->geoms[i].host_tris[j].pointC.dev_uvs), scene->geoms[i].host_tris[j].pointC.host_uvs.size() * sizeof(glm::vec2));
				checkCUDAError("cudaMalloc device_uvs C failed");

				cudaMemcpy(scene->geoms[i].host_tris[j].pointA.dev_uvs, scene->geoms[i].host_tris[j].pointA.host_uvs.data(), scene->geoms[i].host_tris[j].pointA.host_uvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);
				checkCUDAError("cudaMemcpy device_uvs A failed");

				cudaMemcpy(scene->geoms[i].host_tris[j].pointB.dev_uvs, scene->geoms[i].host_tris[j].pointB.host_uvs.data(), scene->geoms[i].host_tris[j].pointB.host_uvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);
				checkCUDAError("cudaMemcpy device_uvs B failed");

				cudaMemcpy(scene->geoms[i].host_tris[j].pointC.dev_uvs, scene->geoms[i].host_tris[j].pointC.host_uvs.data(), scene->geoms[i].host_tris[j].pointC.host_uvs.size() * sizeof(glm::vec2), cudaMemcpyHostToDevice);
				checkCUDAError("cudaMemcpy device_uvs C failed");
			}
			cudaMalloc(&(scene->geoms[i].device_tris), scene->geoms[i].numTris * sizeof(Triangle));
			checkCUDAError("cudaMalloc device_tris failed");
			cudaMemcpy(scene->geoms[i].device_tris, scene->geoms[i].host_tris, scene->geoms[i].numTris * sizeof(Triangle), cudaMemcpyHostToDevice);
			checkCUDAError("cudaMemcpy device_tris failed");
		}
	}

	cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
	checkCUDAError("cudaMalloc dev_geoms failed");
	cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy gev_geoms failed");

	cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
	checkCUDAError("cudaMalloc dev_materials failed");
	cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy dev_materials failed");

	cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
	checkCUDAError("cudaMalloc dev_intersections failed");
	cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
	checkCUDAError("cudaMemcpy dev_intersectionsf ailed");

	cudaMalloc(&dev_textureChannels, scene->textures.size() * sizeof(int));
	checkCUDAError("cudaMalloc dev_textureChannels failed");

	for (int i = 0; i < scene->textures.size(); i++) {
		cudaMemcpy(&dev_textureChannels[i], &scene->textures[i].channels, sizeof(int), cudaMemcpyHostToDevice);
		checkCUDAError("cudaMalloc dev_textureChannels failed");
	}

	cudaMalloc(&dev_textures, scene->textures.size() * sizeof(Texture));
	checkCUDAError("cudaMalloc dev_textures failed");
	cudaMemcpy(dev_textures, scene->textures.data(), scene->textures.size() * sizeof(Texture), cudaMemcpyHostToDevice);
	checkCUDAError("cudaMemcpy dev_textures failed");

	cudaMalloc(&dev_textureObjs, scene->textures.size() * sizeof(cudaTextureObject_t));
	checkCUDAError("cudaMalloc dev_textureObjs ailed");

	dev_texData.resize(scene->textures.size());

	for (int i = 0; i < scene->textures.size(); i++) {
		createTexture(scene->textures[i], scene->textures[i].channels, i);
	}

#if PERF_ANALYSIS
	cudaEventCreate(&beginEvent);
	cudaEventCreate(&endEvent);
#endif

	checkCUDAError("pathtraceInit");
}

void pathtraceFree() {
	cudaFree(dev_image);  // no-op if dev_image is null
	checkCUDAError("cudaFree dev_image failed");
	cudaFree(dev_paths);
	checkCUDAError("cudaFree dev_paths failed");

	int numG = numGeoms;
	Geom* tmp_geom_pointer = new Geom[numG];
	cudaMemcpy(tmp_geom_pointer, dev_geoms, numG * sizeof(Geom), cudaMemcpyDeviceToHost);
	checkCUDAError("cudaMemcpy tmp_geom_pointer failed");

	for (int i = 0; i < numGeoms; i++) {
		if (tmp_geom_pointer[i].type == OBJ || tmp_geom_pointer[i].type == GLTF) {
			int numTris = tmp_geom_pointer[i].numTris;
			Triangle* tmp_tri_pointer = new Triangle[numTris];
			cudaMemcpy(tmp_tri_pointer, tmp_geom_pointer[i].device_tris, numTris * sizeof(Triangle), cudaMemcpyDeviceToHost);
			checkCUDAError("303 cudaMemcpy tmp_tri_pointer failed");

			for (int j = 0; j < numTris; j++) {
				cudaFree(tmp_tri_pointer[j].pointA.dev_uvs);
				checkCUDAError("cudaFree point A device_uvs failed");
				cudaFree(tmp_tri_pointer[j].pointB.dev_uvs);
				checkCUDAError("cudaFree point B device_uvs failed");
				cudaFree(tmp_tri_pointer[j].pointC.dev_uvs);
				checkCUDAError("cudaFree point C device_uvs failed");
			}

			cudaFree(tmp_geom_pointer[i].device_tris);
			checkCUDAError("cudaFree device_tris failed");
		}
	}

	delete[] tmp_geom_pointer;


	for (int i = 0; i < host_textureObjs.size(); i++) {
		cudaDestroyTextureObject(host_textureObjs[i]);
		cudaFreeArray(dev_texData[i]);
	}

	cudaFree(dev_textures);
	cudaFree(dev_geoms);
	cudaFree(dev_materials);
	cudaFree(dev_intersections);
	// TODO: clean up any extra device memory you created

#if PERF_ANALYSIS
	if (beginEvent != NULL) {
		cudaEventDestroy(beginEvent);
	} 
	if (endEvent != NULL) {
		cudaEventDestroy(endEvent);
	}
#endif
	checkCUDAError("pathtraceFree");
}

Geom generateNewReceiverFromCamera(Camera cam) {
	Geom newGeom;
	newGeom.type = CUBE;
	/*newGeom.translation = glm::vec3(cam.position) * 1.1f;*/
	glm::vec3 u = glm::vec3(0.0, 0.0, 1.0);
	glm::vec3 v = cam.view;
	newGeom.translation = glm::vec3(cam.position) - 0.1f * v;
	float x_angle = glm::acos(glm::dot(glm::vec2(u.y, u.z), glm::vec2(v.y, v.z))) * 180.f / PI;
	float y_angle = glm::acos(glm::dot(glm::vec2(u.x, u.z), glm::vec2(v.x, v.z))) * 180.f / PI;
	float z_angle = glm::acos(glm::dot(glm::vec2(u.x, u.y), glm::vec2(v.x, v.y))) * 180.f / PI;
	newGeom.rotation = glm::vec3(x_angle, y_angle, z_angle);
	//this scal is arbitrary
	newGeom.scale = glm::vec3(cam.pixelLength.x * cam.resolution.x * 0.5f, cam.pixelLength.y * cam.resolution.y * 0.5f, 0.001f);

	newGeom.transform = utilityCore::buildTransformationMatrix(
		newGeom.translation, newGeom.rotation, newGeom.scale);
	newGeom.inverseTransform = glm::inverse(newGeom.transform);
	newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);
	return newGeom;
}

__global__ void adjustReceiver(Geom geo, int receiverIndex, Geom* geoms)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	if (index != 0) {
		geoms[receiverIndex] = geo;
	}
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;
	
	if (x < cam.resolution.x && y < cam.resolution.y) {
		int index = x + (y * cam.resolution.x);
		PathSegment& segment = pathSegments[index];

		segment.color = glm::vec3(0.f);
		segment.color2 = glm::vec3(0.f);
		segment.color3 = glm::vec3(0.f);
		segment.depth = 0;

#if ORTHOGRAPHIC
		//when x = 0, y = 0, ray's origin is at [1, 0, 1]
		segment.ray.origin = cam.position
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f);

		segment.ray.direction = cam.view;
		//printf("camera lookAt: %f, %f, %f\n", cam.lookAt.x, cam.lookAt.y, cam.lookAt.z);
		
#else
		segment.ray.origin = cam.position;
		segment.ray.direction = glm::normalize(cam.view
			- cam.right * cam.pixelLength.x * ((float)x - (float)cam.resolution.x * 0.5f)
			- cam.up * cam.pixelLength.y * ((float)y - (float)cam.resolution.y * 0.5f)
		);
#endif


		segment.pixelIndex = index;
		segment.pixelIndex2 = index;
		segment.ae1 = glm::vec2(x, y);
		// debug hard code to 2 instead of traceDepth;
		segment.remainingBounces = traceDepth;
		segment.length1 = 0.f;
		segment.length2 = 0.f;
		segment.length3 = 0.f;
		segment.pixelIndexX = x;
		segment.pixelIndexX2 = x;
		segment.pixelIndexY = y;
		segment.pixelIndexY2 = y;
		segment.checkCameraBlock = true;
		segment.fordebug = false;
	}
}

__global__ void kernComputeBlockToCameraSAR(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];
		if (pathSegment.remainingBounces == 0) {
			return;
		}
		if (!(pathSegment.checkCameraBlock)) {
			return;
		}

		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		glm::vec2 uv = glm::vec2(-1, -1);
		bool outside = true;
		bool hitObj; // for use in procedural texturing
		glm::vec4 tangent = glm::vec4(0, 0, 0, 0);

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv = glm::vec2(-1, -1);
		glm::vec4 tmp_tangent = glm::vec4(0, 0, 0, 0);
		bool tmpHitObj = false;

		// naive parse through global geoms

		// test intersection with big obj box and set a boolean for whether triangle should be checked based on this ray.

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				tmpHitObj = false;
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				tmpHitObj = false;
			}
			else if (geom.type == OBJ || geom.type == GLTF) {
				float boxT = boundBoxIntersectionTest(&geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);

				if (boxT != -1) {
					for (int j = 0; j < geom.numTris; j++) {
#if BUMP_MAP
						cudaTextureObject_t texObj = textureObjs[geom.textureid];
						Texture tex = texs[geom.textureid];
						t = triangleIntersectionTest(&geom, &geom.device_tris[j], pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, texObj, tex, outside);
#else
						t = triangleIntersectionTest(&geom, &geom.device_tris[j], pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, outside);
#endif
						tmpHitObj = true;

						if (t > 0.0f && t_min > t)
						{
							t_min = t;
							hit_geom_index = i;
							intersect_point = tmp_intersect;
							normal = tmp_normal;
							uv = tmp_uv;
							hitObj = tmpHitObj;
							tangent = tmp_tangent;
						}
					}
				}
			}
			else if (geom.type == TRIANGLE) {
				// Only use the first triangle, since in Triangle mode, each geom only has 1 triangle
#if BUMP_MAP
				cudaTextureObject_t texObj = textureObjs[geom.textureid];
				Texture tex = texs[geom.textureid];
				t = triangleIntersectionTest(&geom, &geom.device_tris[0], pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, texObj, tex, outside);
#else
				t = triangleIntersectionTest(&geom, &geom.device_tris[0], pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, outside);
#endif
				tmpHitObj = true;
			}
			// Compute the minimum t from the intersection tests to determine what
			// scene geometry object was hit first.
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				uv = tmp_uv;
				hitObj = tmpHitObj;
				tangent = tmp_tangent;
			}
		}

		if (hit_geom_index == -1)
		{
			//printf("kill this pixel!  %f", pathSegment.ray.direction.x, pathSegment.ray.direction.y, pathSegment.ray.direction.z);
			// if there is nothing, then can hit camera
		}
		else
		{
			if (depth == 2) {
				pathSegments[path_index].color2 = glm::vec3(0.f);
				//printf("my pathSegment pixelIndex2: %d", pathSegment.pixelIndex2);
				//printf("my remainingBounces: %d\nmy debug: %d\n", pathSegment.remainingBounces, pathSegment.fordebug);
			}

		}
		//no matter what, bounce should minus 1
		--(pathSegment.remainingBounces);
		pathSegment.ray.direction = pathSegment.realRayDir;
	}
}

__global__ void computeIntersectionsSAR(
	int depth
	, int num_paths
	, PathSegment* pathSegments
	, Geom* geoms
	, int geoms_size
	, ShadeableIntersection* intersections
) {
	int path_index = blockIdx.x * blockDim.x + threadIdx.x;

	if (path_index < num_paths)
	{
		PathSegment pathSegment = pathSegments[path_index];

		if (pathSegment.remainingBounces == 0) {
			return;
		}
		float t;
		glm::vec3 intersect_point;
		glm::vec3 normal;
		float t_min = FLT_MAX;
		int hit_geom_index = -1;
		glm::vec2 uv = glm::vec2(-1, -1);
		bool outside = true;
		bool hitObj; // for use in procedural texturing
		glm::vec4 tangent = glm::vec4(0, 0, 0, 0);

		glm::vec3 tmp_intersect;
		glm::vec3 tmp_normal;
		glm::vec2 tmp_uv = glm::vec2(-1, -1);
		glm::vec4 tmp_tangent = glm::vec4(0, 0, 0, 0);
		bool tmpHitObj = false;

		// naive parse through global geoms

		// test intersection with big obj box and set a boolean for whether triangle should be checked based on this ray.

		for (int i = 0; i < geoms_size; i++)
		{
			Geom& geom = geoms[i];

			if (geom.type == CUBE)
			{
				t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				tmpHitObj = false;
			}
			else if (geom.type == SPHERE)
			{
				t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
				tmpHitObj = false;
			}
			else if (geom.type == OBJ || geom.type == GLTF) {
				float boxT = boundBoxIntersectionTest(&geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);

				if (boxT != -1) {
					for (int j = 0; j < geom.numTris; j++) {
						t = triangleIntersectionTest(&geom, &geom.device_tris[j], pathSegment.ray, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, outside);
						tmpHitObj = true;

						if (t > 0.0f && t_min > t)
						{
							t_min = t;
							hit_geom_index = i;
							intersect_point = tmp_intersect;
							normal = tmp_normal;
							uv = tmp_uv;
							hitObj = tmpHitObj;
							tangent = tmp_tangent;
						}
					}
				}
			}
			if (t > 0.0f && t_min > t)
			{
				t_min = t;
				hit_geom_index = i;
				intersect_point = tmp_intersect;
				normal = tmp_normal;
				uv = tmp_uv;
				hitObj = tmpHitObj;
				tangent = tmp_tangent;
			}
		}

		if (hit_geom_index == -1)
		{
			pathSegments[path_index].remainingBounces = 0;
			intersections[path_index].t = -1.0f;
		}
		else
		{
			//The ray hits something
			intersections[path_index].t = t_min;
#if LOAD_GLTF
			if (geoms[hit_geom_index].materialOffset != -1) {
				intersections[path_index].materialId = geoms[hit_geom_index].materialid + geoms[hit_geom_index].materialOffset;
			}
			else {
				intersections[path_index].materialId = geoms[hit_geom_index].materialid;
			}
			intersections[path_index].tangent = tangent;
#endif
			intersections[path_index].surfaceNormal = normal;
			intersections[path_index].uv = uv;
#if LOAD_OBJ
			intersections[path_index].textureId = geoms[hit_geom_index].objTexId;
#endif
			intersections[path_index].hasHitObj = hitObj;

			pathSegments[path_index].remainingBounces--;
		}
	}
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths, int depth, Camera cam)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		if (depth == 1) {
			glm::vec3 resultColor = glm::vec3(iterationPath.color.r);
			//glm::vec3 resultColor = iterationPath.color; //direct backscatter
			//glm::vec3 resultColor = iterationPath.color2; //double bounce reflection
			if (glm::isnan(resultColor.r)) {
				resultColor = glm::vec3(0.f);
			}
			image[iterationPath.pixelIndex] += resultColor;
		}
		else if (depth == 2) {
			//glm::vec3 resultColor = glm::vec3(iterationPath.color2.r + iterationPath.color.r);
			//glm::vec3 resultColor = iterationPath.color; //direct backscatter
			glm::vec3 resultColor = glm::vec3(iterationPath.color2.r); //double bounce reflection
			if (glm::isnan(resultColor.r)) {
				resultColor = glm::vec3(0.f);
			}
			image[iterationPath.pixelIndex2] += resultColor;
		}
	}
}

__global__ void finalGatherTest(int nPaths, glm::vec3* image, PathSegment* iterationPaths, float maxRange)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		PathSegment iterationPath = iterationPaths[index];
		image[iterationPath.pixelIndex] += glm::vec3(maxRange/32.f);
	}
}

__global__ void kernFinalColorClean(int nPaths, glm::vec3* image) {
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;

	if (index < nPaths)
	{
		image[index].g = image[index].r;
		image[index].b = image[index].r;
	}
}

// thrust predicate to end rays that don't hit anything
struct invalid_intersection
{
	__host__ __device__
		bool operator()(const PathSegment& path)
	{
		if (path.remainingBounces)
		{
			return true;
		}
		return false;
	}
};

// thrust predicate to comapre one Intersection
struct path_cmp {
	__host__ __device__
		bool operator()(ShadeableIntersection& inter1, ShadeableIntersection& inter2) {
		return inter1.materialId < inter2.materialId;
	}
};

__global__ void kernComputeShadeSAR(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
	, int depth
	, Camera cam
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		ShadeableIntersection intersection = shadeableIntersections[idx];
		if (intersection.t > 0.0f) { // if the intersection exists...
			Material material = materials[intersection.materialId];
			//printf("emissiveFactor %f,	pbrMetallicRoughness %f,	specular %f\n", material.emissiveFactor, material.pbrMetallicRoughness, material.specular);
			/*if (material.emittance > 0.0f) {
				pathSegments[idx].remainingBounces = 0;
				return;
			}*/
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
			thrust::uniform_real_distribution<float> u01(0, 1);
			float Fd = 0.2f;
			float Fb = 0.3f;
			float Fs = 0.6f;
			float Fr = 0.3f;
			//Fd: diffuse reflection coefficient [0....1]
			// material.hasRefractive, REFR
			//Fb: surface brilliance factor [default value: 1].
			// material.indexOfRefraction, REFRIOR
			//Fs: specular reflection coefficient [0....1]
			// material.hasReflective, REFL
			//Fr: roughness factor
			// material.specular.exponent, SPECEX
			
			glm::vec3 N = glm::normalize(intersection.surfaceNormal);
			glm::vec3 L = glm::normalize(-pathSegments[idx].ray.direction);
			glm::vec3 H = glm::normalize(N + L);
			float Id = Fd * pathSegments[idx].color.r * glm::pow(glm::dot(N, L), Fb);
			float Is = Fs * glm::pow(glm::dot(N, H), Fr);
			float resultColor = Id + Is;
			if (depth == 1) {
				glm::vec3 V = glm::reflect(-L , N);
				pathSegments[idx].negPriRay = L;
				pathSegments[idx].color = glm::vec3(resultColor);
				//pathSegments[idx].color += glm::vec3(u01(rng) * 0.1f);   //noise
				pathSegments[idx].length1 = intersection.t;
				pathSegments[idx].depth = 1;
				if (isnan(pathSegments[idx].color.r)) {
					pathSegments[idx].color = glm::vec3(0.f);
				}

				//update ray
				pathSegments[idx].ray.origin = pathSegments[idx].ray.origin + intersection.t * glm::normalize(pathSegments[idx].ray.direction) + 0.001f * N;
				pathSegments[idx].ray.direction = V;
			}
			else if (depth == 2) {
				//test focusing ray with camera plane
				glm::vec3 focusingRayDir = pathSegments[idx].negPriRay;
				glm::vec3 focusingRayOri = pathSegments[idx].ray.origin + intersection.t * pathSegments[idx].ray.direction + 0.001f * N;
				Ray focusing;
				focusing.origin = focusingRayOri;
				focusing.direction = focusingRayDir;
				int pixelIdxX;
				int pixelIdxY;
				float t_cam = squarePlaneIntersectionTest(cam, focusing, pixelIdxX, pixelIdxY);

				if (t_cam > 0.f) {
					pathSegments[idx].color2 = glm::vec3(resultColor * pathSegments[idx].color.r);
					if (isnan(pathSegments[idx].color2.r)) {
						pathSegments[idx].color2 = glm::vec3(0.f);
					}
					//pathSegments[idx].color2 += glm::vec3(u01(rng) * 0.1f);    //noise
					pathSegments[idx].length2 = (intersection.t + t_cam + pathSegments[idx].length1) / 2.f;
					pathSegments[idx].pixelIndexX2 = (pixelIdxX + pathSegments[idx].pixelIndexX) / 2;
					pathSegments[idx].pixelIndexY2 = (pixelIdxY + pathSegments[idx].pixelIndexY) / 2;
					pathSegments[idx].depth = 2;
					pathSegments[idx].pixelIndex2 = pathSegments[idx].pixelIndexX2 + (pathSegments[idx].pixelIndexY2 * cam.resolution.x);
					
					//update Ray
					pathSegments[idx].ray.origin = focusingRayOri;
					pathSegments[idx].ray.direction = focusingRayDir;
					glm::vec3 V = glm::reflect(-L, N);
					pathSegments[idx].realRayDir = V;
					//回血bounce
					++(pathSegments[idx].remainingBounces);
				}
				else {
					pathSegments[idx].checkCameraBlock = false;
				}
			}
		}
		else {
			pathSegments[idx].remainingBounces = 0;
		}
	}
}

__global__ void computeIntersectionsWithLight(
	int iter
	, int num_paths
	, ShadeableIntersection* shadeableIntersections
	, PathSegment* pathSegments
	, Material* materials
) {

}

struct compare_range {
	__host__ __device__
		bool operator()(PathSegment pathSegment1, PathSegment pathSegment2)
	{
		return pathSegment1.length < pathSegment2.length;
	}
};

struct length_zero {
	__host__ __device__
		bool operator()(const PathSegment& pathSegment1)
	{
		return pathSegment1.length1 > 0.f;
	}
};

__global__ void kernTransToAzimuthRange(
	int num_paths,
	PathSegment* pathSegments,
	float maxRange,
	float minRange,
	int camResolutionX,
	int camResolutionY,
	int depth
)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths)
	{
		PathSegment& pathSegment = pathSegments[idx];
		if (depth == 1) {
			int y = (int)glm::floor(((pathSegment.length1 - minRange) / (maxRange - minRange)) * camResolutionY);
			if (y < 0) {
				y = 0;
			}
			if (y >= camResolutionY) {
				y = camResolutionY - 1;
			}
			pathSegment.pixelIndex = pathSegment.pixelIndexX + (y * camResolutionX);
		}
		else if (depth == 2) {
			//range越大，越靠近bottom.
			int y = (int)glm::floor(((pathSegment.length2 - minRange) / (maxRange - minRange)) * camResolutionY);
			if (y < 0) {
				y = 0;
			}
			if (y >= camResolutionY) {
				y = camResolutionY - 1;
			}
			pathSegment.pixelIndex2 = pathSegment.pixelIndexX2 + (y * camResolutionX);
		}
	}
}

__global__ void kernUpdateLength(int num_paths, PathSegment* paths) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		PathSegment& pathSegment = paths[idx];
		if (pathSegment.length3 > 0.f) {
			pathSegment.length = pathSegment.length3;
		}
		else {
			if (pathSegment.length2 > 0.f) {
				pathSegment.length = pathSegment.length2;
			}
			else {
				pathSegment.length = pathSegment.length1;
			}
		}
	}
}

__global__ void kernLetMeCheckRange(
	PathSegment* pathSegments,
	int num_paths
) {
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < num_paths) {
		if (pathSegments[idx].pixelIndex2 == 361235) {
			printf("I am 361235: %f, %f, %f\n", pathSegments[idx].color2.r, pathSegments[idx].color2.g, pathSegments[idx].color2.b);
		}
	}
}

// If not cache intersections
/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter) {
	const int traceDepth = hst_scene->state.traceDepth;
	const Camera& cam = hst_scene->state.camera;/*
	cudaMalloc(&dev_camera, sizeof(Camera));
	cudaMemcpy(dev_camera, &(hst_scene->state.camera), sizeof(Scene), cudaMemcpyHostToDevice);*/
	//so the problem is, eye is wrong, lookAt is correct, view is normalize(lookAt - eye), but since eye is wrong, view is wrong.
	cout << "eye" << endl;
	cout << hst_scene->state.camera.position.x << endl; //0
	cout << hst_scene->state.camera.position.y << endl; //2.5
	cout << hst_scene->state.camera.position.z << endl; //5
	cout << "lookAt" << endl;
	cout << hst_scene->state.camera.lookAt.x << endl;  //0
	cout << hst_scene->state.camera.lookAt.y << endl;  //3
	cout << hst_scene->state.camera.lookAt.z << endl;  //0
	cout << "view" << endl;
	cout << hst_scene->state.camera.view.x << endl;
	cout << hst_scene->state.camera.view.y << endl;
	cout << hst_scene->state.camera.view.z << endl;
	const int pixelcount = cam.resolution.x * cam.resolution.y;

	// 2D block for generating ray from camera
	const dim3 blockSize2d(8, 8);
	const dim3 blocksPerGrid2d(
		(cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
		(cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

	// 1D block for path tracing
	const int blockSize1d = 128;

	///////////////////////////////////////////////////////////////////////////

	// Recap:
	// * Initialize array of path rays (using rays that come out of the camera)
	//   * You can pass the Camera object to that kernel.
	//   * Each path ray must carry at minimum a (ray, color) pair,
	//   * where color starts as the multiplicative identity, white = (1, 1, 1).
	//   * This has already been done for you.
	// * For each depth:
	//   * Compute an intersection in the scene for each path ray.
	//     A very naive version of this has been implemented for you, but feel
	//     free to add more primitives and/or a better algorithm.
	//     Currently, intersection distance is recorded as a parametric distance,
	//     t, or a "distance along the ray." t = -1.0 indicates no intersection.
	//     * Color is attenuated (multiplied) by reflections off of any object
	//   * TODO: Stream compact away all of the terminated paths.
	//     You may use either your implementation or `thrust::remove_if` or its
	//     cousins.
	//     * Note that you can't really use a 2D kernel launch any more - switch
	//       to 1D.
	//   * TODO: Shade the rays that intersected something or didn't bottom out.
	//     That is, color the ray by performing a color computation according
	//     to the shader, then generate a new ray to continue the ray path.
	//     We recommend just updating the ray's PathSegment in place.
	//     Note that this step may come before or after stream compaction,
	//     since some shaders you write may also cause a path to terminate.
	// * Finally, add this iteration's results to the image. This has been done
	//   for you.

	// TODO: perform one iteration of path tracing

	Geom nextGeom = generateNewReceiverFromCamera(cam);
	adjustReceiver << <dim3(1), dim3(1) >> > (nextGeom, hst_scene->state.receiverIndex, dev_geoms);


	generateRayFromCamera << <blocksPerGrid2d, blockSize2d >> > (cam, iter, traceDepth, dev_paths);
	checkCUDAError("generate camera ray");

	int depth = 0;
	PathSegment* dev_path_end = dev_paths + pixelcount;
	int num_paths = pixelcount;

	// --- PathSegment Tracing Stage ---
	// Shoot ray into scene, bounce between objects, push shading chunks

	bool iterationComplete = false;

	int currNumPaths = num_paths;

	dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;

#if PERF_ANALYSIS
	cudaEventRecord(beginEvent);
#endif

	while (!iterationComplete) {

		// clean shading chunks
		cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));
		checkCUDAError("Dev Intersections");

		// tracing
		dim3 numblocksPathSegmentTracing = (currNumPaths + blockSize1d - 1) / blockSize1d;

		computeIntersectionsSAR << <numblocksPathSegmentTracing, blockSize1d >> > (
			depth
			, currNumPaths
			, dev_paths
			, dev_geoms
			, hst_scene->geoms.size()
			, dev_intersections
			);
		checkCUDAError("trace one bounce");

		cudaDeviceSynchronize();

		// TODO:
		// --- Shading Stage ---
		// Shade path segments based on intersections and generate new rays by
		// evaluating the BSDF.
		// Start off with just a big kernel that handles all the different
		// materials you have in the scenefile.
		// TODO: compare between directly shading the path segments and shading
		// path segments that have been reshuffled to be contiguous in memory.

		// 1. sort pathSegments by material type
		// This becomes very slow?
		// 2. shade the ray and spawn new path segments using BSDF
		// this function generates a new ray to replace the old one using BSDF

#if SARNAIVE
		++depth;
		kernComputeShadeSAR << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			currNumPaths,
			dev_intersections,
			dev_paths,
			dev_materials,
			depth,
			cam
);
		if (depth > 1) {
			kernComputeBlockToCameraSAR << <numblocksPathSegmentTracing, blockSize1d >> > (
				depth
				, currNumPaths
				, dev_paths
				, dev_geoms
				, hst_scene->geoms.size()
				, dev_intersections
				);
		}
		
#else
		kernComputeShade << <numblocksPathSegmentTracing, blockSize1d >> > (
			iter,
			currNumPaths,
			dev_intersections,
			dev_paths,
			dev_materials
			, dev_textureObjs
			, dev_textureChannels
			);
#endif
		
		cudaDeviceSynchronize();
		// 4. remove_if sorts all contents such that useless paths are all at the end.
		// if the remainingBounces = 0 (any material that doesn't hit anything or number of depth is at its limit)
		dev_path_end = thrust::stable_partition(thrust::device, dev_paths, dev_paths + currNumPaths, invalid_intersection());

		// nothing shows up if i set it out side of the if/else statement.
		currNumPaths = dev_path_end - dev_paths;
		// printf("curNum Paths: %i \n", currNumPaths);

		// don't need to remove intersections because new intersections will be computed based on sorted dev_paths
		// thrust uses exclusive start and end pointers, so if end pointer is the same as start pointer, we have no rays left.
		if (currNumPaths < 1)
		{
			iterationComplete = true;
		}


		if (depth == traceDepth) {
			iterationComplete = true;
		}

		if (guiData != NULL)
		{
			guiData->TracedDepth = depth;
			//guiData->TracedDepth = 8;
		}
	}


	kernUpdateLength << <numBlocksPixels, blockSize1d >> > (num_paths, dev_paths);
	//calculate maxRange and minRange to form image in azimuth-range plane
	PathSegment* longestSeg = thrust::max_element(thrust::device, dev_paths, dev_paths + num_paths, compare_range());
	PathSegment* shortestSeg = thrust::min_element(thrust::device, dev_paths, dev_paths + num_paths, compare_range());
	PathSegment host_maxRange;
	PathSegment host_minRange;
	cudaMemcpy(&host_maxRange, longestSeg, sizeof(PathSegment), cudaMemcpyDeviceToHost);
	cudaMemcpy(&host_minRange, shortestSeg, sizeof(PathSegment), cudaMemcpyDeviceToHost);
	
	printf("minRange: %f \n", host_minRange.length);
	printf("maxRange: %f \n", host_maxRange.length);
	
	for (int i = 1; i <= traceDepth; ++i) {
		if (i == 2) {
			continue;
		}
		//kernTransToAzimuthRange << <numBlocksPixels, blockSize1d >> > (num_paths, dev_paths, host_maxRange.length, host_minRange.length, cam.resolution.x, cam.resolution.y, i);
	}

	for (int i = 1; i <= traceDepth; ++i) {
		if (i == 2) {
			continue;
		}
		finalGather << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image, dev_paths, i, cam);
	}

	
	cudaDeviceSynchronize(); // maybe dont need

	kernFinalColorClean << <numBlocksPixels, blockSize1d >> > (num_paths, dev_image);

	///////////////////////////////////////////////////////////////////////////

	// Send results to OpenGL buffer for rendering
	sendImageToPBO << <blocksPerGrid2d, blockSize2d >> > (pbo, cam.resolution, iter, dev_image);

	cudaDeviceSynchronize(); // maybe dont need

	// Retrieve image from GPU
	cudaMemcpy(hst_scene->state.image.data(), dev_image,
		pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

	checkCUDAError("pathtrace");
}