#define TINYOBJLOADER_IMPLEMENTATION // define this in only *one* .cc
#include "tiny_obj_loader.h"

#include "main.h"
#include "preview.h"
#include <cstring>
#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/vector_angle.hpp>

static std::string startTimeString;

// For camera controls
static bool leftMousePressed = false;
static bool rightMousePressed = false;
static bool middleMousePressed = false;
static double lastX;
static double lastY;

static bool camchanged = true;
static float dtheta = 0, dphi = 0;
static glm::vec3 cammove;


float zoom, theta, phi;
glm::vec3 cameraPosition;
glm::vec3 ogLookAt; // for recentering the camera

Scene* scene;
GuiDataContainer* guiData;
RenderState* renderState;
int iteration;

int width;
int height;

#define useSAR 0
#define constRayCast 1

//All these params are for SAR tracing
bool usingSAR;
bool copyCamera;
Trajectory* traj;


bool sceneIsReady;
bool sceneWasReset = false;

void LoadNewSceneFromFile(std::string& file)
{
	printf("Load me baby: %s", file.c_str());
	sceneIsReady = false;
	sceneWasReset = true;

	// debug print after loading scene, ensure the scene has a geometry. If it has geometry
	pathtraceFree();
	//cudaDeviceReset();

	printf("Creating new scene... \n");
	scene->~Scene();
	scene = new Scene(file);

	// Load scene file
	//scene = new Scene(sceneFile);

	//Create Instance for ImGUIData
	//guiData = new GuiDataContainer();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera& cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);


	sceneIsReady = true;

	guiData->TracedDepth = scene->state.traceDepth;
	printf("debug print just for pause purpose \n");

	InitImguiData(guiData);
	InitDataContainer(guiData);


	//initCuda();

	// INITCUDA
	//cudaGLSetGLDevice(0);


	// Clean up on program exit
	//atexit(cleanupCuda);

	//delete scene;
	//scene = nullptr;

	//// Initialize other stuff
	//initVAO();
	GLuint positionLocation = 0;
	GLuint texcoordsLocation = 1;

	GLfloat vertices[] = {
		-1.0f, -1.0f,
		1.0f, -1.0f,
		1.0f,  1.0f,
		-1.0f,  1.0f,
	};

	GLfloat texcoords[] = {
		1.0f, 1.0f,
		0.0f, 1.0f,
		0.0f, 0.0f,
		1.0f, 0.0f
	};

	GLushort indices[] = { 0, 1, 3, 3, 1, 2 };

	GLuint vertexBufferObjID[3];
	glGenBuffers(3, vertexBufferObjID);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)positionLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(positionLocation);

	glBindBuffer(GL_ARRAY_BUFFER, vertexBufferObjID[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(texcoords), texcoords, GL_STATIC_DRAW);
	glVertexAttribPointer((GLuint)texcoordsLocation, 2, GL_FLOAT, GL_FALSE, 0, 0);
	glEnableVertexAttribArray(texcoordsLocation);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, vertexBufferObjID[2]);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

	//initTextures();
	GLuint displayImage;

	glGenTextures(1, &displayImage);
	glBindTexture(GL_TEXTURE_2D, displayImage);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_BGRA, GL_UNSIGNED_BYTE, NULL);

	//initCuda();

	//initPBO();
	int num_texels = width * height;
	int num_values = num_texels * 4;
	int size_tex_data = sizeof(GLubyte) * num_values;

	// Generate a buffer ID called a PBO (Pixel Buffer Object)
	glGenBuffers(1, &pbo);

	// Make this the current UNPACK buffer (OpenGL is state-based)
	glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);

	// Allocate data for the buffer. 4-channel 8-bit image
	glBufferData(GL_PIXEL_UNPACK_BUFFER, size_tex_data, NULL, GL_DYNAMIC_COPY);
	cudaGLRegisterBufferObject(pbo);
}

//-------------------------------
//-------------MAIN--------------
//-------------------------------

int main(int argc, char** argv) {
	usingSAR = false;
#if useSAR
	usingSAR = true;
#endif
	startTimeString = currentTimeString();

	if (argc < 2) {
		printf("Usage: %s SCENEFILE.txt\n", argv[0]);
		return 1;
	}

	const char* sceneFile = argv[1];

	// Load scene file
	scene = new Scene(sceneFile);

	//Create Instance for ImGUIData
	guiData = new GuiDataContainer();

	// Set up camera stuff from loaded path tracer settings
	iteration = 0;
	renderState = &scene->state;
	Camera& cam = renderState->camera;
	width = cam.resolution.x;
	height = cam.resolution.y;

	glm::vec3 view = cam.view;
	glm::vec3 up = cam.up;
	glm::vec3 right = glm::cross(view, up);
	up = glm::cross(right, view);

	cameraPosition = cam.position;

	// compute phi (horizontal) and theta (vertical) relative 3D axis
	// so, (0 0 1) is forward, (0 1 0) is up
	glm::vec3 viewXZ = glm::vec3(view.x, 0.0f, view.z);
	glm::vec3 viewZY = glm::vec3(0.0f, view.y, view.z);
	phi = glm::acos(glm::dot(glm::normalize(viewXZ), glm::vec3(0, 0, -1)));
	theta = glm::acos(glm::dot(glm::normalize(viewZY), glm::vec3(0, 1, 0)));
	ogLookAt = cam.lookAt;
	zoom = glm::length(cam.position - ogLookAt);

	//Sets up the vehicle flight
	renderState->usingSar = usingSAR;
	if (usingSAR) {
		copyCamera = true;
		traj = &scene->traj;
		renderState->moveReceiver = false;
		constructTrajectory();
		initReceiver();
	}

	// Initialize CUDA and GL components
	init(LoadNewSceneFromFile);

	// Initialize ImGui Data
	InitImguiData(guiData);
	InitDataContainer(guiData);

	sceneIsReady = true;

	// GLFW main loop
	mainLoop();

	return 0;
}

void saveImage() {
	float samples = iteration;
	// output image file
	image img(width, height);

	for (int x = 0; x < width; x++) {
		for (int y = 0; y < height; y++) {
			int index = x + (y * width);
			glm::vec3 pix = renderState->image[index];
			img.setPixel(width - 1 - x, y, glm::vec3(pix) / samples);
		}
	}

	std::string filename = renderState->imageName;
	std::ostringstream ss;
	ss << filename << "." << startTimeString << "." << samples << "samp";
	filename = ss.str();

	// CHECKITOUT
	img.savePNG(filename);
	//img.saveHDR(filename);  // Save a Radiance HDR file
}

void initializeSpotlightModel() {
	glm::vec3 closeLook = traj->targetLocationAvg - traj->closestApproach;
	closeLook.y = 0.0;
	float flightDist = glm::atan(traj->travelDistance / 2) * length(closeLook);
	closeLook = glm::normalize(closeLook);
	glm::vec3 yUp = glm::vec3(0.0, 1.0, 0.0);

	glm::vec3 flightPath = glm::normalize(glm::cross(closeLook, yUp));

	glm::vec3 startLoc = traj->closestApproach + flightDist * flightPath;
	glm::vec3 endLoc = traj->closestApproach - flightDist * flightPath;

	glm::vec3 travelVec = glm::normalize(endLoc - startLoc);
	float travelDistance = glm::distance(endLoc, startLoc);
	float stepSize = travelDistance / (traj->snapshotCount - 1);

	

	for (int i = 0; i < traj->snapshotCount; i++) {
		glm::vec3 antennaPos = travelVec * (stepSize * i) + startLoc;
		traj->vehicleTraj.push_back(antennaPos);
		traj->lookPositions.push_back(traj->targetLocationAvg);
		traj->rightVecs.push_back(travelVec);
	}
}

void constructTrajectory() {

	//Make these adjustable paramaters
	traj->iterations = 10;
	traj->mode = SPOTLIGHT;
	traj->snapshotCount = 100;
	traj->travelDistance = 20 * PI / 180.f;

	if (copyCamera) {
		traj->closestApproach = cameraPosition;
		traj->targetLocationAvg = ogLookAt;

		Camera* ref = &scene->state.camera;
		traj->antenna.fov = ref->fov;
		traj->antenna.resolution = ref->resolution;
		traj->closestApproach = ref->position;
		traj->targetLocationAvg = ref->lookAt;
		traj->antenna = *ref;
	}
	else {
		//TODO: Maybe add importable antenna model
	}

	switch (traj->mode)
	{
	case SPOTLIGHT:
		initializeSpotlightModel();
	default:
		break;
	}
}

void initReceiver() {
	//Geom newGeom;
	//Material receiverMat; 
	//int g_idx = scene->geoms.size();
	//int m_idx = scene->materials.size();
	//newGeom.materialid = m_idx;

	//renderState->receiverIndex = g_idx;
	//newGeom.type = CUBE;
	//newGeom.translation = glm::vec3(traj->antenna.position) *1.1f;
	//glm::vec3 u = glm::vec3(0.0, 0.0, 1.0);
	//glm::vec3 v = traj->antenna.view;
	//float x_angle = glm::acos(glm::dot(glm::vec2(u.y, u.z), glm::vec2(v.y, v.z))) * 180.f / PI;
	//float y_angle = glm::acos(glm::dot(glm::vec2(u.x, u.z), glm::vec2(v.x, v.z))) * 180.f / PI;
	//float z_angle = glm::acos(glm::dot(glm::vec2(u.x, u.y), glm::vec2(v.x, v.y))) * 180.f / PI;
	//newGeom.rotation = glm::vec3(x_angle, y_angle, z_angle);
	////this scal is arbitrary
	//newGeom.scale = glm::vec3(5.0, 5.0, .1);

	//newGeom.transform = utilityCore::buildTransformationMatrix(
	//	newGeom.translation, newGeom.rotation, newGeom.scale);
	//newGeom.inverseTransform = glm::inverse(newGeom.transform);
	//newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

	//receiverMat.emittance = 1.0;
	//receiverMat.color = glm::vec3(1.0, 1.0, 1.0);

	//scene->materials.push_back(receiverMat);
	//scene->geoms.push_back(newGeom);

	Geom newGeom = generateNewReceiverFromCamera(traj->antenna);
	Material receiverMat;

	int g_idx = scene->geoms.size();
	int m_idx = scene->materials.size();

	newGeom.materialid = m_idx;
	renderState->receiverIndex = g_idx;

	receiverMat.emittance = 1.0;
	receiverMat.color = glm::vec3(1.0, 1.0, 1.0);

	scene->materials.push_back(receiverMat);
	scene->geoms.push_back(newGeom);
	scene->numGeoms = g_idx + 1;
}

//Adjusts the camera pointer based on the inputs
void adjustCamera(glm::vec3 position, glm::vec3 lookAt, glm::vec3 rightVec, Camera* cam_ptr) {
	Camera& camera = *cam_ptr;
	camera.position = position;
	camera.lookAt = lookAt;
	camera.up = glm::normalize(glm::cross(lookAt - position, rightVec));
	camera.view = glm::normalize(camera.lookAt - camera.position);
	camera.right = glm::normalize(glm::cross(camera.view, camera.up));

	// Unclear if FOV needs adjustment
	//float yscaled = tan(fovy * (PI / 180));
	//float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
	//float fovx = (atan(xscaled) * 180) / PI;
	//camera.fov = glm::vec2(fovx, fovy);
	//camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
	//	2 * yscaled / (float)camera.resolution.y);
}

void runCuda() {
	if (camchanged) {
		iteration = 0;
		Camera& cam = renderState->camera;
		cameraPosition.x = zoom * sin(phi) * sin(theta);
		cameraPosition.y = zoom * cos(theta);
		cameraPosition.z = zoom * cos(phi) * sin(theta);

		cam.view = -glm::normalize(cameraPosition);
		glm::vec3 v = cam.view;
		glm::vec3 u = glm::vec3(0, 1, 0);//glm::normalize(cam.up);
		glm::vec3 r = glm::cross(v, u);
		cam.up = glm::cross(r, v);
		cam.right = r;

		cam.position = cameraPosition;
		cameraPosition += cam.lookAt;
		cam.position = cameraPosition;
		camchanged = false;
		
		//we assume camera's default plane's normal was point in -z, lay on x-y plane, with
		//size 2x2, located at the center.(thus, 4 points are (1, 1), (1, -1), (-1, 1), (-1, -1))
		//first create a vector orthogonal to both vectors
		glm::mat4 rot;
		float yscaled = tan(cam.fov.y * (PI / 180));
		float xscaled = (yscaled * cam.resolution.x) / cam.resolution.y;
		if (glm::vec3(0, 0, -1) == cam.view) {
			rot = glm::mat4(1.f);
		}
		else {
			glm::vec3 rotAxis = glm::cross(glm::vec3(0, 0, -1), cam.view);
			float angRad = glm::angle(glm::vec3(0, 0, -1), cam.view);
			rot = glm::rotate(glm::mat4(1.f), angRad, rotAxis);
		}
		glm::mat4 translation = glm::translate(glm::mat4(1.0), glm::vec3(cam.position));
		//the size of camera is determined by xscaled and yscaled, camera is by default 1 in front of the eye.
		glm::mat4 scale = glm::scale(glm::vec3(xscaled, yscaled, 1.f)); //1, 1, 1
		glm::mat4 trans = translation * rot * scale;
		cam.transform = trans;  //transform to global space
		cam.inverseTransform = glm::inverse(trans);  //transform to camera space
	}

	// Map OpenGL buffer object for writing from CUDA on a single GPU
	// No data is moved (Win & Linux). When mapped to CUDA, OpenGL should not use this buffer

	if (iteration == 0) {
		if (!sceneWasReset) {
			pathtraceFree();
		}
		pathtraceInit(scene);
	}

	if (usingSAR) {
		if (iteration < traj->iterations /** traj->snapshotCount*/) {

			if (iteration % traj->iterations == 0) {
				//std::cout << iteration << std::endl;
				//Adjust Camera Pos
				int idx = iteration / traj->iterations;
				renderState->moveReceiver = true;
				renderState->moveReceiver = true;
				adjustCamera(traj->vehicleTraj[idx], traj->lookPositions[idx], traj->rightVecs[idx], &scene->state.camera);
			}
			uchar4* pbo_dptr = NULL;
			iteration++;
			cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

			// execute the kernel
			int frame = 0;
			pathtrace(pbo_dptr, frame, iteration);
			renderState->moveReceiver = false;

			// unmap buffer object
			cudaGLUnmapBufferObject(pbo);
		}
		else {
			saveImage();
			pathtraceFree();
			cudaDeviceReset();
			exit(EXIT_SUCCESS);
		}
	}
	else {
		if (iteration < renderState->iterations) {
			uchar4* pbo_dptr = NULL;
			iteration++;
			cudaGLMapBufferObject((void**)&pbo_dptr, pbo);

			// execute the kernel
			int frame = 0;
			pathtrace(pbo_dptr, frame, iteration);

			// unmap buffer object
			cudaGLUnmapBufferObject(pbo);
		}
		else {
			saveImage();
			pathtraceFree();
			cudaDeviceReset();
			exit(EXIT_SUCCESS);
		}
	}
	
}

void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	if (action == GLFW_PRESS) {
		switch (key) {
		case GLFW_KEY_ESCAPE:
			saveImage();
			glfwSetWindowShouldClose(window, GL_TRUE);
			break;
		case GLFW_KEY_S:
			saveImage();
			break;
		case GLFW_KEY_SPACE:
			camchanged = true;
			renderState = &scene->state;
			Camera& cam = renderState->camera;
			cam.lookAt = ogLookAt;
			break;
		}
	}
}

void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods) {
	if (MouseOverImGuiWindow())
	{
		return;
	}
	leftMousePressed = (button == GLFW_MOUSE_BUTTON_LEFT && action == GLFW_PRESS);
	rightMousePressed = (button == GLFW_MOUSE_BUTTON_RIGHT && action == GLFW_PRESS);
	middleMousePressed = (button == GLFW_MOUSE_BUTTON_MIDDLE && action == GLFW_PRESS);
}

void mousePositionCallback(GLFWwindow* window, double xpos, double ypos) {
	if (xpos == lastX || ypos == lastY) return; // otherwise, clicking back into window causes re-start
	if (leftMousePressed) {
		// compute new camera parameters
		phi -= (xpos - lastX) / width;
		theta -= (ypos - lastY) / height;
		theta = std::fmax(0.001f, std::fmin(theta, PI));
		camchanged = true;
	}
	else if (rightMousePressed) {
		zoom += (ypos - lastY) / height;
		zoom = std::fmax(0.1f, zoom);
		camchanged = true;
	}
	else if (middleMousePressed) {
		renderState = &scene->state;
		Camera& cam = renderState->camera;
		glm::vec3 forward = cam.view;
		forward.y = 0.0f;
		forward = glm::normalize(forward);
		glm::vec3 right = cam.right;
		right.y = 0.0f;
		right = glm::normalize(right);

		cam.lookAt -= (float)(xpos - lastX) * right * 0.01f;
		cam.lookAt += (float)(ypos - lastY) * forward * 0.01f;
		camchanged = true;
	}
	lastX = xpos;
	lastY = ypos;
}
