#pragma once

#include <vector>
#include "scene.h"

enum Instruction {
	DOWN,
	CROSS,
	UP
};

void InitDataContainer(GuiDataContainer* guiData);
void pathtraceInit(Scene *scene);
void pathtraceFree();
Geom generateNewReceiverFromCamera(Camera cam);
void pathtrace(uchar4 *pbo, int frame, int iteration);
