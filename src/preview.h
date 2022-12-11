#pragma once

#include "main.h"

extern GLuint pbo;

std::string currentTimeString();
bool init(OnLoadNewScene fpOnLoadNewScene);
void mainLoop();

bool MouseOverImGuiWindow();
void InitImguiData(GuiDataContainer* guiData);