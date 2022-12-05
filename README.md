CIS565 Final Project - Hardware Accelerated SAR Simulator
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* David Li, Xiaoyu Du, Di Lu
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

## Introduction and Motivation

In 3D rendering, Pathtracing is a technique that generates realistic looking scenes/images by simulating light ray bounces. For this project, I implemented a CUDA path tracer for the GPU. In order to get the least noisy final output, 5000 calls to pathtrace are made whenever the camera is moved. The result of all 5000 pathtrace calls are then averaged to produce the final output. For each call to pathtrace, the light rays in the scene will bounce a maximum of 8 times.

For this pathtracer, we parallelize operations by Rays (AKA Path Segments), and made sure to sync all threads before moving on to the next parallel operation.

Overall, this project is a continuation of learning how to write CUDA kernel functions, optimize performance by adding memory coalescence, and very simple acceleration structures. The second part of the project introduced me to using TinyObjLoader, CudaTextureObjects, and various rendering techniques to get specific visual effects.

_GLTF CREDITS GO HERE_

## Scene File Description

The scene files used in this project are laid out as blocks of text in this order: Materials, Textures (if any), and Objects in the scene. Each Object has a description of its translation, scale, and rotation, as well as which material it's using, and if it's a basic shape (such as sphere or cube), then that is also specified. If not a basic shape, then it specifies a path to its obj. If the Object also has a texture, then it will refer to the Id of the texture. Using Ids to keep track of scene attributes prevent over-copying of shared data between Objects.

## Core Features

**SAR Wave Simulation**

**SAR Backscatter**

**Vehicle Movement**

**Acceleration Structures**

**GLTF Support**

## Performance Analysis

## Bloopers! :)

