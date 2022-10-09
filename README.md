CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Di Lu
  * [LinkedIn](https://www.linkedin.com/in/di-lu-0503251a2/)
  * [personal website](https://www.dluisnothere.com/)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

## Introduction

In 3D rendering, Pathtracing is a technique that generates realistic looking scenes/images by simulating light ray bounces. For this project, I implemented a CUDA path tracer for the GPU. In order to get the least noisy final output, 5000 calls to pathtrace are made whenever the camera is moved. The result of each pathtrace call is then averaged to produce the final output.

Overall, this project is a continuation of learning how to write CUDA kernel functions, optimize performance by adding memory coalescence, and very simple acceleration structures. The second part of the project introduced me to using TinyObjLoader, CudaTextureObjects, and various rendering techniques to get specific types of images:

1. Core Features: 
*  Simple Diffuse, Specular, and Imperfect Specular BSDF shading
* 
3. Additional Features

![](img/main.png)

![](img/mainAntialiasing.png)

## Core Features
1. Shading kernel with BSDF Evaluation for Diffuse and Perfect/Imperfect Specular
2. Path continuation/termination using Stream Compaction
3. Contiguous arrangement of materials based on materialId
4. First bounce caching

![](img/part1Final.png)
![](img/part1FinalSpecular.png)
![](img/imperfectSpecular.png)

## Additional Features
### Refractive Material

![](img/Refractive.png)

### Depth of Field

![](img/noDepthOfField.png)

![](img/depthFieldFinal.png)

### Stochastic Sampled Anti-Aliasing

![](img/noAntialiasing.png)

![](img/antialiasing5000samp.png)

### Direct Lighting

![](img/NoDirectLighting.png)

![](img/DirectLighting.png)

### Arbitrary Mesh Loading with TinyObjLoader

![](img/refractiveKitty.png)

### UV Texture, Procedural Texture and Bump Mapping

![](img/completeMario2.png)

![](img/myLink.png)

![](img/nobump.png)

![](img/yesBumpmap.png)

![](img/procedural.png)

## Performance Analysis

## Bloopers! :)

Too much antialiasing: me without my contacts

![](img/antialiasing1.png)
