CUDA Path Tracer
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Di Lu
  * [LinkedIn](https://www.linkedin.com/in/di-lu-0503251a2/)
  * [personal website](https://www.dluisnothere.com/)
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

## Introduction

In this project, I implemented a CUDA path tracer for the GPU. Previously in Advanced Rendering, I implemented a Monte Carlo Path Tracer for the CPU. In this project, the path tracer is

## Core Features
1. Shading kernel with BSDF Evaluation for Diffuse and Specular
2. Path continuation/termination using Stream Compaction
3. Contiguous arrangement of materials based on materialId
4. First bounce caching

![](img/part1Final.png)
![](img/part1FinalSpecular.png)

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
### Arbitrary Mesh Loading with TinyObjLoader
### UV Texture and Bump Mapping

## Performance Analysis

## Bloopers! :)


