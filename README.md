CIS565 Final Project - Hardware Accelerated SAR Simulator
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* David Li, Xiaoyu Du, Di Lu
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

## Introduction and Motivation

In 3D rendering, Pathtracing is a technique that generates realistic looking scenes/images by simulating light ray bounces. To that effect, SAR simulation works very similarly. Both processes cast rays from a starting point: In path tracing, the pinhole camera; In SAR simulation, the antenna which emits waves. This means operations to make an SAR simulator are embarrassingly parallel and can be optimized by the GPU. Our team lends our knowledge of GPU-based raytracing to better understand its applications outside of visible light-based image outputs. 

Our hope is that this can contribute to the development of more open-source SAR simulators that can be helpful for aircraft/vehicle designers. We also hope that these outputs can be used as easy data-gathering for any AI-based image-recognition for the typically blurry images produced by real SARs.

_BEST FINAL RENDER/IMAGE GALLERY GOES HERE_

_GLTF CREDITS GO HERE_

## Background: Synthetic Aperture Radar

SAR (Synthetic Aperture Radar) is a type of Radar commonly used in military Aircrafts for its advantage in creating images that does not depend on lighting or weather. SARs emit radar waves and capture signals that are bounced back in order to construct an image. 

## Scene File Description

The scene files used in this project are laid out as blocks of text in this order: Materials, Textures (if any), and Objects in the scene. Each Object has a description of its translation, scale, and rotation, as well as which material it's using, and if it's a basic shape (such as sphere or cube), then that is also specified. If not a basic shape, then it specifies a path to its obj. If the Object also has a texture, then it will refer to the Id of the texture. Using Ids to keep track of scene attributes prevent over-copying of shared data between Objects.

## Core Features

**SAR Wave Simulation**
We use the lambertian and specular reflection models to simulate the behavior of radar signal. We give the user the flexibility to adapt the lambertian and specular reflection property of the material to simulate the interaction of EM wave with different materials. 

**SAR Backscatter**
Backscatter are radar signals that reflected back to the SAR sensor. signals can be directly backscatter to the sensor or have multiple bounces before reaching the SAR sensor. The amplitude and the range of the signal are recorded by the sensor in a 3D coordinate of azimuth(moving direction of the antenna), elevation(position of hitted object on the elevation plane) and range(the distance between antenna and the object hitted).

**Vehicle Movement**

**Acceleration Structures**

**GLTF Support**

## Performance Analysis

## Bloopers! :)

