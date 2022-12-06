CIS565 Final Project - Hardware Accelerated SAR Simulator
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* David Li, Xiaoyu Du, Di Lu
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti
* Timeline: This project was worked on from 11/7/2022 to 12/12/2022

## Introduction and Motivation

In 3D rendering, Pathtracing is a technique that generates realistic looking scenes/images by simulating light ray bounces. To that effect, SAR simulation works very similarly. Both processes cast rays from a starting point: In path tracing, the pinhole camera; In SAR simulation, the antenna which emits waves. This means operations to make an SAR simulator are "embarrassingly parallel" and can be optimized by the GPU. Our team lends our knowledge of GPU-based raytracing to better understand its applications outside of visible light-based image outputs. To that effect, we are not Radar experts and our research may not be entirely physically accurate. 

Our hope is that this can contribute to the development of more open-source SAR simulators that can be helpful for aircraft/vehicle designers. We also hope that these outputs can be used as easy data-gathering for any AI-based image-recognition for the typically blurry images produced by real SARs.

_BEST FINAL RENDER/IMAGE GALLERY GOES HERE_

_GLTF CREDITS GO HERE_

## Crash Course: Synthetic Aperture Radar

SAR (Synthetic Aperture Radar) is a type of Radar that is commonly used in military Aircrafts for its advantage in creating images that does not depend on lighting or weather. SARs emit radar waves and capture signals that are bounced back in order to construct an image. 

**Real World SARs**

Traditional radars are stationary, and they rely on the large wavelength of the radio waves emitted to penetrate harsh weather conditions and generate clear images. However, the disadvantage here is that longer wavelengths require a longer antenna to emit. It also takes a long time to reflect back to the antenna. With moving aircrafts, both of these conditions are not ideal. 

In order to bypass this issue, a smaller antenna is mounted onto a moving vehicle, and the vehicle takes various snapshots of the environment over a moving trajectory. This creates the illusion of a larger radar aperture, producing a "Synthetic Aperture" radar. the quality of the resulting image heavily depends on the length of the antenna and the wavelength of the radio signal. 

**SAR Simulation Research**

Modeling the behavior of materials and SAR rays appears to be a non-trivial task. Based on our research and readings, only diffuse surfaces and specular surfaces have mathematical models developed.

## Scene File Description

The scene files used in this project are laid out as blocks of text in this order: Materials, Textures (if any), and Objects in the scene. Each Object has a description of its translation, scale, and rotation, as well as which material it's using, and if it's a basic shape (such as sphere or cube), then that is also specified. If not a basic shape, then it specifies a path to its obj. If the Object also has a texture, then it will refer to the Id of the texture. Using Ids to keep track of scene attributes prevent over-copying of shared data between Objects.

## Core Features

### SAR Wave Simulation

We use the lambertian and specular reflection models to simulate the behavior of radar signal. We give the user the flexibility to adapt the lambertian and specular reflection property of the material to simulate the interaction of EM wave with different materials. 

### SAR Backscatter

Backscatter are radar signals that reflected back to the SAR sensor. signals can be directly backscatter to the sensor or have multiple bounces before reaching the SAR sensor. The amplitude and the range of the signal are recorded by the sensor in a 3D coordinate of azimuth(moving direction of the antenna), elevation(position of hitted object on the elevation plane) and range(the distance between antenna and the object hitted).

### Vehicle Movement

A key part of SAR simulation is the movement of the "vehicle" In the real world, the SAR is mounted on either an aircraft or a satilite. The SAR then bounces infared waves towards a target and "listens" for radar waves that have scattered back. From this, the SAR image can be constructed. 

In an effort to model this behavior. We use the parthtracer camera as both an antenna and receiver. Radar waves modeled as rays are shot from the camera, and are accumulated if they bounce back towards the camera. There are different SAR modes that can be modeled. The one we chose to focus on is called spotlight. Spotlight SAR involves keeping the camera focused on the same point as the vehicle moves. 

### Acceleration Structures

We are using a kd-tree as a bounding volume hierarchy. A kd-tree is a binary tree where each node is split along one axis-aligned hyperplane. The k in a kd tree represents the number of dimensions. So for our uses k is three. The root node is split along the x-axis. The next layer is split along Y, then Z and then back to X etc.

### GLTF Support

## Performance Analysis

## Bloopers! :)

