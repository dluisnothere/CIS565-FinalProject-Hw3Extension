CIS565 Final Project - Hardware Accelerated SAR Simulator
================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Final Project**

* David Li, Xiaoyu Du, Di Lu
* Tested on: Windows 11, i7-12700H @ 2.30GHz 32GB, NVIDIA GeForce RTX 3050 Ti

## Introduction and Motivation

In 3D rendering, Pathtracing is a technique that generates realistic looking scenes/images by simulating light ray bounces. To that effect, SAR simulation works very similarly. Both processes cast rays from a starting point: In path tracing, the pinhole camera; In SAR simulation, the antenna which emits waves. This means operations on the SAR simulator are embarrassingly parallel. 

## Background: Synthetic Aperture Radar

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

