#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "cuda_runtime.h"

/**
 * Handy-dandy hash function that provides seeds for random number generation.
 */
__host__ __device__ inline unsigned int utilhash(unsigned int a) {
    a = (a + 0x7ed55d16) + (a << 12);
    a = (a ^ 0xc761c23c) ^ (a >> 19);
    a = (a + 0x165667b1) + (a << 5);
    a = (a + 0xd3a2646c) ^ (a << 9);
    a = (a + 0xfd7046c5) + (a << 3);
    a = (a ^ 0xb55a4f09) ^ (a >> 16);
    return a;
}

// CHECKITOUT
/**
 * Compute a point at parameter value `t` on ray `r`.
 * Falls slightly short so that it doesn't intersect the object it's hitting.
 */
__host__ __device__ glm::vec3 getPointOnRay(Ray r, float t) {
    return r.origin + (t - .0001f) * glm::normalize(r.direction);
}

/**
 * Multiplies a mat4 and a vec4 and returns a vec3 clipped from the vec4.
 */
__host__ __device__ glm::vec3 multiplyMV(glm::mat4 m, glm::vec4 v) {
    return glm::vec3(m * v);
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed cube. Untransformed,
 * the cube ranges from -0.5 to 0.5 in each axis and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float boxIntersectionTest(Geom box, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;
}

// CHECKITOUT
/**
 * Test intersection between a ray and a transformed sphere. Untransformed,
 * the sphere always has radius 0.5 and is centered at the origin.
 *
 * @param intersectionPoint  Output parameter for point of intersection.
 * @param normal             Output parameter for surface normal.
 * @param outside            Output param for whether the ray came from outside.
 * @return                   Ray parameter `t` value. -1 if no intersection.
 */
__host__ __device__ float sphereIntersectionTest(Geom sphere, Ray r,
        glm::vec3 &intersectionPoint, glm::vec3 &normal, bool &outside) {
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0) {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0) {
        return -1;
    } else if (t1 > 0 && t2 > 0) {
        t = min(t1, t2);
        outside = true;
    } else {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside) {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ float boundBoxIntersectionTest(Geom* geom, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside) {
    Ray q;
    q.origin = multiplyMV(geom->inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom->inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 bbmin = geom->bound.minCorner;
    glm::vec3 bbmax = geom->bound.maxCorner;

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz) {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/ {
            // divide by 2 if everything goes wrong
            float t1 = (bbmin[xyz] - q.origin[xyz]) / qdxyz;
            float t2 = (bbmax[xyz] - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin) {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax) {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0) {
        outside = true;
        if (tmin <= 0) {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(geom->transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(geom->invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }
    return -1;

}

#if BUMP_MAP
__device__ float triangleIntersectionTest(Geom* geom, Triangle* triangle, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2& uv, cudaTextureObject_t& texObject, Texture& tex, bool& outside) {

    glm::vec3 screenPA = glm::vec3(geom->transform * triangle->pointA.pos);
    glm::vec3 screenPB = glm::vec3(geom->transform * triangle->pointB.pos);
    glm::vec3 screenPC = glm::vec3(geom->transform * triangle->pointC.pos);

    glm::vec3 baryPosition;

    bool doesIntersect = glm::intersectRayTriangle(r.origin, r.direction, screenPA, screenPB, screenPC, baryPosition);

    float u = baryPosition.r;
    float v = baryPosition.g;
    float t = baryPosition.b;

    if (!doesIntersect) {
        return -1.0f;
    }

    intersectionPoint = getPointOnRay(r, t);

    // calculate bump map value
    float4 texColor = tex2D<float4>(texObject, uv[0], uv[1]);
    glm::vec3 pointColor = glm::vec3(texColor.x, texColor.y, texColor.z);

    float uOffset = 1.f / tex.width;
    float vOffset = 1.f / tex.height;

    // calculate right neighbor uv:
    glm::vec2 rightUV = glm::vec2(u + uOffset, v);

    // subtract color from its right neighbor
    float4 rightColor = tex2D<float4>(texObject, rightUV[0], rightUV[1]);
    glm::vec3 rightPointColor = glm::vec3(rightColor.x, rightColor.y, rightColor.z);
    glm::vec3 colorDiffRight = pointColor - rightPointColor;

    // calculate down neighbor uv:
    glm::vec2 downUV = glm::vec2(u, v + vOffset);

    // subtract color from its down neighbor
    float4 downColor = tex2D<float4>(texObject, downUV[0], downUV[1]);
    glm::vec3 downPointColor = glm::vec3(downColor.x, downColor.y, downColor.z);
    glm::vec3 colorDiffDown = pointColor - downPointColor;

    glm::vec3 prevNormal = glm::vec3((1 - u - v) * triangle->pointA.nor + u * triangle->pointB.nor + v * triangle->pointC.nor);

    glm::vec3 tangent = cross(prevNormal, r.direction);

    normal = glm::normalize(prevNormal + colorDiffDown * cross(prevNormal, glm::vec3(1, 0, 0)) + colorDiffRight * cross(prevNormal, glm::vec3(0, 1, 0)));

    if (geom->textureid != -1) {
        uv = glm::vec2((1 - u - v) * triangle->pointA.uv + u * triangle->pointB.uv + v * triangle->pointC.uv);
    }

    if (!outside) {
        normal *= -1.f;
    }

    return t;

}
#else
__host__ __device__ float triangleIntersectionTest(Geom* geom, Triangle* triangle, Ray r,
    glm::vec3& intersectionPoint, glm::vec3& normal, glm::vec2 &uv, glm::vec4 &tangent, bool& outside) {

    glm::vec3 screenPA = glm::vec3(geom->transform * triangle->pointA.pos);
    glm::vec3 screenPB = glm::vec3(geom->transform * triangle->pointB.pos);
    glm::vec3 screenPC = glm::vec3(geom->transform * triangle->pointC.pos);

    //printf("pa: %f, %f, %f \n", screenPA.x, screenPA.y, screenPA.z);

    float debugax = screenPA.x;
    float debugay = screenPA.y;
    float debugaz = screenPA.z;

    glm::vec3 baryPosition;

    bool doesIntersect = glm::intersectRayTriangle(r.origin, r.direction, screenPA, screenPB, screenPC, baryPosition);

    float u = baryPosition.r;
    float v = baryPosition.g;
    float t = baryPosition.b;

    if (!doesIntersect) {
        return -1.0f;
    }

    intersectionPoint = getPointOnRay(r, t);

    normal = glm::vec3((1 - u- v) * triangle->pointA.nor + u * triangle->pointB.nor + v * triangle->pointC.nor);
 
#if LOAD_OBJ
    if (geom->objTexId != -1) {
        uv = glm::vec2((1 - u - v) * triangle->pointA.dev_uvs[0] + u * triangle->pointB.dev_uvs[0] +  v * triangle->pointC.dev_uvs[0]);
    }
#endif

#if LOAD_GLTF
    // just use normal uv values for now and not worry about the other texcoords
    uv = glm::vec2((1 - u - v) * triangle->pointA.dev_uvs[0] + u * triangle->pointB.dev_uvs[0] + v * triangle->pointC.dev_uvs[0]);
    tangent = glm::vec4((1 - u - v) * triangle->pointA.tan + u * triangle->pointB.tan + v * triangle->pointC.tan);
#endif

    if (!outside) {
        normal *= -1.f;
    }

    return t;

}
#endif

__host__ __device__ float squarePlaneIntersectionTest(Camera cam, Ray r, int& pixelIdxX, int& pixelIdxY) {
    //transform the ray
    Ray rInv;
    rInv.origin = multiplyMV(cam.inverseTransform, glm::vec4(r.origin, 1));
    rInv.direction = glm::mat3(cam.inverseTransform) * r.direction;
    float denominator = glm::dot(glm::vec3(0, 0, -1), rInv.direction);
    if (denominator == 0.f) {
        return -1;
    }
    float t = glm::dot(glm::vec3(0, 0, -1), (glm::vec3(0.f, 0.f, 0.f) - rInv.origin)) / glm::dot(glm::vec3(0, 0, -1), rInv.direction);
    //normal = glm::mat3(cam.transform) * glm::vec3(0, 0, -1);
    glm::vec3 intersectionPoint = rInv.origin + rInv.direction * t;
    if (t > 0 && intersectionPoint.x >= -1.f && intersectionPoint.x <= 1.f && intersectionPoint.y >= -1.f && intersectionPoint.y <= 1.f)
    {
        pixelIdxX = glm::floor(((1.f - intersectionPoint.x) * 0.5) * cam.resolution.x); //when intersectionPoint.x = -1, pixelIdx = max
        pixelIdxY = glm::floor(((1.f - intersectionPoint.y) * 0.5) * cam.resolution.y);
        return t;
    }
    return -1;
}
