#pragma once

#include <glm/glm.hpp>
#include <glm/gtx/intersect.hpp>

#include "sceneStructs.h"
#include "utilities.h"
#include "cuda_runtime.h"

#define USE_REC 0
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

__host__ __device__ float boundBoxNodeIntersectionTest(Geom* geom, Ray r, glm::vec3& intersectionPoint, glm::vec3& normal, bool& outside, BoundBox bb) {
    Ray q;
    q.origin = multiplyMV(geom->inverseTransform, glm::vec4(r.origin, 1.0f));
    q.direction = glm::normalize(multiplyMV(geom->inverseTransform, glm::vec4(r.direction, 0.0f)));

    glm::vec3 bbmin = bb.minCorner;
    glm::vec3 bbmax = bb.maxCorner;

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
        uv = glm::vec2((1 - u - v) * triangle->pointA.uv + u * triangle->pointB.uv +  v * triangle->pointC.uv);
    }
#endif

#if LOAD_GLTF
    // just use normal uv values for now and not worry about the other texcoords
    glm::vec2 pointUvs = triangle->pointA.uv;
    if (pointUvs.x > -1) {
        uv = glm::vec2((1 - u - v) * triangle->pointA.uv + u * triangle->pointB.uv + v * triangle->pointC.uv);
    }

    if (tangent.length() > 0) {
        tangent = glm::vec4((1 - u - v) * triangle->pointA.tan + u * triangle->pointB.tan + v * triangle->pointC.tan);
    }
#endif

    if (!outside) {
        normal *= -1.f;
    }

    return t;

}

__host__ __device__ float treeIntersectionTest(
    Geom* geom
    , Ray r
    , glm::vec3& intersectionPoint
    , glm::vec3& normal
    , glm::vec2& uv
    , glm::vec4& tangent
    , bool& outside
    , KDNode* trees
    , int node_idx
    , int thread_idx) {

    

    bool hitObj; // for use in procedural texturing
    glm::vec3 tmp_intersect;
    glm::vec3 tmp_normal;
    glm::vec2 tmp_uv = glm::vec2(-1, -1);
    glm::vec4 tmp_tangent = glm::vec4(0, 0, 0, 0);
    bool tmpHitObj = false;
    bool changedTmin = false;

    float t_min = FLT_MAX;

    
    //printf("%d \n", node_idx);    
    //printf("1\n");
#if USE_REC  
    KDNode& node = trees[node_idx];
#if USE_KD_VEC
    
    float t = -1;
    for (int i = 0; i < node.numIndices; i++) { //KD_DEBUG
        /*if (node_idx == 1 && thread_idx == 87973) {
            int a = node.device_trisIndices[i];
            int b = geom->numTris;
            printf("1: tris index %d i %d\n", a, thread_idx);
        }

        if (node_idx == 2 && thread_idx == 413182) {
            int a = node.device_trisIndices[i];
            int b = geom->numTris;
            printf("2: tris index %d i %d\n", a, thread_idx);
        }*/
        
        
        t = triangleIntersectionTest(geom, &geom->device_tris[node.device_trisIndices[i]], r, tmp_intersect, tmp_normal, tmp_uv, outside);

        tmpHitObj = true;
        if (t > 0.0f && t_min > t)
        {
            /*if (node_idx == 1) {
                int a = node.device_trisIndices[i];
                int b = geom->numTris;
                printf("1 hit: tris index %d tris size %d\n", a, b);
            }

            if (node_idx == 2) {
                int a = node.device_trisIndices[i];
                int b = geom->numTris;
                printf("2 hit: tris index %d tris size %d\n", a, b);
            }*/
            t_min = t;
            intersectionPoint = tmp_intersect;
            normal = tmp_normal;
            uv = tmp_uv;
            hitObj = tmpHitObj;
            changedTmin = true;
        }
    }
#else
    float t = triangleIntersectionTest(geom, &geom->device_tris[node.trisIndex], r, tmp_intersect, tmp_normal, tmp_uv, outside);

    tmpHitObj = true;
    if (t > 0.0f && t_min > t)
    {
        t_min = t;
        intersectionPoint = tmp_intersect;
        normal = tmp_normal;
        uv = tmp_uv;
        hitObj = tmpHitObj;
        changedTmin = true;
        return t_min;
    }
#endif 
    if (changedTmin) {
        return t_min;
    }

    float t_near = -1;
    float t_far = -1;
    //printf("Node near %i node far %i node current %i\n", (int)node.near_node, (int)node.far_node, node_idx);
    if (node.near_node >= 0) {
        t_near = boundBoxNodeIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside, trees[node.near_node].bound);
    }
    if (node.far_node >= 0) {
        t_far = boundBoxNodeIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside, trees[node.far_node].bound);
    }

    //May need to check both case
    int first_node = -1;
    int secon_node = -1;
    if (node_idx == 0) {
        //printf("t_near: %f, t_far: %f Node: %i\n", t_near, t_far, node_idx);
    }
    if (t_near > -.01 && t_far > -.01) {
        //Check smaller first
        first_node = t_near < t_far ? node.near_node : node.far_node;
        secon_node = t_near >= t_far ? node.near_node : node.far_node;
        //printf("Check: %f %f, Node: %i %i, result: %i %i\n", t_near, t_far, node.near_node, node.far_node, first_node, secon_node);
    }
    else {
        //Check larger, either other one or both is negative -1
        first_node = t_near >= t_far ? node.near_node : node.far_node;
        //printf("Check: %f %f, Node: %i %i, result: %i\n", t_near, t_far, node.near_node, node.far_node, first_node);
    }
 

    if (first_node >= 0) {
    //if ((first_node >= 0 || true) && node.far_node != -1) {
        t = treeIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_uv, outside, trees, first_node, thread_idx);
        //t = treeIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_uv, outside, trees, node.far_node, thread_idx);

        tmpHitObj = true;
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            intersectionPoint = tmp_intersect;
            normal = tmp_normal;
            uv = tmp_uv;
            hitObj = tmpHitObj;
            changedTmin = true;
            return t_min;
        }
    }

    if (secon_node >= 0) {
        t = treeIntersectionTest(geom, r, tmp_intersect, tmp_normal, tmp_uv, outside, trees, secon_node, thread_idx);

        tmpHitObj = true;
        if (t > 0.0f && t_min > t)
        {
            t_min = t;
            intersectionPoint = tmp_intersect;
            normal = tmp_normal;
            uv = tmp_uv;
            hitObj = tmpHitObj;
            changedTmin = true;
            return t_min;
        }
    }
    if (changedTmin)
    {
        return t_min;
    }
#else 
//Can use stack to store triangles and nodes
    int nodeStack[512] = { -1 };
    int stack_ptr = 0;
    nodeStack[stack_ptr] = node_idx;
    int numTris = geom->numTris;
    /*if (thread_idx != 456415) {
        return -1;
    }*/
    int counter = 0;
    while (true) {
        if (stack_ptr >= 510) {
            printf("Previous values %d %d %d %d\n", nodeStack[stack_ptr - 3], nodeStack[stack_ptr - 2], nodeStack[stack_ptr - 1], thread_idx);
            printf("Positive Shit\n");
        }
        
        if (stack_ptr < 0) {
            return -1;
        }
        if (nodeStack[stack_ptr] < 0) {
            //printf("Stack Pointer about to access %i\n", stack_ptr);
            /*if (thread_idx == 456415) {
                printf("Popping %d with indes %d count %d\n", nodeStack[stack_ptr], stack_ptr, counter);
                counter++;
            }*/
            int compareTris = nodeStack[stack_ptr] * -1 - 1;
            float t = triangleIntersectionTest(geom, &geom->device_tris[compareTris], r, tmp_intersect, tmp_normal, tmp_uv, tmp_tangent, outside);

            tmpHitObj = true;
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                intersectionPoint = tmp_intersect;
                normal = tmp_normal;
                uv = tmp_uv;
                hitObj = tmpHitObj;
                changedTmin = true;
                return t_min;
            }
            stack_ptr--;
            
        }
        else {
            /*if (thread_idx == 456415) {
                printf("Popping %d with indes %d count %d\n", nodeStack[stack_ptr], stack_ptr, counter);
                counter++;
            }*/
            KDNode& node = trees[nodeStack[stack_ptr]];
            int trisIndex = node.trisIndex;
            int near_node = node.near_node;
            int far_node = node.far_node;
            stack_ptr--;

            float t_near = -1;
            float t_far = -1;

            if (near_node >= 0) {
                t_near = boundBoxNodeIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside, trees[near_node].bound);
            }
            if (far_node >= 0) {
                t_far = boundBoxNodeIntersectionTest(geom, r, tmp_intersect, tmp_normal, outside, trees[far_node].bound);
            }

            //May need to check both case
            int first_node = -1;
            int secon_node = -1;
            if (node_idx == 0) {
                //printf("t_near: %f, t_far: %f Node: %i\n", t_near, t_far, node_idx);
            }
            if (t_near > -.01 && t_far > -.01) {
                //Check smaller first
                first_node = t_near < t_far ? node.near_node : node.far_node;
                secon_node = t_near >= t_far ? node.near_node : node.far_node;
                //printf("Check: %f %f, Node: %i %i, result: %i %i\n", t_near, t_far, node.near_node, node.far_node, first_node, secon_node);
            }
            else {
                //Check larger, either other one or both is negative -1
                first_node = t_near >= t_far ? node.near_node : node.far_node;
                //printf("Check: %f %f, Node: %i %i, result: %i\n", t_near, t_far, node.near_node, node.far_node, first_node);
            }
            //printf("On Call: %d %d %d\n", secon_node, trisIndex * -1, first_node);
            if (secon_node >= 0) {
                stack_ptr++;
                /*if (thread_idx == 456415) {
                    printf("Pushing %d with indes %d count %d\n", secon_node, stack_ptr, counter);
                    counter++;
                }*/
                nodeStack[stack_ptr] = secon_node;
            }
            
            stack_ptr++;
            /*if (thread_idx == 456415) {
                printf("Pushing %d with indes %d count %d\n", trisIndex * -1, stack_ptr, counter);
                counter++;
            }*/
            nodeStack[stack_ptr] = (trisIndex + 1) * -1;

            if (first_node >= 0) {
                stack_ptr++;
                /*if (thread_idx == 456415) {
                    printf("Pushing %d with indes %d count %d\n", first_node, stack_ptr, counter);
                    counter++;
                }*/
                nodeStack[stack_ptr] = first_node;
            }
            
        }
    }


#endif

    return -1;
}

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
