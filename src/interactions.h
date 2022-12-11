#pragma once

#include "intersections.h"
#include "stb_image.h"
#include "cuda_runtime.h"

// CHECKITOUT
/**
 * Computes a cosine-weighted random direction in a hemisphere.
 * Used for diffuse lighting.
 */
__host__ __device__
glm::vec3 calculateRandomDirectionInHemisphere(
        glm::vec3 normal, thrust::default_random_engine &rng) {
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(1, 0, 0);
    } else if (abs(normal.y) < SQRT_OF_ONE_THIRD) {
        directionNotNormal = glm::vec3(0, 1, 0);
    } else {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

/**
 * Scatter a ray with some probabilities according to the material properties.
 * For example, a diffuse surface scatters in a cosine-weighted hemisphere.
 * A perfect specular surface scatters in the reflected ray direction.
 * In order to apply multiple effects to one surface, probabilistically choose
 * between them.
 *
 * The visual effect you want is to straight-up add the diffuse and specular
 * components. You can do this in a few ways. This logic also applies to
 * combining other types of materias (such as refractive).
 *
 * - Always take an even (50/50) split between a each effect (a diffuse bounce
 *   and a specular bounce), but divide the resulting color of either branch
 *   by its probability (0.5), to counteract the chance (0.5) of the branch
 *   being taken.
 *   - This way is inefficient, but serves as a good starting point - it
 *     converges slowly, especially for pure-diffuse or pure-specular.
 * - Pick the split based on the intensity of each material color that you hit, and divide
 *   branch result by that branch's probability (whatever probability you use).
 *
 * This method applies its changes to the Ray parameter `ray` in place.
 * It also modifies the color `color` of the ray in place.
 *
 * You may need to change the parameter list for your purposes!
 */

// helper function to output new direction
__device__
glm::vec3 calculatePbrMetallicRoughness(cudaTextureObject_t& metallicTexture,
    glm::vec3 reflection,
    glm::vec2 metallicUv,
    PathSegment& pathSegment,
    ShadeableIntersection& intersection,
    thrust::default_random_engine& rng) 
{
    float mu = metallicUv.x;
    float mv = metallicUv.y;

    //printf("tangent: %f, %f, %f, %f \n", intersection.tangent.x, intersection.tangent.y, intersection.tangent.z, intersection.tangent.w);

    float4 metallicRoughness = tex2D<float4>(metallicTexture, mu, mv);
    float rough = metallicRoughness.y;
    float metal = metallicRoughness.z;

    glm::vec3 newDirection = glm::vec3();

    thrust::uniform_real_distribution<float> u01(0, 1);
    float x1 = u01(rng);
    float x2 = u01(rng);

    float theta = atan(rough * sqrt(x1) / sqrt(1 - x1));
    float phi = 2 * PI * x2;

    newDirection.x = cos(phi) * sin(theta);
    newDirection.y = sin(phi) * sin(theta);
    newDirection.z = cos(theta);

    glm::mat3 worldToLocal2D;
    worldToLocal2D[2] = intersection.surfaceNormal;
    worldToLocal2D[1] = glm::vec3(intersection.tangent);
    worldToLocal2D[0] = glm::cross(intersection.surfaceNormal, worldToLocal2D[1]) * intersection.tangent.w;

    glm::vec3 r = glm::normalize(worldToLocal2D * reflection);

    glm::mat3 sampleToLocal;
    sampleToLocal[2] = r;
    sampleToLocal[0] = glm::normalize(glm::vec3(0, r.z, -r.y));
    sampleToLocal[1] = glm::cross(sampleToLocal[2], sampleToLocal[1]);

    glm::mat3 localToWorld = glm::inverse(worldToLocal2D);
    glm::mat3 sampleToWorld = localToWorld * sampleToLocal;

    newDirection = glm::normalize(sampleToWorld * newDirection);
    return newDirection;
    
}

__device__
void scatterRay(
        PathSegment & pathSegment,
        glm::vec3 intersectionPoint,
        ShadeableIntersection& intersection,
        const Material &m,
        cudaTextureObject_t* texObjects,
        int numChannels,
        thrust::default_random_engine &rng) {
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    // treat the rest as perfectly specular.
    // assuming there's only one light

    // based on float value of reflective and refractive
    // with float percent likelihood the ray goes reflective vs. refractive

    thrust::uniform_real_distribution<float> u01(0, 1);

    float randGen = u01(rng);

    glm::vec3 pointColor;

    // If base color exists
    if (m.pbrMetallicRoughness.baseColorOffset >= 0) {
        int baseColorTexId = m.pbrMetallicRoughness.baseColorOffset + m.pbrMetallicRoughness.baseColorIdx;
        // printf("baseColorOffset: %i, textureId: %i \n", m.pbrMetallicRoughness.baseColorOffset, intersection.textureId);
        float u = intersection.uv[0];
        float v = intersection.uv[1];

        float4 finalcol = tex2D<float4>(texObjects[baseColorTexId], u, v);

        //printf("finalCol: %f, %f, %f \n", finalcol.x, finalcol.y, finalcol.z);
        pointColor = glm::vec3(finalcol.x, finalcol.y, finalcol.z);

    }
    else if (intersection.textureId >= 0) {
        float u = intersection.uv[0];
        float v = intersection.uv[1];

        float4 finalcol = tex2D<float4>(texObjects[intersection.textureId], u, v);
        pointColor = glm::vec3(finalcol.x, finalcol.y, finalcol.z);

    }
    else {
        pointColor = m.color;
    }

    // 0.5 is a placeholder
    if (m.pbrMetallicRoughness.metallicRoughnessOffset >= 0) {
        // do metallic calculations and whatnot. If oBJ or none of these, then just move on to the reflective crap.
        glm::vec3 reflection = glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal);

        // just use intersection.uv for now and hope the textures align in the location.
        int metallicRoughnessTexId = m.pbrMetallicRoughness.metallicRoughnessOffset + m.pbrMetallicRoughness.metallicRoughnessIdx;

        glm::vec3 pbrDirection = calculatePbrMetallicRoughness(texObjects[metallicRoughnessTexId], reflection, intersection.uv, pathSegment, intersection, rng);
        Ray newRay = {
            intersectionPoint,
            pbrDirection
        };

        PathSegment newPath = {
            newRay,
            pointColor * pathSegment.color
        };
        newPath.pixelIndex = pathSegment.pixelIndex;
        newPath.remainingBounces = pathSegment.remainingBounces;

        pathSegment = newPath;

    } else if (randGen <= m.hasReflective) {
        // take a reflective ray
        glm::vec3 newDirection = glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal);
        Ray newRay = {
            intersectionPoint,
            newDirection
        };

        PathSegment newPath = {
            newRay,
            m.specular.color * pointColor * pathSegment.color * m.hasReflective
        };
        newPath.pixelIndex = pathSegment.pixelIndex;
        newPath.remainingBounces = pathSegment.remainingBounces;

        pathSegment = newPath;
    }
    else if (randGen <= m.hasReflective + m.hasRefractive) {
        // take a refractive ray
        float airIOR = 1.0f;
        float eta = airIOR / m.indexOfRefraction;

        float cosTheta = glm::dot(-1.f * pathSegment.ray.direction, intersection.surfaceNormal);

        // then entering
        bool entering = cosTheta > 0;

        if (!entering) {
            eta = 1.0f / eta; // invert eta
        }

        float sinThetaI = sqrt(1.0 - cosTheta * cosTheta);
        float sinThetaT = eta * sinThetaI;

        glm::vec3 newDirection = pathSegment.ray.direction;

        // if total internal reflection
        if (sinThetaT >= 1) {
            newDirection = glm::normalize(glm::reflect(pathSegment.ray.direction, intersection.surfaceNormal));
        }
        else {
            newDirection = glm::normalize(glm::refract(pathSegment.ray.direction, intersection.surfaceNormal, eta));
        }

        glm::vec3 newColor = pathSegment.color * pointColor * m.specular.color;

        Ray newRay = {
            intersectionPoint + 0.001f * pathSegment.ray.direction,
            newDirection
        };

        PathSegment newPath = {
            newRay,
            newColor
        };

        newPath.pixelIndex = pathSegment.pixelIndex;
        newPath.remainingBounces = pathSegment.remainingBounces;

        pathSegment = newPath;
    }
    else {
        // only diffuse
        glm::vec3 newDirection = calculateRandomDirectionInHemisphere(intersection.surfaceNormal, rng);
        Ray newRay = {
            intersectionPoint,
            newDirection
        };

        PathSegment newPath = {
            newRay,
            pointColor * pathSegment.color
        };

        newPath.pixelIndex = pathSegment.pixelIndex;
        newPath.remainingBounces = pathSegment.remainingBounces;

        pathSegment = newPath;
    }
}