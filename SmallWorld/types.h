#ifndef TYPES_H
#define TYPES_H

#include <limits>

// Change n_type_t if more than 2^7-1 majors are required.
// Change v_size_t if more than 2^16-1 vertices are required.
// Change e_size_t if more than 2^32-1 edges are required.

// Vertex type
typedef unsigned char v_type_t;

// Vertex size
typedef unsigned short v_size_t;

// Edge size
typedef unsigned int e_size_t;

// Course cost and weekly hours
typedef unsigned int cost_t;

// Vertex weight
typedef float weight_t;

// Type used in masks (not necessarily bool to avoid the slower std::vector<bool>)
typedef unsigned char mask_t;

// Type used in numbers present in input files
typedef unsigned int input_t;

template<typename T>
constexpr T MAX = std::numeric_limits<T>::max();
template<typename T>
constexpr T INF = std::numeric_limits<T>::infinity();

#endif
