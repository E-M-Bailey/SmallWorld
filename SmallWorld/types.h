#ifndef TYPES_H
#define TYPES_H

// Change n_type_t if more than 2^7-1 majors are required.
// Change n_size_t if more than 2^16-1 vertices are required.
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

#endif
