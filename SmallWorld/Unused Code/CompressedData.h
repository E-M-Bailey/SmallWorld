#ifndef COMPRESSED_DATA_H
#define COMPRESSED_DATA_H

#include <algorithm>
#include <numeric>
#include <vector>

#include "types.h"

struct CompressedData
{
	// Number of equivalence classes
	v_size_t vct;
	// Equivalence classes are in [0, srcct) iff their vertices are in src.
	v_size_t srcct;
	// Compressed adjacency structure
	std::vector<std::vector<v_size_t>> adj;
	// Cardinality of eqch equivalence class
	std::vector<v_size_t> mult;
	// Vertex weight of each equivalence class
	std::vector<weight_t> weights;
	// Equivalence class of each vertex (or MAX<uli> for masked-out vertices)
	std::vector<v_size_t> map;

	inline CompressedData(v_size_t vct, v_size_t srcct, std::vector<std::vector<v_size_t>>&& adj, std::vector<weight_t>&& weights, std::vector<v_size_t>&& mult, std::vector<v_size_t>&& map) :
		vct(vct),
		srcct(srcct),
		adj(adj),
		weights(weights),
		mult(mult),
		map(map)
	{}
};

// Removes masked-out vertices and converts the rest into an adjacency structure of structural equivalence classes.
// Runs in linearithmic time (negligible compared to Brandes' algorithm)
inline CompressedData compress(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask, const std::vector<mask_t>& src, const std::vector<weight_t>& weights)
{
	v_size_t vct = static_cast<v_size_t>(adj.size());

	// Masked adjacency structure
	std::vector<std::vector<v_size_t>> madj(vct);
	for (v_size_t v = 0; v < vct; v++)
		if (mask[v])
			for (v_size_t w : adj[v])
				if (mask[w])
					madj[v].push_back(w);

	// List of vertices where masked-out vertices are at the end, vertices in src are at the beginning, and structural equivalence classes are contiguous.
	std::vector<v_size_t> perm(vct);
	std::iota(perm.begin(), perm.end(), 0);
	std::sort(perm.begin(), perm.end(),
		[&](v_size_t l, v_size_t r) -> bool
		{
			// Masked-out vertices go to the end.
			if (!mask[l]) return false;
			if (!mask[r]) return true;
			// Vertices in src go to the beginning.
			if (src[l] != src[r]) return src[l] > src[r];
			// Else sort by vertex weight
			if (weights[l] != weights[r]) return weights[l] < weights[r];
			// Else sort by masked degree
			const std::vector<v_size_t>& lAdj = madj[l];
			const std::vector<v_size_t>& rAdj = madj[r];
			if (lAdj.size() != rAdj.size()) return lAdj.size() < rAdj.size();
			// Else sort lexicographically by masked neighborhood.
			return lAdj < rAdj;
			// l and r are structurally equivalent iff every compared quality is the same, so at this point order does not matter.
		});

	// Stores each vertex's equivalence class
	std::vector<v_size_t> map(vct, MAX<v_size_t>);
	// Stores a representative vertex of each equivalence class
	std::vector<v_size_t> reps;
	std::vector<weight_t> cweights;
	std::vector<v_size_t> mult;
	reps.reserve(vct);
	v_size_t csrcct = 0, p;
	// Compute map, reps, cweights, mult, cvct, and srcct.
	for (v_size_t i = 0; i < vct; i++)
	{
		v_size_t v = perm[i];
		if (!mask[v]) break;
		// If v starts a new equivalence class, increment counters
		if (i == 0 || src[v] != src[p] || weights[v] != weights[p] || madj[v] != madj[p])
		{
			reps.push_back(v);
			cweights.push_back(weights[v]);
			mult.push_back(0);
			csrcct += src[v];
		}
		v_size_t idx = static_cast<v_size_t>(reps.size()) - 1;
		map[v] = idx;
		mult[idx]++;
		p = v;
	}
	// Number of equivalence classes
	v_size_t cvct = static_cast<v_size_t>(reps.size());

	// Adjacency structrue of equivalence classes
	std::vector<std::vector<v_size_t>> cadj;
	cadj.reserve(cvct);
	for (v_size_t i = 0; i < cvct; i++)
	{
		v_size_t v = reps[i];
		std::vector<v_size_t>& vAdj = madj[v];
		// Map v's adjacency list
		for (v_size_t& w : vAdj)
			w = map[w];
		// Sort v's adjacency list with masked-out vertices at the end
		std::sort(vAdj.begin(), vAdj.end());
		// Find the end of the masked-in vertices in vAdj
		std::vector<v_size_t>::iterator last = vAdj.begin();
		while (last != vAdj.end() && *last != MAX<v_size_t>)
			last++;
		// Remove duplicate equivalence classes from v's adjacency list
		std::vector<v_size_t>::const_iterator clast = std::unique(vAdj.begin(), last);
		// Add v's adjacency list to the fuly compressed adjacency structure
		cadj.emplace_back(vAdj.cbegin(), clast);
	}

	return CompressedData(cvct, csrcct, std::move(cadj), std::move(cweights), std::move(mult), std::move(map));
}

#endif
