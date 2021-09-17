#ifndef BLOCK_CUT_FOREST_H
#define BLOCK_CUT_FOREST_H

#include <vector>

#include "types.h"
#include "CompressedData.h"
#include "DFS.h"

// A forest consisting of a block-cut tree for each component.
struct BlockCutForest
{
	// Sorted by vertex index (ascending).
	std::vector<v_size_t> vertToComp;
	// Indices in this list are referred to as component indices.
	// Sorted by size (ascending), then lexicographically (ascending).
	// Each list is sorted by vertex index (ascending).
	std::vector<std::vector<v_size_t>> compToVerts;

	// Sorted by vertex index (ascending).
	// Each list is sorted by bicomponent index (ascending).
	std::vector<std::vector<v_size_t>> vertToBicomps;
	// Indices in this list are referred to as bicomponent indices.
	// Sorted by cut point count (ascending), then by total size (ascending), then lexicographically (ascending).
	// Thus, leaves are grouped at the beginning.
	// Each list is sorted by the number of bicomponents vertices are in (descending), then by vertex index (ascending).
	// Thus, cut points are grouped at the beginning of each list.
	std::vector<std::vector<v_size_t>> bicompToVerts;
	// Sorted by bicomponent index (ascending).
	std::vector<v_size_t> bicompToCutCount;

	// Sorted by bicomponent index (ascending).
	std::vector<v_size_t> bicompToComp;
	// Sorted by component index (ascending).
	// Each list is sorted by bicomponent index (ascending).
	// Thus, leaves are grouped at the beginning of each list.
	std::vector<std::vector<v_size_t>> compToBicomps;
	// Sorted by component index (ascending).
	std::vector<v_size_t> compToLeafCount;

	explicit inline BlockCutForest(const CompressedData& data) :
		vertToComp(data.vct),
		vertToBicomps(data.vct)
	{
		// DFS computes unmapped versions of compToVerts, vertToComp, bicompToVerts, vertToBicomps, compToBicomps, and bicompToComp.
		// Of these, compToVerts, bicompToVerts, bicompToComp, and compToBicomps are stored in an auxiliary vector until sorting.
		// vertToComp and vertToBicomps are stored in their final location.
		// Sorting happens after the dfs phase.
		// Then, bicompToCutCount, compToBicomps, bicompToComp, and compToLeafCount are computed.
		// This entire constructor runs in O(m lg n) time with m edges and n vertices.

		std::vector<std::vector<v_size_t>> unmappedCompToVerts;
		std::vector<std::vector<v_size_t>> unmappedBicompToVerts;
		std::vector<v_size_t> unmappedBicompToComp;
		std::vector<std::vector<v_size_t>> unmappedCompToBicomps;

		//////////////////
		// PHASE 1: DFS //
		//////////////////

		v_size_t compIdx;
		v_size_t numI = 0;
		std::vector<v_size_t> num(data.vct, 0);
		std::vector<v_size_t> low(data.vct, 0);
		std::vector<std::pair<v_size_t, v_size_t>> edgeStack;
		std::vector<mask_t> inBicomp(data.vct, false);

		const auto visitStart = [&](v_size_t u, v_size_t v) -> void
		{
			low[v] = num[v] = ++numI;
			if (u == v)
			{
				compIdx = unmappedCompToVerts.size();
				unmappedCompToVerts.emplace_back();
				unmappedCompToBicomps.emplace_back();
			}
			else
				edgeStack.emplace_back(u, v);
			unmappedCompToVerts.back().push_back(v);
			vertToComp[v] = static_cast<v_size_t>(unmappedCompToVerts.size()) - 1;
		};

		const auto visitEnd = [&](v_size_t u, v_size_t v) -> void
		{
			v_size_t bicompIdx = unmappedBicompToVerts.size();
			if (u == v)
			{
				// if this is the only vertex in its component, add it as a singleton bicomponent
				if (unmappedCompToBicomps.back().empty())
				{
					unmappedCompToBicomps.back().push_back(bicompIdx);
					vertToBicomps[v].push_back(bicompIdx);
					unmappedBicompToComp.push_back(compIdx);
					unmappedBicompToVerts.push_back(std::vector<v_size_t>{v});
				}
				return;
			}
			low[u] = std::min(low[u], low[v]);
			// If v is an articulation point, add a new biconnected component.
			if (low[v] < num[u])
				return;

			unmappedBicompToVerts.emplace_back();
			std::pair<v_size_t, v_size_t> edge;

			unmappedBicompToComp.push_back(compIdx);
			unmappedCompToBicomps.back().push_back(bicompIdx);

			// For each edge on the stack until u-v, add it to the current bicomponent if not already added
			// edge.first will be added as edge.second lower on the stack.
			while (edge = edgeStack.back(), edgeStack.pop_back(), num[edge.first] >= num[v])
				if (!inBicomp[edge.second])
				{
					unmappedBicompToVerts.back().push_back(edge.second);
					vertToBicomps[edge.second].push_back(bicompIdx);
					inBicomp[edge.second] = true;
				}

			assert(edge == std::make_pair(u, v));

			if (!inBicomp[v])
			{
				unmappedBicompToVerts.back().push_back(v);
				vertToBicomps[v].push_back(bicompIdx);
			}

			if (!inBicomp[u])
			{
				unmappedBicompToVerts.back().push_back(u);
				vertToBicomps[u].push_back(bicompIdx);
			}

			// Reset inBicomp for the next bicomponent.
			for (v_size_t x : unmappedBicompToVerts.back())
				inBicomp[x] = false;
		};

		const auto isVisited = [&](v_size_t v) -> bool
		{
			return num[v] != 0;
		};

		const auto visitAgain = [&](v_size_t u, v_size_t v, v_size_t w) -> void
		{
			if (w == u || num[w] >= num[v]) return;
			edgeStack.emplace_back(v, w);
			low[v] = std::min(low[v], num[w]);
		};

		DFS dfs(visitStart, visitEnd, isVisited, visitAgain);
		dfs(data.adj, std::vector<mask_t>(data.vct, true));

		//////////////////////
		// PHASE 2: SORTING //
		//////////////////////

		v_size_t compCount = unmappedCompToVerts.size();
		v_size_t bicompCount = unmappedBicompToVerts.size();

		// Compute mapping for sorting components
		for (std::vector<v_size_t>& comp : unmappedCompToVerts)
			std::sort(comp.begin(), comp.end());
		// Maps sorted index -> unsorted index
		std::vector<v_size_t> compMap(compCount);
		std::iota(compMap.begin(), compMap.end(), 0);
		std::sort(compMap.begin(), compMap.end(),
			[&unmappedCompToVerts](v_size_t l, v_size_t r) -> bool
			{
				std::vector<v_size_t>& lComp = unmappedCompToVerts[l];
				std::vector<v_size_t>& rComp = unmappedCompToVerts[r];
				// By size (ascending)
				if (lComp.size() != rComp.size())
					return lComp.size() < rComp.size();
				// Lexicographically (ascending)
				return lComp < rComp;
			});

		// Maps unsorted index -> sorted index
		std::vector<v_size_t> invCompMap(compCount);
		for (v_size_t i = 0; i < compCount; i++)
			invCompMap[compMap[i]] = i;

		// Compute mapping for sorting bicomponents and the number of cut points in each bicomponent.
		for (std::vector<v_size_t>& bicomp : unmappedBicompToVerts)
			std::sort(bicomp.begin(), bicomp.end(),
				[&](v_size_t l, v_size_t r) -> bool
				{
					std::vector<v_size_t>& lBicomps = vertToBicomps[l];
					std::vector<v_size_t>& rBicomps = vertToBicomps[r];
					// By bicomponent count (descending)
					if (lBicomps.size() != rBicomps.size())
						return lBicomps.size() > rBicomps.size();
					// By index (ascending)
					return l < r;
				}
		);

		// The number of cut points in each bicomp.
		std::vector<v_size_t> unmappedBicompToCutCount(bicompCount, 0);
		for (std::vector<v_size_t>& bicomps : vertToBicomps)
		{
			assert(!bicomps.empty());
			if (bicomps.size() > 1)
				for (v_size_t bicomp : bicomps)
					unmappedBicompToCutCount[bicomp]++;
		}

		// Maps sorted index -> unsorted index
		std::vector<v_size_t> bicompMap(bicompCount);
		std::iota(bicompMap.begin(), bicompMap.end(), 0);
		std::sort(bicompMap.begin(), bicompMap.end(),
			[&](v_size_t l, v_size_t r) -> bool
			{
				v_size_t lCutCount = unmappedBicompToCutCount[l];
				v_size_t rCutCount = unmappedBicompToCutCount[r];
				// By cut point count (ascending)
				if (lCutCount != rCutCount)
					return lCutCount < rCutCount;
				std::vector<v_size_t> lBicomp = bicompToVerts[l];
				std::vector<v_size_t> rBicomp = bicompToVerts[r];
				// By size (ascending)
				if (lBicomp.size() != rBicomp.size())
					return lBicomp.size() < rBicomp.size();
				// Lexicographically (ascending)
				return lBicomp < rBicomp;
			}
		);

		// Maps unsorted index -> sorted index
		std::vector<v_size_t> invBicompMap(bicompCount);
		for (v_size_t i = 0; i < bicompCount; i++)
			invBicompMap[bicompMap[i]] = i;

		// Apply computed maps from comps
		compToVerts.reserve(compCount);
		compToBicomps.reserve(compCount);
		for (v_size_t i = 0; i < compCount; i++)
		{
			compToVerts.push_back(std::move(unmappedCompToVerts[compMap[i]]));
			compToBicomps.push_back(std::move(unmappedCompToBicomps[compMap[i]]));
		}

		// Apply computed maps from bicomps
		bicompToVerts.reserve(bicompCount);
		bicompToComp.reserve(bicompCount);
		for (v_size_t i = 0; i < bicompCount; i++)
		{
			bicompToVerts.push_back(std::move(unmappedBicompToVerts[bicompMap[i]]));
			bicompToComp.push_back(unmappedBicompToComp[bicompMap[i]]);
		}

		// Apply computed maps to comps
		for (v_size_t& comp : vertToComp)
			comp = invCompMap[comp];
		for (v_size_t& comp : bicompToComp)
			comp = invCompMap[comp];

		// Apply computed maps to bicomps
		for (std::vector<v_size_t>& bicomps : vertToBicomps)
			for (v_size_t& bicomp : bicomps)
				bicomp = invBicompMap[bicomp];
		for (std::vector<v_size_t>& bicomps : compToBicomps)
			for (v_size_t& bicomp : bicomps)
				bicomp = invBicompMap[bicomp];

		// Then, bicompToCutCount, compToBicomps, bicompToComp, and compToLeafCount are computed.

		// PHASE 3: 

		//std::cout << "";
	}
};

#endif