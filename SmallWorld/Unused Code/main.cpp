#include <algorithm>
#include <assert.h>
#include <atomic>
#include <bitset>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <numeric>
#include <queue>
#include <random>
#include <stdint.h>
#include <string>
#include <unordered_map>
#include <vector>

#include "BlockCutForest.h"
#include "CompressedData.h"
#include "DFS.h"
#include "types.h"

#define DO_BC true
#define TIME_BC true
#define DO_BC_PROG true
#define NUM_THREADS 12

// Returns the number of elements v of vec such that mask[v] is true.
inline v_size_t maskedCount(const std::vector<v_size_t>& vec, const std::vector<mask_t>& mask)
{
	v_size_t count = 0;
	for (v_size_t v : vec)
		count += mask[v];
	return count;
}

// Returns a pair of the largest masked size in vecs and its first index.
inline std::pair<v_size_t, v_size_t> maxMaskedCount(const std::vector<std::vector<v_size_t>>& vecs, const std::vector<mask_t> mask)
{
	if (vecs.empty()) return { MAX<v_size_t>, MAX<v_size_t> };
	v_size_t maxIdx = 0, maxSize = maskedCount(vecs[0], mask);
	for (v_size_t i = 1; i < vecs.size(); i++)
	{
		v_size_t size = maskedCount(vecs[i], mask);
		if (size > maxSize)
		{
			maxIdx = i;
			maxSize = size;
		}
	}
	return { maxSize, maxIdx };
}

// Masked-out vertices have a value of 0 in the returned vector.
inline std::vector<v_size_t> degrees(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask)
{
	v_size_t vct = static_cast<v_size_t>(adj.size());
	std::vector<v_size_t> deg(vct, 0);
	for (v_size_t v = 0; v < vct; v++)
		if (mask[v])
			for (v_size_t w : adj[v])
				deg[v] += mask[w];
	return deg;
}

// Returns all vertices with a nonzero degree.
inline std::vector<v_size_t> non0degrees(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask)
{
	v_size_t vct = static_cast<v_size_t>(adj.size());
	std::vector<v_size_t> degs = degrees(adj, mask);
	std::vector<v_size_t> non0;
	non0.reserve(vct);
	for (v_size_t v = 0; v < vct; v++)
		if (degs[v])
			non0.push_back(v);
	return non0;
}

// Returns the connected components
inline std::vector<std::vector<v_size_t>> components(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask)
{
	v_size_t vct = static_cast<v_size_t>(adj.size());
	std::vector<mask_t> visited(vct, false);
	std::vector<std::vector<v_size_t>> comps;

	const auto visitStart = [&](v_size_t u, v_size_t v) -> void
	{
		visited[v] = true;
		if (u == v)
			comps.emplace_back();
		comps.back().push_back(v);
	};

	const auto visitEnd = [](v_size_t u, v_size_t v) -> void {};

	const auto isVisited = [&](v_size_t v) -> bool
	{
		return visited[v];
	};

	const auto visitAgain = [](v_size_t u, v_size_t v, v_size_t w) -> void {};

	DFS dfs(visitStart, visitEnd, isVisited, visitAgain);
	dfs(adj, mask);

	v_size_t compct = comps.size();

	for (std::vector<v_size_t>& comp : comps)
		std::sort(comp.begin(), comp.end());

	// Sorts by ascending size and then by ascending contents lexicographically
	std::vector<v_size_t> compMap(compct);
	std::iota(compMap.begin(), compMap.end(), 0);
	std::sort(compMap.begin(), compMap.end(),
		[&comps](v_size_t l, v_size_t r) -> bool
		{
			std::vector<v_size_t>& lComp = comps[l];
			std::vector<v_size_t>& rComp = comps[r];
			return lComp.size() < rComp.size() || lComp.size() == rComp.size() && lComp < rComp;
		});

	std::vector<std::vector<v_size_t>> mappedComps(compct);
	for (v_size_t i = 0; i < compct; i++)
		mappedComps[i] = std::move(comps[compMap[i]]);

	return mappedComps;
}

// Returns the biconnected components of the masked graph.
inline std::vector<std::vector<v_size_t>> bicomponents(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask)
{
	v_size_t vct = adj.size();
	v_size_t numI = 0;
	std::vector<v_size_t> num(vct, 0);
	std::vector<v_size_t> low(vct, 0);
	std::vector<std::vector<v_size_t>> bicomps;
	std::vector<std::pair<v_size_t, v_size_t>> edgeStack;
	std::vector<mask_t> inBicomp(vct, false);
	bool isSingleton;

	const auto visitStart = [&](v_size_t u, v_size_t v) -> void
	{
		low[v] = num[v] = ++numI;
		if (u != v)
			edgeStack.emplace_back(u, v);
		isSingleton = u == v;
	};

	const auto visitEnd = [&](v_size_t u, v_size_t v) -> void
	{
		if (isSingleton)
			bicomps.push_back(std::vector<v_size_t>{v});
		if (u == v)
			return;

		low[u] = std::min(low[u], low[v]);
		// If v is an articulation point, add a new biconnected component.
		if (low[v] < num[u])
			return;

		std::vector<v_size_t> curBicomp;
		std::pair<v_size_t, v_size_t> edge;

		// For each edge on the stack until u-v, add it to the current bicomponent if not already added
		// edge.first will be added as edge.second lower on the stack.
		while (edge = edgeStack.back(), edgeStack.pop_back(), num[edge.first] >= num[v])
			if (!inBicomp[edge.second])
			{
				curBicomp.push_back(edge.second);
				inBicomp[edge.second] = true;
			}

		assert(edge == std::make_pair(u, v));

		if (!inBicomp[edge.second])
			curBicomp.push_back(edge.second);

		if (!inBicomp[edge.first])
			curBicomp.push_back(edge.first);

		// Reset inBicomp for the next bicomponent.
		for (v_size_t x : curBicomp)
			inBicomp[x] = false;

		// Add the current biconnected component to the list.
		bicomps.emplace_back(std::move(curBicomp));
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
	dfs(adj, mask);

	v_size_t bicompct = bicomps.size();

	for (std::vector<v_size_t>& bicomp : bicomps)
		std::sort(bicomp.begin(), bicomp.end());

	// Sorts by ascending size and then by ascending contents lexicographically
	std::vector<v_size_t> compMap(bicompct);
	std::iota(compMap.begin(), compMap.end(), 0);
	std::sort(compMap.begin(), compMap.end(),
		[&](v_size_t l, v_size_t r) -> bool
		{
			std::vector<v_size_t>& lBicomp = bicomps[l];
			std::vector<v_size_t>& rBicomp = bicomps[r];
			return lBicomp.size() < rBicomp.size() || lBicomp.size() == rBicomp.size() && lBicomp < rBicomp;
		});

	std::vector<std::vector<v_size_t>> mappedBicomps(bicompct);
	for (v_size_t i = 0; i < bicompct; i++)
		mappedBicomps[i] = std::move(bicomps[compMap[i]]);

	return mappedBicomps;
}

// See https://doi.ieeecomputersociety.org/10.1109/SocialCom-PASSAT.2012.66
namespace betweenness_centrality
{
	

	inline std::vector<weight_t> internal(const CompressedData& data)
	{
		std::vector<weight_t> bc(data.vct);

		for (v_size_t v = 0; v < data.srcct; v++)
		{
			// Number of vertex pairs in v
			e_size_t num = static_cast<e_size_t>(data.mult[v]) * (data.mult[v] - 1);
			if (num == 0) continue;
			// Number of shortest paths for each pair
			v_size_t den = 0;
			weight_t minWt = INF<weight_t>;
			for (v_size_t w : data.adj[v])
			{
				weight_t wt = data.weights[w];
				if (wt < minWt)
				{
					minWt = wt;
					den = data.mult[w];
				}
				else if (wt == minWt)
					den += data.mult[w];
			}
			// Sum over all pairs of vertices in v of the dependency on any one vertex in w.
			weight_t dep = num / static_cast<weight_t>(den);
			for (v_size_t w : data.adj[v])
				bc[w] += dep;
		}

		return bc;
	}

	inline std::vector<weight_t> external(const CompressedData& data)
	{



		//// Compute the block-cut tree
		//std::vector<std::vector<v_size_t>> bicomps = bicomponents(data.adj, std::vector<mask_t>(data.vct, true));
		//v_size_t maxs = 0;
		//for (const std::vector<v_size_t>& bicomp : bicomps)
		//	maxs = std::max(maxs, static_cast<v_size_t>(bicomp.size()));

		//std::vector<v_size_t> artPts;
		//artPts.reserve(bicomps.size() - 1);
		//std::vector<std::vector<v_size_t>> bicompMap(data.vct);
		//for (v_size_t i = 0; i < bicomps.size(); i++)
		//	for (v_size_t v : bicomps[i])
		//	{
		//		if (bicompMap[v].size() == 1)
		//			artPts.push_back(v);
		//		bicompMap[v].push_back(i);
		//	}

		//// TODO temporary
		return std::vector<weight_t>();
	}

	inline std::vector<weight_t> compute(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask, const std::vector<mask_t>& src, const std::vector<weight_t>& weights)
	{
		CompressedData data = compress(adj, mask, src, weights);

		std::vector<weight_t> in = internal(data);
		std::vector<weight_t> ex = external(data);

		//for (v_size_t v = 0; v < data.vct; v++)
		//	in[v] += ex[v];
		return in;
	}
}

struct GraphData
{
	v_size_t stct;
	v_size_t crsct;
	v_size_t vct;
	std::vector<std::vector<v_size_t>> adj;
	std::vector<cost_t> costs;
	std::vector<weight_t> weights;
	std::vector<mask_t> stMask;
	std::string name;

	inline GraphData(v_size_t stct, v_size_t crsct, std::string&& name) :
		stct(stct),
		crsct(crsct),
		vct(stct + crsct),
		adj(vct),
		costs(vct),
		weights(vct),
		stMask(vct, false),
		name(name)
	{
		std::fill_n(stMask.begin(), stct, true);
	}

	inline GraphData(v_size_t stct, v_size_t crsct, const std::string& name) :
		GraphData(stct, crsct, std::move(std::string(name)))
	{}
};

//template<typename T = double>
struct Percent
{
	double val;

	explicit inline Percent(double prop) :
		val(prop * 100)
	{}

	inline Percent(double num, double den) :
		val(num * 100 / den)
	{}

	friend inline std::ostream& operator<<(std::ostream& strm, const Percent& pc)
	{
		return strm << pc.val << '%';
	}
};

inline void analyze(cost_t totalCost, const GraphData& data, const std::vector<mask_t>& mask)
{
	std::vector<v_size_t> non0degs = non0degrees(data.adj, mask);
	assert(!non0degs.empty());
	v_size_t non0ct = maskedCount(non0degs, data.stMask);
	Percent inclPc(non0ct, data.stct);

	std::vector<std::vector<v_size_t>> comps = components(data.adj, mask);
	assert(!comps.empty());
	v_size_t maxCompSize = maxMaskedCount(comps, data.stMask).first;
	Percent maxCompPc(maxCompSize, data.stct);

	std::vector<std::vector<v_size_t>> bicomps = bicomponents(data.adj, mask);
	assert(!bicomps.empty());
	v_size_t maxBicompSize = maxMaskedCount(bicomps, data.stMask).first;
	Percent maxBicompPc(maxBicompSize, data.stct);
	v_size_t maxBicompIncCrs = 0;
	for (const std::vector<v_size_t>& bicomp : bicomps)
		maxBicompIncCrs = std::max(maxBicompIncCrs, static_cast<v_size_t>(bicomp.size()));

	betweenness_centrality::compute(data.adj, mask, std::vector<mask_t>(data.vct, true), data.weights);

	std::cout
		<< std::setw(6) << totalCost
		<< std::setw(6) << data.stct
		<< std::setw(6) << non0ct
		<< std::setw(6) << maxCompSize
		<< std::setw(6) << maxBicompSize
		<< std::setw(6) << std::accumulate(mask.cbegin(), mask.cend(), 0)
		<< std::endl;


	std::cout
		<< std::setw(6) << totalCost
		<< ' ' << std::fixed << std::setprecision(3) << std::setw(7) << inclPc
		<< ' ' << std::fixed << std::setprecision(3) << std::setw(7) << maxCompPc
		<< ' ' << std::fixed << std::setprecision(3) << std::setw(7) << maxBicompPc
		<< std::endl;

}

inline void testWCStrategy(const GraphData& data, const std::vector<cost_t> maxCosts)
{
	// Courses in descending order of student count.
	std::vector<v_size_t> courses(data.crsct);
	std::iota(courses.begin(), courses.end(), data.stct);
	std::sort(courses.begin(), courses.end(),
		[&](v_size_t l, v_size_t r) -> bool
		{
			return data.adj[l].size() > data.adj[r].size();
		});

	for (cost_t maxCost : maxCosts)
	{
		std::vector<mask_t> mask(data.vct, true);
		cost_t totalCost = 0;
		for (v_size_t v : courses)
		{
			cost_t newTC = totalCost + data.costs[v];
			if (newTC <= maxCost)
			{
				mask[v] = false;
				totalCost = newTC;
			}
		}
		analyze(totalCost, data, mask);
	}
}

inline GraphData readWC(std::istream& strm)
{
	std::string name;
	std::getline(strm, name);

	typedef std::unordered_map<input_t, v_size_t> IdMap;
	IdMap stMap, crsMap;

	input_t stId, crsId;

	std::vector<std::pair<v_size_t, v_size_t>> edges;
	while (strm >> stId >> crsId)
	{
		IdMap::iterator stIter = stMap.try_emplace(stId, static_cast<v_size_t>(stMap.size())).first;
		IdMap::iterator crsIter = crsMap.try_emplace(crsId, static_cast<v_size_t>(crsMap.size())).first;
		edges.emplace_back(stIter->second, crsIter->second);
	}

	GraphData data(static_cast<v_size_t>(stMap.size()), static_cast<v_size_t>(crsMap.size()), std::move(name));

	for (const std::pair<v_size_t, v_size_t>& edge : edges)
	{
		v_size_t v = edge.first, w = edge.second + data.stct;
		data.adj[v].push_back(w);
		data.adj[w].push_back(v);
	}

	// Needed for identifying structurally equivalent vertices
	for (std::vector<v_size_t>& vAdj : data.adj)
		std::sort(vAdj.begin(), vAdj.end());

	std::fill_n(data.costs.begin(), data.stct, MAX<cost_t>);
	for (v_size_t v = data.stct; v < data.vct; v++)
		data.costs[v] = static_cast<cost_t>(data.adj[v].size());

	std::fill_n(data.weights.begin(), data.stct, static_cast<weight_t>(0));
	std::fill_n(data.weights.begin() + data.stct, data.crsct, static_cast<weight_t>(1));

	return data;
}

inline GraphData fileData(const char* filename)
{
	std::ifstream strm(filename);
	std::string fmt;
	std::getline(strm, fmt);
	if (fmt == "WC")
		return readWC(strm);
	else
	{
		std::cerr << "Unknown File Format \"" << fmt << '\"' << std::endl;
		exit(-1);
	}
}

inline void test(const GraphData& data, const std::vector<cost_t>& maxCosts)
{
	std::cout << std::string(32, '-') << std::endl << data.name << std::endl;
	std::cout << std::endl << "Weeden/Cornwell Strategy" << std::endl;
	testWCStrategy(data, maxCosts);
}

inline std::vector<cost_t> sizeThresholdMaxCosts(const GraphData& data, const std::vector<v_size_t>& sizes)
{
	v_size_t numThresholds = static_cast<v_size_t>(sizes.size());
	std::vector<cost_t> maxCosts(numThresholds, 0);
	for (v_size_t i = 0; i < numThresholds; i++)
		for (v_size_t v = data.stct; v < data.vct; v++)
			if (data.adj[v].size() >= sizes[i])
				maxCosts[i] += data.costs[v];
	return maxCosts;
}

int main()
{
	const std::vector<std::vector<v_size_t>> adj = {
		{  },
		{  2 },
		{  1,  3,  4,  5,  6 },
		{  2,  4 },
		{  2,  3 },
		{  2,  6,  7 },
		{  2,  5,  7 },
		{  5,  6,  8, 11 },
		{  7,  9, 11, 12, 14, 15 },
		{  8, 10, 11 },
		{  9, 11, 16, 17, 18 },
		{  7,  8,  9, 10 },
		{  8, 13 },
		{ 12, 14, 15 },
		{  8, 13 },
		{  8, 13 },
		{ 10 },
		{ 10, 18 },
		{ 10, 17 }
	};
	std::vector<weight_t> weights(adj.size(), 1);
	std::vector<mask_t> mask(adj.size(), true);

	std::vector<std::vector<v_size_t>> com = components(adj, mask);
	std::vector<std::vector<v_size_t>> bcom = bicomponents(adj, mask);

	std::vector<v_size_t> map(adj.size());
	std::iota(map.begin(), map.end(), 0);
	CompressedData cdata(
		adj.size(),
		adj.size(),
		std::vector<std::vector<v_size_t>>(adj),
		std::vector<weight_t>(adj.size(), 1),
		std::vector<v_size_t>(adj.size(), 1),
		std::vector<v_size_t>(map)
	);
	BlockCutForest bcf(cdata);

	std::cout << "";
	return 0;
}

int mainn(int argc, const char* argv[])
{
	const std::vector<v_size_t> sizes{ MAX<v_size_t>, 100, 75, 50, 40, 30 };
	for (int i = 1; i < argc; i++)
	{
		GraphData data = fileData(argv[i]);
		std::vector<cost_t> maxCosts = sizeThresholdMaxCosts(data, sizes);
		test(data, maxCosts);
	}
	return 0;
}
