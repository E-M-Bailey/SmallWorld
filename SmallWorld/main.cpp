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

#include "types.h"

#define DO_BC true
#define TIME_BC true
#define DO_BC_PROG true
#define NUM_THREADS 12

template<typename T>
constexpr T MAX = std::numeric_limits<T>::max();
template<typename T>
constexpr T INF = std::numeric_limits<T>::infinity();

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

//struct DFSData
//{
//	const std::vector<std::vector<v_size_t>>& adj;
//	const std::vector<mask_t>& mask;
//};

// visitStart and visitEnd are passed u and v.
// isVisited is passed the vertex in question.
// visitPreRecursion, visitPostRecursion, and visitAgain are passed u, v, and w.
// u == v iff this iteration is the first in its connected component.
// All methods have void return type except isVisited, which returns a bool corresponding to whether a vertex is already visited.
template<typename VisitStart, typename VisitEnd, typename IsVisited, typename VisitPreRecursion, typename VisitPostRecursion, typename VisitAgain>
class DFS
{
public:
	const VisitStart& visitStart;
	const VisitEnd& visitEnd;
	const IsVisited& isVisited;
	const VisitPreRecursion& visitPreRecursion;
	const VisitPostRecursion& visitPostRecursion;
	const VisitAgain& visitAgain;


	inline DFS(const VisitStart& visitStart, const VisitEnd& visitEnd, const IsVisited& isVisited, const VisitPreRecursion& visitPreRecursion, const VisitPostRecursion& visitPostRecursion, const VisitAgain& visitAgain) :
		visitStart(visitStart),
		visitEnd(visitEnd),
		isVisited(isVisited),
		visitPreRecursion(visitPreRecursion),
		visitPostRecursion(visitPostRecursion),
		visitAgain(visitAgain)
	{}

private:
	inline void recurse(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask, v_size_t u, v_size_t v)
	{
		visitStart(u, v);
		for (v_size_t w : adj[v])
		{
			if (!mask[w]) continue;

			if (isVisited(w))
				visitAgain(u, v, w);
			else
			{
				visitPreRecursion(u, v, w);
				// Stack overflow is possible here if the stack size is too small.
				recurse(adj, mask, v, w);
				visitPostRecursion(u, v, w);
			}
		}
		visitEnd(u, v);
	}

public:
	inline void operator()(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask)
	{
		v_size_t vct = adj.size();
		for (v_size_t s = 0; s < vct; s++)
			if (mask[s] && !isVisited(s))
				recurse(adj, mask, s, s);
	}
};

// Returns the connected components
inline std::vector<std::vector<v_size_t>> components(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask)
{
	v_size_t vct = static_cast<v_size_t>(adj.size());
	std::vector<mask_t> visited(vct, false);
	std::vector<std::vector<v_size_t>> comps;

	const auto visitStart = [&visited, &comps](v_size_t u, v_size_t v) -> void
	{
		visited[v] = true;
		if (u == v)
			comps.emplace_back();
		comps.back().push_back(v);
	};

	const auto visitEnd = [](v_size_t u, v_size_t v) -> void {};

	const auto isVisited = [&visited](v_size_t v) -> bool
	{
		return visited[v];
	};

	const auto visitPreRecursion = [](v_size_t u, v_size_t v, v_size_t w) -> void {};

	const auto visitPostRecursion = [](v_size_t u, v_size_t v, v_size_t w) -> void {};

	const auto visitAgain = [](v_size_t u, v_size_t v, v_size_t w) -> void {};

	DFS dfs(visitStart, visitEnd, isVisited, visitPreRecursion, visitPostRecursion, visitAgain);
	dfs(adj, mask);

	//v_size_t vct = static_cast<v_size_t>(adj.size());
	//// Whether each vertex has been visited.
	//std::vector<mask_t> visited(vct, false);
	//// List of components.
	//std::vector<std::vector<v_size_t>> comps;
	//v_size_t curIdx = 0;
	//std::vector<v_size_t> dfsStack;
	//
	//for (v_size_t s = 0; s < vct; s++)
	//{
	//	// Skip visited vertices.
	//	if (!mask[s] || visited[s]) continue;
	//	std::vector<v_size_t> curComp;
	//	dfsStack.push_back(s);
	//	while (!dfsStack.empty())
	//	{
	//		// Pop v from the DFS stack.
	//		v_size_t v = dfsStack.back();
	//		dfsStack.pop_back();
	//		// Skip visited vertices
	//		if (visited[v]) continue;
	//		visited[v] = true;
	//		// Add v to the current component.
	//		curComp.push_back(v);
	//		// Push neighbors onto the dfs stack.
	//		for (v_size_t w : adj[v])
	//			if (mask[w])
	//				dfsStack.push_back(w);
	//	}
	//	// Add the current component to the list.
	//	comps.push_back(std::move(curComp));
	//}
	return comps;
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

	const auto visitStart = [&numI, &num, &low](v_size_t u, v_size_t v) -> void
	{
		low[v] = num[v] = ++numI;
	};

	const auto visitEnd = [](v_size_t u, v_size_t v) -> void {};

	const auto isVisited = [&num](v_size_t w) -> bool
	{
		return num[w] != 0;
	};

	const auto visitPreRecursion = [&num, &edgeStack](v_size_t u, v_size_t v, v_size_t w) -> void
	{
		edgeStack.emplace_back(v, w);
	};

	const auto visitPostRecursion = [&num, &low, &bicomps, &edgeStack, &inBicomp](v_size_t u, v_size_t v, v_size_t w) -> void
	{
		low[v] = std::min(low[v], low[w]);
		// If v is an articulation point, add a new biconnected component.
		if (low[w] >= num[v])
		{
			std::vector<v_size_t> curBicomp;
			std::pair<v_size_t, v_size_t> edge;

			// For each edge on the stack until u-v, add it to the current bicomponent if not already added
			// edge.first will be added as edge.second lower on the stack.
			while (edge = edgeStack.back(), edgeStack.pop_back(), num[edge.first] >= num[w])
				if (!inBicomp[edge.second])
				{
					curBicomp.push_back(edge.second);
					inBicomp[edge.second] = true;
				}

			assert(edge == std::make_pair(v, w));

			if (!inBicomp[edge.second])
				curBicomp.push_back(edge.second);

			if (!inBicomp[edge.first])
				curBicomp.push_back(edge.first);

			// Reset inBicomp for the next bicomponent.
			for (v_size_t x : curBicomp)
				inBicomp[x] = false;

			// Add the current biconnected component to the list.
			bicomps.emplace_back(std::move(curBicomp));
		}
	};

	const auto visitAgain = [&num, &low, &edgeStack](v_size_t u, v_size_t v, v_size_t w) -> void
	{
		if (w == u || num[w] >= num[v]) return;
		edgeStack.emplace_back(v, w);
		low[v] = std::min(low[v], num[w]);
	};

	DFS dfs(visitStart, visitEnd, isVisited, visitPreRecursion, visitPostRecursion, visitAgain);
	dfs(adj, mask);

	//for (std::vector<v_size_t>& bicomp : bicomps) std::sort(bicomp.begin(), bicomp.end());
	//std::sort(bicomps.begin(), bicomps.end(), [](const std::vector<v_size_t>& l, const std::vector<v_size_t>& r) { return r.size() < l.size(); });

	return bicomps;

	//BicomponentHelper helper(adj, mask);
	//for (v_size_t s = 0; s < helper.vct; s++)
	//	helper(s);
	//return helper.bicomps;
}

// See https://doi.ieeecomputersociety.org/10.1109/SocialCom-PASSAT.2012.66
namespace betweenness_centrality
{
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
			[&madj, &mask, &src, &weights](v_size_t l, v_size_t r) -> bool
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
		// Compute the block-cut tree
		std::vector<std::vector<v_size_t>> bicomps = bicomponents(data.adj, std::vector<mask_t>(data.vct, true));
		v_size_t maxs = 0;
		for (const std::vector<v_size_t>& bicomp : bicomps)
			maxs = std::max(maxs, static_cast<v_size_t>(bicomp.size()));

		std::vector<v_size_t> artPts;
		artPts.reserve(bicomps.size() - 1);
		std::vector<std::vector<v_size_t>> bicompMap(data.vct);
		for (v_size_t i = 0; i < bicomps.size(); i++)
			for (v_size_t v : bicomps[i])
			{
				if (bicompMap[v].size() == 1)
					artPts.push_back(v);
				bicompMap[v].push_back(i);
			}

		// TODO temporary
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
		[&data](v_size_t l, v_size_t r) -> bool
		{
			return data.adj[l].size() > data.adj[r].size();
		}
	);

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

int main(int argc, const char* argv[])
{
	const std::vector<v_size_t> sizes{ MAX<v_size_t>, 100, 75, 50, 40, 30 };
	for (int i = 1; i < argc; i++)
	{
		GraphData data = fileData(argv[i]);
		std::vector<cost_t> maxCosts = sizeThresholdMaxCosts(data, sizes);
		test(data, maxCosts);
	}
}


//struct BicomponentHelper
//{
//	const v_size_t vct;
//	const std::vector<std::vector<v_size_t>>& adj;
//	const std::vector<mask_t>& mask;
//
//	v_size_t numI = 0;
//	// Index (starting at 1) of each vertex in the DFS forest.
//	std::vector<v_size_t> num;
//	// Low-point of each vertex in the DFS palm forest.
//	std::vector<v_size_t> low;
//	// List of bicomponents.
//	std::vector<std::vector<v_size_t>> bicomps;
//	std::vector<std::pair<v_size_t, v_size_t>> edgeStack;
//	std::vector<mask_t> inBicomp;
//
//	inline BicomponentHelper(const std::vector<std::vector<v_size_t>>& adj, const std::vector<mask_t>& mask) :
//		vct(static_cast<v_size_t>(adj.size())),
//		adj(adj),
//		mask(mask),
//		num(vct, 0),
//		low(vct, 0),
//		inBicomp(vct, false)
//	{}
//
//	inline void biconnect(v_size_t u, v_size_t v)
//	{
//		// Number vertices, starting at 1.
//		low[v] = num[v] = ++numI;
//		for (v_size_t w : adj[v])
//		{
//			if (!mask[w]) continue;
//			// Recurse on unvisited vertices.
//			if (num[w] == 0)
//			{
//				edgeStack.emplace_back(v, w);
//				// Stack overflow errors are possible here; simply increase stack size until this is no longer an issue.
//				biconnect(v, w);
//
//				low[v] = std::min(low[v], low[w]);
//				// If v is an articulation point, add a new biconnected component.
//				if (low[w] >= num[v])
//				{
//					std::vector<v_size_t> curBicomp;
//					std::pair<v_size_t, v_size_t> edge;
//
//					// For each edge on the stack until u-v, add it to the current bicomponent if not already added
//					// edge.first will be added as edge.second lower on the stack.
//					while (edge = edgeStack.back(), edgeStack.pop_back(), num[edge.first] >= num[w])
//						if (!inBicomp[edge.second])
//						{
//							curBicomp.push_back(edge.second);
//							inBicomp[edge.second] = true;
//						}
//
//					assert(edge == std::make_pair(v, w));
//
//					if (!inBicomp[edge.second])
//						curBicomp.push_back(edge.second);
//
//					if (!inBicomp[edge.first])
//						curBicomp.push_back(edge.first);
//
//					// Reset inBicomp for the next bicomponent.
//					for (v_size_t x : curBicomp)
//						inBicomp[x] = false;
//
//					// Add the current biconnected component to the list.
//					bicomps.emplace_back(std::move(curBicomp));
//				}
//			}
//			// Update v's low-point if adjacent to a lower vertex (besides its parent).
//			else if (w != u && num[w] < num[v])
//			{
//				edgeStack.emplace_back(v, w);
//				low[v] = std::min(low[v], num[w]);
//			}
//		}
//	}
//
//	inline void operator()(v_size_t s)
//	{
//		if (mask[s] && num[s] == 0)
//			biconnect(0, s);
//	}
//};
