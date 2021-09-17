/*
* A replication of some of the computations from Weeden and Cornwell's paper
* Written by Evan Bailey
*
* NOTE: This program does give the correct result for Cornell's University dataset. However, the code is fairly messy, it would be difficult to
*       adapt to a non-bipartite graph, and some performance improvements are likely still possible. I am currently working on another version to
*       address these issues.
*
* Input Format (file name passed as first parameter in stdin)
* Line 1: The string "WC" (without quotes)
* Line 2: Two space-separated integers: the number of students and the number of courses
* Onward: Five space-separated integers per line: the student id, the course id, the student's major (in [1, 6]),
*         a 1 for grads and a 0 for undergrads, and the course hours (always 1 for the Cornell dataset).
*
* Output format (excluding timing messages):
* Lines 1-2 are blank.
* Line 3: The string "UNIVERSITY"
* Then, several pairs of lines follow, each corresponding to a different threshold of total section cost removed. For each iteration:
* Line 1: The runtime of the betweenness centrality procedure (this can be disabled by setting the macro TIME_BC to false)
* Line 2: Four integers and one decimal: The total cost of removed sections, the number of included students, the largest component's size, the
*                                        largest biconnected component's size, and the maximum normalized betweenness centrality of any vertex.
* 
* Note: If the PERCENT macro is set to true, percentages will be output for included students, component size, and biconnected component size.
* 
* The total costs used were selected to correspond to the same groups of removed sections used in Weeden and Cornwell's paper.
* 
* On my machine, compiled by MSVC with /O2 enabled and NUM_THREADS set to 12, the first betweenness centrality iteration usually takes 12-15 seconds
* and the last iteration usually takes around 7-8 seconds.
*/

#include <algorithm>
#include <assert.h>
#include <atomic>
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
#include <unordered_set>
#include <vector>

#include "types.h"

#define USE_GPU false

#define DO_BC true
#define TIME_BC true
#define DO_BC_PROG true
#define NUM_THREADS 12
#define PERCENT true

//template<typename T>
//constexpr T MAX = std::numeric_limits<T>::max();

// Maps students to contracted vertices
std::vector<v_size_t> STMAP;
// Number of contracted student vertices
v_size_t STCT;
// Number of contracted course vertices
v_size_t CRSCT;
// Number of contracted vertices
v_size_t VCT;
// Number of uncontracted student vertices
v_size_t USTCT;
// Number of uncontracted vertices
v_size_t UVCT;

// Adjacency structure of contracted graph
std::vector<std::vector<v_size_t>> ADJ;
// Number of students/courses mapping to each contracted vertex
// TODO maybe decrease to 1 byte per vertex
std::vector<v_size_t> MULT;
// Number of students in each course.
std::vector<v_size_t> CRSDEGS;
// Student type (0 for courses) of one student of each contracted vertex
std::vector<v_type_t> TYPES;
// Vertex weight (0 for students) of each contracted vertex
std::vector<weight_t> WEIGHTS;
// Cost for each course
std::vector<cost_t> COSTS;
// Whether each contracted vertex has not been removed
// True for students with at least 1 remaining course, false for other students after a call to incl.
std::vector<mask_t> MASK;

// Connected components of the graph in order of nondescending numbers of students.
std::vector<std::vector<v_size_t>> CCS;
// Number of students in each connected component in nondescending order.
std::vector<v_size_t> CCSIZES;

// Biconnected components of the graph in order of nondescending numbers of students.
// Note that biconnectivity is defined in terms of the uncontracted graph.
// Vertices are pairs containing their contracted index and an identifier unique to other vertices that contract to the same vertex.
std::vector<std::vector<std::pair<v_size_t, v_size_t>>> BCCS;
// Number of students in each biconnected component in nondescending order.
std::vector<v_size_t> BCCSIZES;

// Betweenness centrality of each node accounting for multiplicities.
std::vector<weight_t> BC;

template<typename T>
inline std::vector<T> mkIota(T size, T val = 0)
{
	std::vector<T>iota(size);
	std::iota(iota.begin(), iota.end(), val);
	return iota;
}

template<typename T, typename Pred = std::less<T>>
inline std::vector<T> mkPerm(T size, Pred pred = Pred())
{
	std::vector<T> perm = mkIota(size);
	std::sort(perm.begin(), perm.end(), pred);
	return perm;
}

template<typename T, typename Pred = std::less<T>>
inline std::vector<T> mkPerm(T size, T val, Pred pred = Pred())
{
	std::vector<T> perm = mkIota(size, val);
	std::sort(perm.begin(), perm.end(), pred);
	return perm;
}

template<typename T, typename Pred = std::less<T>>
struct idx_cmp
{
private:
	const std::vector<T>& vec;
	const Pred pred;

public:
	inline idx_cmp(const std::vector<T>& vec, const Pred& pred = Pred()) :
		vec(vec),
		pred(pred)
	{}

	template<typename U>
	inline bool operator()(U l, U r) const
	{
		return pred(vec[l], vec[r]);
	}
};

template<typename T, typename Pred = std::less<T>>
struct pair_lex_cmp
{
private:
	const Pred pred;

public:
	inline pair_lex_cmp(const Pred& pred = Pred()) :
		pred(pred)
	{}

	inline bool operator()(const std::pair<T, T>& l, const std::pair<T, T>& r) const
	{
		return pred(l.first, r.first) || l.first == r.first && pred(l.second, r.second);
	}
};

template<typename T = size_t, typename Pred = std::less<T>>
struct size_cmp
{
private:
	const Pred pred;

public:
	inline size_cmp(const Pred& pred = Pred()) :
		pred(pred)
	{}

	template<typename U>
	inline bool operator()(const U& l, const U& r) const
	{
		return pred(l.size(), r.size());
	}
};

template<typename T, typename U>
struct std::hash<std::pair<T, U>>
{
	inline constexpr size_t operator()(const std::pair<T, U>& val) const noexcept
	{
		return std::hash<T>()(val.first) ^ std::hash<U>()(val.second);
	}
};

inline constexpr bool isSt(v_type_t vertType)
{
	return vertType != 0;
}

inline constexpr bool isCrs(v_type_t vertType)
{
	return vertType == 0;
}

inline constexpr v_type_t getMajor(v_type_t vertType)
{
	return vertType / 2;
}

inline constexpr bool isGrad(v_type_t vertType)
{
	return vertType & 1;
}

inline constexpr v_type_t getType(v_type_t major, bool grad)
{
	return major * 2 + grad;
}

void wcData(const char* location)
{
	std::ifstream fin(location);

	std::string dummy;
	if (!(fin >> dummy) || dummy != "WC")
	{
		std::cerr << "Cannot read input!";
		std::exit(-1);
	}

	fin >> USTCT >> CRSCT;
	UVCT = USTCT + CRSCT;

	COSTS = std::vector<cost_t>(CRSCT);
	CRSDEGS = std::vector<v_size_t>(CRSCT);

	std::vector<std::vector<v_size_t>> stadj(USTCT), crsadj(CRSCT);
	std::vector<v_type_t> sttp(USTCT);
	std::vector<weight_t> crswt(CRSCT, 1);

	// Read File Data
	v_size_t stid, crsid, major, grad;
	cost_t time;
	v_size_t line = 2;
	while (fin >> stid >> crsid >> major >> grad >> time)
	{
		stid--;
		crsid--;
		assert(stid < USTCT);
		assert(crsid < CRSCT);
		assert(major > 0 && major < 7);
		assert(grad < 2);
		sttp[stid] = getType(major, grad);
		stadj[stid].push_back(crsid);
		crsadj[crsid].push_back(stid);
		crswt[crsid] = static_cast<weight_t>(1) / time;
		COSTS[crsid] += time;
		CRSDEGS[crsid]++;
		line++;
	}
	fin.close();

	// Combine Structurally Equivalent Vertices
	for (v_size_t stid = 0; stid < USTCT; stid++)
		std::sort(stadj[stid].begin(), stadj[stid].end());
	for (v_size_t crsid = 0; crsid < CRSCT; crsid++)
		std::sort(crsadj[crsid].begin(), crsadj[crsid].end());

	static const auto stcmp = [&sttp, &stadj](v_size_t lstid, v_size_t rstid) -> bool
	{
		v_type_t ltp = sttp[lstid], rtp = sttp[rstid];
		if (ltp != rtp)
			return ltp < rtp;
		const std::vector<v_size_t> &ladj = stadj[lstid], &radj = stadj[rstid];
		if (ladj.size() != radj.size())
			return ladj.size() < radj.size();
		return std::lexicographical_compare(ladj.begin(), ladj.end(), radj.begin(), radj.end());
	};

	std::vector<v_size_t> stp = mkPerm(USTCT, stcmp);

	STMAP = std::vector<v_size_t>(USTCT, 0);
	STCT = 0;
	for (v_size_t stidp = 1; stidp < USTCT; stidp++)
	{
		v_size_t prev = stp[stidp - 1], cur = stp[stidp];
		bool equivalent = isGrad(sttp[prev]) == isGrad(sttp[cur]) && stadj[prev] == stadj[cur];
		STMAP[cur] = STCT += !equivalent;
	}
	VCT = ++STCT + CRSCT;

	ADJ = std::vector<std::vector<v_size_t>>(VCT);
	MULT = std::vector<v_size_t>(VCT, 0);
	TYPES = std::vector<v_type_t>(VCT, 0);
	WEIGHTS = std::vector<weight_t>(VCT, 0);
	MASK = std::vector<mask_t>(VCT, true);

	std::vector<v_size_t> crsdeg(CRSCT, 0);
	std::vector<mask_t> repeat(USTCT, 0);

	for (v_size_t stid = 0; stid < USTCT; stid++)
	{
		v_size_t v = STMAP[stid];
		assert(MULT[v] < std::numeric_limits<decltype(MULT)::value_type>::max());
		if (repeat[stid] = MULT[v]++ != 0) continue;
		std::vector<v_size_t>& adj = ADJ[v];
		const std::vector<v_size_t>& uadj = stadj[stid];
		v_size_t deg = static_cast<v_size_t>(uadj.size());
		adj = std::vector<v_size_t>(deg);
		for (v_size_t i = 0; i < deg; i++)
		{
			v_size_t crsid = uadj[i];
			adj[i] = crsid + STCT;
			crsdeg[crsid]++;
		}
		TYPES[v] = sttp[stid];
	}
	for (v_size_t crsid = 0; crsid < CRSCT; crsid++)
	{
		v_size_t v = crsid + STCT;
		MULT[v] = 1;
		std::vector<v_size_t>& adj = ADJ[v];
		const std::vector<v_size_t>& uadj = crsadj[crsid];
		v_size_t deg = crsdeg[crsid], udeg = static_cast<v_size_t>(uadj.size());
		adj = std::vector<v_size_t>();
		adj.reserve(deg);
		for (v_size_t i = 0; i < udeg; i++)
		{
			v_size_t stid = uadj[i];
			if (repeat[stid]) continue;
			adj.push_back(STMAP[stid]);
		}
		assert(adj.size() == deg);
		WEIGHTS[v] = crswt[crsid];
	}
}

// These are used for debugging

void uncontractedData()
{
	STMAP = { 0, 1, 2 };
	STCT = 3;
	CRSCT = 2;
	VCT = 5;
	USTCT = 3;
	UVCT = 5;
	ADJ.resize(5);
	ADJ[0] = { 3, 4 };
	ADJ[1] = { 3 };
	ADJ[2] = { 3, 4 };
	ADJ[3] = { 0, 1, 2 };
	ADJ[4] = { 0, 2 };
	MULT = { 1, 1, 1, 1, 1 };
	CRSDEGS = { 3, 2 };
	TYPES = { 2, 5, 2, 0, 0 };
	WEIGHTS = { 0, 0, 0, 1, 1 };
	COSTS = { 3, 2 };
	MASK = { true, true, true, true, true };
}

void unweightedData()
{
	STMAP = { 0, 1, 0 };
	STCT = 2;
	CRSCT = 2;
	VCT = 4;
	USTCT = 3;
	UVCT = 5;
	ADJ.resize(4);
	ADJ[0] = { 2, 3 };
	ADJ[1] = { 2 };
	ADJ[2] = { 0, 1 };
	ADJ[3] = { 0 };
	MULT = { 2, 1, 1, 1 };
	CRSDEGS = { 3, 2 };
	TYPES = { 2, 5, 0, 0 };
	WEIGHTS = { 0, 0, 1, 1 };
	COSTS = { 3, 2 };
	MASK = { true, true, true, true };
}

void weightedData()
{
	STMAP = { 0, 1, 0 };
	STCT = 2;
	CRSCT = 2;
	VCT = 4;
	USTCT = 3;
	UVCT = 5;
	ADJ.resize(4);
	ADJ[0] = { 2, 3 };
	ADJ[1] = { 2 };
	ADJ[2] = { 0, 1 };
	ADJ[3] = { 0 };
	MULT = { 2, 1, 1, 1 };
	CRSDEGS = { 3, 2 };
	TYPES = { 2, 5, 0, 0 };
	WEIGHTS = { 0, 0, 1, 0.5 };
	COSTS = { 3, 4 };
	MASK = { true, true, true, true };
}

const std::vector<cost_t> WC_MAXTC{ 0, 36081, 44442, 60766, 68474, 78458 };

v_size_t incl()
{
	v_size_t ret = 0;
	for (v_size_t v = 0; v < STCT; v++)
	{
		if (!MASK[v]) continue;
		MASK[v] = false;
		for (v_size_t w : ADJ[v])
			if (MASK[w])
			{
				MASK[v] = true;
				ret += MULT[v];
				break;
			}
	}
	return ret;
}

v_size_t ccs()
{
	std::vector<v_size_t> ccidx(VCT, MAX<v_size_t>);
	std::vector<v_size_t> ccsiz;
	std::vector<v_size_t> stck;
	for (v_size_t i = 0; i < VCT; i++)
	{
		if (!MASK[i] || ccidx[i] != MAX<v_size_t>) continue;
		v_size_t idx = static_cast<v_size_t>(ccsiz.size());
		stck.push_back(i);
		ccsiz.push_back(0);
		while (!stck.empty())
		{
			v_size_t v = stck.back();
			stck.pop_back();
			if (ccidx[v] != MAX<v_size_t>) continue;
			ccidx[v] = idx;
			if (v < STCT) ccsiz[idx] += MULT[v];
			for (v_size_t w : ADJ[v])
				if (MASK[w]) stck.push_back(w);
		}
	}
	v_size_t num = static_cast<v_size_t>(ccsiz.size());
	std::vector<v_size_t> perm = mkPerm(num, idx_cmp(ccsiz));
	std::vector<v_size_t> inv(num);
	for (v_size_t i = 0; i < num; i++)
		inv[perm[i]] = i;
	CCS = std::vector<std::vector<v_size_t>>(num);
	for (v_size_t i = 0; i < VCT; i++)
		if (MASK[i])
			CCS[inv[ccidx[i]]].push_back(i);
	CCSIZES = std::vector<v_size_t>(num);
	for (v_size_t idx = 0; idx < num; idx++)
	{
		std::sort(CCS[idx].begin(), CCS[idx].end());
		CCSIZES[idx] = ccsiz[perm[idx]];
	}
	return CCSIZES.empty() ? 0 : CCSIZES.back();
}

v_size_t bccs()
{
	typedef std::pair<v_size_t, v_size_t> Vert;
	typedef std::pair<Vert, Vert> Edge;

	struct Helper
	{
		v_size_t numI;
		v_size_t bccIdx;
		std::vector<std::vector<v_size_t>> num, low;
		std::vector<Edge> stck;
		std::vector<std::vector<Vert>> bicomps;
		std::vector<v_size_t> bccsizes;

		inline Helper() :
			numI(0),
			bccIdx(0),
			num(VCT),
			low(VCT)
		{
			for (v_size_t i = 0; i < VCT; i++)
				if (MASK[i])
				{
					num[i].resize(MULT[i], 0);
					low[i].resize(MULT[i], 0);
				}
		}

		void operator()(Vert u, Vert v)
		{
			size_t uv = u.first, ui = u.second, vv = v.first, vi = v.second;
			assert(num[vv][vi] == 0);
			low[vv][vi] = num[vv][vi] = ++numI;
			for (v_size_t wv : ADJ[vv])
				if (MASK[wv])
					for (v_size_t wi = 0; wi < MULT[wv]; wi++)
					{
						Vert w(wv, wi);
						if (num[wv][wi] == 0)
						{
							stck.emplace_back(v, w);
							// If stack overflow errors occur, increase call stack size. Infinite recursion should not be possible.
							(*this)(v, w);
							low[vv][vi] = std::min(low[vv][vi], low[wv][wi]);
							if (low[wv][wi] >= num[vv][vi])
							{
								std::unordered_set<Vert> inBicomp;
								std::vector<Vert> verts;
								Vert u1, u2;
								v_size_t size = 0;
								while (std::tie(u1, u2) = stck.back(), stck.pop_back(), num[u1.first][u1.second] >= num[wv][wi])
									if (inBicomp.insert(u2).second)
									{
										size += u2.first < STCT;
										verts.push_back(u2);
									}
								assert(u1 == v && u2 == w);
								if (inBicomp.insert(v).second)
								{
									size += vv < STCT;
									verts.push_back(v);
								}
								if (inBicomp.insert(w).second)
								{
									size += wv < STCT;
									verts.push_back(w);
								}
								std::sort(verts.begin(), verts.end(), pair_lex_cmp<v_size_t>());
								bicomps.push_back(std::move(verts));
								bccsizes.push_back(size);
							}
						}
						else if (w != u && num[wv][wi] < num[vv][vi])
						{
							stck.emplace_back(v, w);
							low[vv][vi] = std::min(low[vv][vi], num[wv][wi]);
						}
					}
		}
	} helper;

	for (v_size_t i = 0; i < VCT; i++)
		if (MASK[i])
			for (v_size_t j = 0; j < MULT[i]; j++)
				if (helper.num[i][j] == 0)
					helper(Vert(0, 0), Vert(i, j));
	assert(helper.stck.empty());
	v_size_t num = static_cast<v_size_t>(helper.bicomps.size());
	BCCS = std::vector<std::vector<Vert>>(num);
	BCCSIZES = std::vector<v_size_t>(num);
	std::vector<v_size_t> perm = mkPerm(num, idx_cmp(helper.bicomps, size_cmp()));
	for (v_size_t i = 0; i < num; i++)
	{
		BCCS[i] = std::move(helper.bicomps[perm[i]]);
		BCCSIZES[i] = helper.bccsizes[perm[i]];
	}
	return BCCSIZES.empty() ? 0 : BCCSIZES.back();
}

#if USE_GPU

static_assert(false, "GPU not yet supported, #define USE_GPU to false.");

#else

struct BcHelper
{
	inline void operator()(std::atomic<v_size_t>* bcSPtr, std::mutex* bcMutexPtr, std::vector<mask_t> src, e_size_t maxQ) const
	{
		std::atomic<v_size_t>& bcS = *bcSPtr;
		std::mutex& bcMutex = *bcMutexPtr;

		e_size_t ect = 0;
		//std::vector<std::pair<v_size_t, v_size_t*>> CADJ(VCT);
		for (v_size_t v = 0; v < VCT; v++)
		{
			v_size_t deg = 0;
			for (v_size_t w : ADJ[v])
				deg += MASK[w];
			//CADJ[v].first = deg;
			ect += deg;
		}
		ect /= 2;
		std::vector<v_size_t> CADJSP(ect * 2);

		std::vector<v_size_t> PSP(ect * 2);
		std::vector<v_size_t*> P(VCT);
		{
			v_size_t* pPtr = PSP.data();
			//v_size_t* aPtr = CADJSP.data();
			for (v_size_t i = 0; i < VCT; i++)
				if (MASK[i])
				{
					//v_size_t deg = CADJ[i].first;
					P[i] = pPtr;
					//CADJ[i].second = aPtr;
					//for (v_size_t w : ADJ[i])
						//if (MASK[w]) *(aPtr++) = w;
					//pPtr += deg;
					for (v_size_t w : ADJ[i])
						pPtr += MASK[w];
				}
			//assert(pPtr == &PSP.back() + 1);
			//assert(aPtr == &CADJSP.back() + 1);
		}

		std::vector<v_size_t> S;
		S.reserve(ect);
		std::vector<weight_t> threadBC(VCT, 0);

		std::vector<v_size_t> Ps(VCT, 0);
		std::vector<weight_t> SP(VCT);
		std::vector<weight_t> D(VCT);
		std::vector<weight_t> DEP(VCT);
		typedef std::pair<weight_t, v_size_t> Entry;
		std::vector<Entry> Q(maxQ);
		e_size_t Qs = 0;
		v_size_t s;
		for (s = bcS++; s < VCT; s = bcS++)
		{
#if DO_BC_PROG
			if (s == 0 || (s - 1) * 100 / VCT < s * 100 / VCT)
			{
				bcMutex.lock();
				std::cout << '\r' << s * 100 / VCT << '%' << std::flush;
				bcMutex.unlock();
			}
#endif
			if (!src[s]) continue;
			assert(MULT[s]);
			assert(S.empty());
			std::fill(Ps.begin(), Ps.end(), 0);
			std::fill(SP.begin(), SP.end(), static_cast<weight_t>(0));
			SP[s] = static_cast<weight_t>(MULT[s]);
			std::fill(D.begin(), D.end(), static_cast<weight_t>(-1));
			D[s] = static_cast<weight_t>(0);
			std::fill(DEP.begin(), DEP.end(), static_cast<weight_t>(0));
			assert(Qs == 0);
			Q[Qs++] = { static_cast<weight_t>(0), s };
			while (Qs)
			{
				v_size_t v = Q[0].second;
				e_size_t p = 0, c, l, r;
				Entry x = Q[--Qs], ql, qr, qc;
				for (; (l = p * 2 + 1) < Qs; p = c)
				{
					ql = Q[l];
					r = l + 1;
					bool goR = r < Qs && (qr = Q[r]).first < ql.first;
					c = goR ? r : l;
					qc = goR ? qr : ql;
					if (qc.first >= x.first) break;
					Q[p] = qc;
				}
				Q[p] = x;
				S.push_back(v);
				weight_t dv = D[v] + WEIGHTS[v];
				//v_size_t adjS = static_cast<v_size_t>(ADJ[v].size());
				//const std::pair<v_size_t, const v_size_t*> adj = CADJ[v];
				const std::pair<v_size_t, const v_size_t*> adj = { ADJ[v].size(), &ADJ[v].front() };
				//if (!adj.first) continue;
				const v_size_t* wPtr = adj.second;
				const v_size_t* end = wPtr + adj.first;
				wPtr += 3;
				for (; wPtr < end; wPtr += 4)
				{
					v_size_t w = *(wPtr - 3);
					if (MASK[w])
					{
						weight_t dw = dv + WEIGHTS[w];
						weight_t& dwr = D[w];
						if (dwr < 0)
						{
							for (c = Qs++; c > 0 && dw < Q[p = (c - 1) / 2].first; c = p)
								Q[c] = Q[p];
							Q[c] = { dwr = dw, w };
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
						else if (dwr == dw)
						{
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
					}
					w = *(wPtr - 2);
					if (MASK[w])
					{
						weight_t dw = dv + WEIGHTS[w];
						weight_t& dwr = D[w];
						if (dwr < 0)
						{
							for (c = Qs++; c > 0 && dw < Q[p = (c - 1) / 2].first; c = p)
								Q[c] = Q[p];
							Q[c] = { dwr = dw, w };
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
						else if (dwr == dw)
						{
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
					}
					w = *(wPtr - 1);
					if (MASK[w])
					{
						weight_t dw = dv + WEIGHTS[w];
						weight_t& dwr = D[w];
						if (dwr < 0)
						{
							for (c = Qs++; c > 0 && dw < Q[p = (c - 1) / 2].first; c = p)
								Q[c] = Q[p];
							Q[c] = { dwr = dw, w };
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
						else if (dwr == dw)
						{
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
					}
					w = *wPtr;
					if (MASK[w])
					{
						weight_t dw = dv + WEIGHTS[w];
						weight_t& dwr = D[w];
						if (dwr < 0)
						{
							for (c = Qs++; c > 0 && dw < Q[p = (c - 1) / 2].first; c = p)
								Q[c] = Q[p];
							Q[c] = { dwr = dw, w };
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
						else if (dwr == dw)
						{
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
					}
				}
				v_size_t w;
				// Fallthrough is intentional.
				switch (adj.first % 4)
				{
				case 3:
					w = *(wPtr - 1);
					if (MASK[w])
					{
						weight_t dw = dv + WEIGHTS[w];
						weight_t& dwr = D[w];
						if (dwr < 0)
						{
							for (c = Qs++; c > 0 && dw < Q[p = (c - 1) / 2].first; c = p)
								Q[c] = Q[p];
							Q[c] = { dwr = dw, w };
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
						else if (dwr == dw)
						{
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
					}
				case 2:
					w = *(wPtr - 2);
					if (MASK[w])
					{
						weight_t dw = dv + WEIGHTS[w];
						weight_t& dwr = D[w];
						if (dwr < 0)
						{
							for (c = Qs++; c > 0 && dw < Q[p = (c - 1) / 2].first; c = p)
								Q[c] = Q[p];
							Q[c] = { dwr = dw, w };
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
						else if (dwr == dw)
						{
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
					}
				case 1:
					w = *(wPtr - 3);
					if (MASK[w])
					{
						weight_t dw = dv + WEIGHTS[w];
						weight_t& dwr = D[w];
						if (dwr < 0)
						{
							for (c = Qs++; c > 0 && dw < Q[p = (c - 1) / 2].first; c = p)
								Q[c] = Q[p];
							Q[c] = { dwr = dw, w };
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
						else if (dwr == dw)
						{
							SP[w] += SP[v] * MULT[w];
							P[w][Ps[w]++] = v;
						}
					}
				}
			}
			while (!S.empty())
			{
				v_size_t w = S.back();
				S.pop_back();
				weight_t f = (src[w] ? MULT[w] + DEP[w] : DEP[w]) * MULT[w] / SP[w];
				for (v_size_t i = 0, v; i < Ps[w]; i++)
					v = P[w][i], DEP[v] += SP[v] * f;
				if (w != s)
					threadBC[w] += DEP[w] * MULT[s];
			}
		}
#if DO_BC_PROG
		if (s == VCT)
		{
			bcMutex.lock();
			std::cout << '\r' << std::flush;
			bcMutex.unlock();
		}
#endif
		bcMutex.lock();
		for (v_size_t i = 0; i < VCT; i++)
			if (MASK[i])
				BC[i] += threadBC[i] / MULT[i];
		bcMutex.unlock();
	}
};

// norm == 0: No normalization
// norm == 1: (n-2)(n-1)/2 normalization
// norm == 2: n(n-1)/2 normalization
inline weight_t bc(const std::vector<mask_t>& srcIn, v_type_t norm)
{
	assert(norm < 3);
	BC = std::vector<weight_t>(VCT, 0);
	std::atomic<v_size_t> bcS = 0;
	std::mutex bcMutex;
	// TODO measure exactly
	//v_size_t maxD = 0;
	//for (const std::vector<v_size_t>& adj : ADJ)
	//	maxD = std::max(maxD, static_cast<v_size_t>(adj.size()));
	e_size_t maxQ = 50000;
	maxQ = maxQ / 2 + 1;
	std::vector<mask_t> SRC(VCT);
	for (v_size_t i = 0; i < VCT; i++)
		SRC[i] = srcIn[i] & MASK[i];

	// Add paths between structurally equivalent pairs of vertices
	for (v_size_t v = 0; v < VCT; v++)
		if (SRC[v] && MULT[v] > 1)
		{
			e_size_t n = MULT[v];
			n = n * (n - 1);
			v_size_t d = 0;
			weight_t minWt = std::numeric_limits<weight_t>::infinity();
			for (v_size_t w : ADJ[v])
			{
				if (!MASK[w]) continue;
				weight_t wt = WEIGHTS[w];
				if (wt < minWt)
				{
					d = MULT[w];
					minWt = wt;
				}
				else if (wt == minWt)
					d += MULT[w];
			}
			weight_t dbc = static_cast<weight_t>(n) / d;
			for (v_size_t w : ADJ[v])
				if (MASK[w] && WEIGHTS[w] == minWt) BC[w] += dbc;
		}

	std::vector<std::thread> threads;
	for (v_size_t th = 0; th < NUM_THREADS; th++)
		threads.emplace_back(BcHelper(), &bcS, &bcMutex, SRC, maxQ);
	for (std::thread& thread : threads)
		thread.join();

	v_size_t n = 0;
	for (v_size_t v = 0; v < VCT; v++)
		if (MASK[v]) n += MULT[v];

	weight_t maxBC = 0;
	weight_t f =
		norm == 0 ? static_cast<weight_t>(0.5) :
		norm == 1 ? 1 / (static_cast<weight_t>(n - 2) * (n - 1)) :
		1 / (static_cast<weight_t>(n) * (n - 1));
	for (v_size_t v = 0; v < VCT; v++)
		if (MASK[v])
			maxBC = std::max(maxBC, BC[v] *= f);
	return maxBC;
}
#endif

template<unsigned int P, unsigned int W, bool I>
struct Format {};
template<unsigned int P, unsigned int W, bool I>
inline std::ostream& operator<<(std::ostream& strm, const Format<P, W, I>&)
{
	if (!I) strm << ' ';
	return strm << std::fixed << std::setprecision(P) << std::setw(W);
}

void analyze(unsigned int tc)
{
	const static Format<6, 6, true> ifmt;
	const static Format<1, 5, false> fmt;
	const static Format<5, 7, false> bfmt;

	weight_t f = static_cast<weight_t>(100) / USTCT;
	v_size_t inclSt = incl();
	v_size_t maxCC = ccs();
	v_size_t maxBCC = bccs();
#if DO_BC
#if TIME_BC
	typedef std::chrono::high_resolution_clock Clock;
	typedef Clock::time_point Time;
	Time start = Clock::now();
#endif
	std::vector<mask_t> src(VCT, false);
	for (v_size_t v : CCS.back())
		src[v] = true;
	weight_t maxBC = bc(src, 1);
#if TIME_BC
	Time end = Clock::now();
	std::cout
		<< std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << "s "
		<< std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() % 1000 << "ms" << std::endl;
#endif
#endif
	weight_t inclStd = inclSt * f;
	weight_t maxCCd = maxCC * f;
	weight_t maxBCCd = maxBCC * f;

	std::cout
		<< ifmt << tc
#if PERCENT
		<< fmt << inclStd << '%'
		<< fmt << maxCCd << '%'
		<< fmt << maxBCCd << '%'
#else
		<< fmt << inclSt
		<< fmt << maxCC
		<< fmt << maxBCC
#endif
#if DO_BC
		<< bfmt << maxBC
#endif
		<< std::endl;
}

void testWC(const std::string& name)
{
	std::cout << std::endl << std::endl << name << std::endl;
	std::vector<v_size_t> perm = mkPerm(CRSCT, idx_cmp(CRSDEGS, std::greater<size_t>()));
	v_size_t idx = 0;
	cost_t tc = 0;
	for (cost_t maxTC : WC_MAXTC)
	{
		while (idx < CRSCT)
		{
			v_size_t crsid = perm[idx];
			v_size_t v = crsid + STCT;
			if (!MASK[v]) continue;
			cost_t newTC = tc + COSTS[crsid];
			if (newTC > maxTC) break;
			MASK[v] = false;
			tc = newTC;
			idx++;
		}
		analyze(tc);
	}
}

int main(int argc, const char* argv[])
{
	if (argc != 2)
	{
		std::cerr << "Pass the input file as the first and only argument." << std::endl;
		std::exit(-1);
	}
	wcData(argv[1]);
	testWC("UNIVERSITY");
	std::cout << std::endl;
	return 0;
}
