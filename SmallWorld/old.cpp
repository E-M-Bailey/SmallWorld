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

#define DO_BC false
#define TIME_BC true
#define DO_BC_PROG true
#define NUM_THREADS 12

template<typename T>
constexpr T MAX = std::numeric_limits<T>::max();

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
	fin >> dummy;

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
		//for (v_size_t i = 0; i < VCT; i++)
		//	if (MASK[i])
		//		delete[] P[i];
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
	//perThread();

	v_size_t n = 0;
	for (v_size_t v = 0; v < VCT; v++)
		if (MASK[v]) n += MULT[v];

	weight_t maxBC = 0;
	weight_t f =
		norm == 0 ? static_cast<weight_t>(0.5) :
		norm == 1 ? 1 / (static_cast<weight_t>(n - 2) * (n - 1)) :
		1 / (static_cast<weight_t>(n) * (n - 1));
	//std::cout << "f=" << f << ",n=" << n << std::endl;
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
		<< fmt << inclSt
		//<< fmt << inclStd
		<< fmt << maxCC
		//<< fmt << maxCCd
		<< fmt << maxBCC
		//<< fmt << maxBCCd
#if DO_BC
		<< bfmt << maxBC
#endif
		<< std::endl;
	//std::cout << BCCS.size() << std::endl;
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

void setUndergradMask()
{
	for (v_size_t v = 0; v < VCT; v++)
	{
		v_type_t type = TYPES[v];
		MASK[v] = isCrs(type) || !isGrad(type);
	}
}

int main(int argc, const char* argv[])
{
	wcData("Cornell.txt");
	//uncontractedData();
	//unweightedData();
	//weightedData();
	//randomData();
	//DATA_SRC();

	testWC("UNIVERSITY");
	//setUndergradMask();
	//testWC("UNDERGRAD");
	std::cout << std::endl;
	return 0;
}

//template<bool N = false>
//Ftype betcaOld()
//{
//	unsigned int crsct = 0;
//	unsigned int ect = 0;
//	unsigned int* crsmap = new unsigned int[CRSCT];
//	for (unsigned int crsid = 0; crsid < CRSCT; crsid++)
//		if (!REMOVED[crsid])
//			crsmap[crsid] = crsct++;
//
//	unsigned int* stdeg = new unsigned int[STCT];
//	unsigned int** stadj = new unsigned int*[STCT];
//	for (unsigned int stid = 0; stid < STCT; stid++)
//	{
//		unsigned int& deg = stdeg[stid] = 0;
//		for (unsigned int crsid : STADJ[stid])
//			deg += !REMOVED[crsid];
//		ect += deg;
//		unsigned int*& adj = stadj[stid] = new unsigned int[deg];
//		unsigned int j = 0;
//		for (unsigned int crsid : STADJ[stid])
//			if (!REMOVED[crsid])
//				adj[j++] = crsmap[crsid];
//	}
//
//	unsigned int* crsdeg = new unsigned int[crsct];
//	const unsigned int** crsadj = new const unsigned int*[crsct];
//	Ftype* weights = new Ftype[crsct];
//	for (unsigned int crsid = 0; crsid < CRSCT; crsid++)
//		if (!REMOVED[crsid])
//		{
//			unsigned int mapped = crsmap[crsid];
//			crsdeg[mapped] = CRSADJ[crsid].size();
//			crsadj[mapped] = CRSADJ[crsid].data();
//			weights[mapped] = 0.5 / TIMES[crsid];
//		}
//	unsigned int n = STCT + crsct;
//	Ftype* betca = new Ftype[n];
//
//	compBetcA(STCT, crsct, ect, stdeg, crsdeg, stadj, crsadj, weights, betca);
//
//	Ftype factor = N ? 2. / ((n - 1) * (n - 2)) : 1;
//
//	Ftype ret = 0;
//	for (unsigned int stid = 0; stid < STCT; stid++)
//		ret = std::max(ret, BETCA[stid] = factor * betca[stid]);
//	for (unsigned int crsid = 0; crsid < CRSCT; crsid++)
//		if (!REMOVED[crsid])
//			ret = std::max(ret, BETCA[STCT + crsid] = factor * betca[STCT + crsmap[crsid]]);
//
//	delete[] crsmap;
//	delete[] stdeg;
//	for (unsigned int stid = 0; stid < STCT; stid++) delete[] stadj[stid];
//	delete[] stadj;
//	delete[] crsdeg;
//	delete[] crsadj;
//	delete[] weights;
//	delete[] betca;
//	return ret;
//}

// Barring memory limitations, up to 2^16-1 vertices, 2^32-1 edges*, and 2^7-1 majors are supported. See global.h to increase these limitations if necessary.
// *There cannot be more than 
//typedef float Ftype;
// cuda.cu functions
//void compBetcA(
//	unsigned int stct,
//	unsigned int crsct,
//	unsigned int ect,
//	const unsigned int* stdeg,
//	const unsigned int* crsdeg,
//	const unsigned int* const* stadj,
//	const unsigned int* const* crsadj,
//	const Ftype* times,
//	//double* betcaOut
//	Ftype* betcaOut
//);
//
//unsigned int STCT = 0;
//unsigned int CRSCT = 0;
//double STCTD = 0;
//double CRSCTD = 0;
//
// Edge weights are implicitly half the reciprocal of the TIMES value of their course.
//
// Indexed by stid
//std::vector<unsigned int> MAJORS;
//std::vector<unsigned int> GRADS;
//std::vector<std::vector<unsigned int>> STADJ;
//std::vector<std::vector<double>> STWT;
//
// Indexed by crsid
//std::vector<unsigned int> TIMES;
//std::vector<unsigned int> COSTS;
//std::vector<std::vector<unsigned int>> CRSADJ;
//std::vector<unsigned int> REMOVED;
//std::vector<std::vector<double>> CRSWT;
//
//std::vector<Ftype> BETCA;
//std::vector<double> BETCB;


//template<typename T>
//class UnionFind
//{
//	T k;
//	mutable std::vector<T> P;
//	std::vector<T> R;
//
//public:
//	inline UnionFind(T n) :
//		k(n),
//		P(n),
//		R(n, 0)
//	{
//		std::iota(P.begin(), P.end(), 0);
//	}
//
//	inline T size() const
//	{
//		return P.size();
//	}
//
//	inline T rCount() const
//	{
//		return k.size();
//	}
//
//	// Path Halving
//	inline T find(T x) const
//	{
//		while (P[x] != x) x = P[x] = P[P[x]];
//		return x;
//	}
//
//	inline T merge(T x, T y)
//	{
//		if ((x = find(x)) != (y = find(y)))
//		{
//			k--;
//			if (R[x] < R[y]) std::swap(x, y);
//			P[y] = x;
//			R[x] += R[x] == R[y];
//		}
//		return x;
//	}
//
//	inline std::vector<T> getReps() const
//	{
//		std::vector<T> ret;
//		ret.reserve(k);
//		for (T i = 0; i < size(); i++)
//			if (P[i] == i) ret.push_back(i);
//		assert(ret.size() == k);
//		return ret;
//	}
//};

//std::vector<unsigned int> S;
//S.reserve(CRSCT);
//for (const auto& adj : CRSADJ) S.push_back(adj.size());
//std::sort(S.rbegin(), S.rend());
//unsigned int n100 = 0, n75 = 0, n50 = 0, n40 = 0, n30 = 0;
//unsigned int c100 = 0, c75 = 0, c50 = 0, c40 = 0, c30 = 0;
//unsigned int i = 0;
//for (; S[i] >= 100; i++)
//{
//	n100++;
//	c100 += S[i];
//}
//n75 = n100;
//c75 = c100;
//for (; S[i] >= 75; i++)
//{
//	n75++;
//	c75 += S[i];
//}
//n50 = n75;
//c50 = c75;
//for (; S[i] >= 50; i++)
//{
//	n50++;
//	c50 += S[i];
//}
//n40 = n50;
//c40 = c50;
//for (; S[i] >= 40; i++)
//{
//	n40++;
//	c40 += S[i];
//}
//n30 = n40;
//c30 = c40;
//for (; S[i] >= 30; i++)
//{
//	n30++;
//	c30 += S[i];
//}
//std::cout << n100 << ' ' << c100 << std::endl;
//std::cout << n75 << ' ' << c75 << std::endl;
//std::cout << n50 << ' ' << c50 << std::endl;
//std::cout << n40 << ' ' << c40 << std::endl;
//std::cout << n30 << ' ' << c30 << std::endl;

//void wcDataOld(const char* location)
//{
//	std::ifstream csv(location);
//	unsigned int stid, crsid, major, grad, time = 1;
//	while (csv >> stid >> crsid >> major >> grad)
//	{
//		stid--;
//		crsid--;
//		major--;
//		if (stid >= STCT)
//		{
//			STCT = stid + 1;
//			MAJORS.resize(STCT);
//			GRADS.resize(STCT);
//			STADJ.resize(STCT);
//			//STWT.resize(STCT);
//		}
//		MAJORS[stid] = major;
//		GRADS[stid] = grad;
//		STADJ[stid].push_back(crsid);
//		//STWT[stid].push_back(wt);
//		if (crsid >= CRSCT)
//		{
//			CRSCT = crsid + 1;
//			TIMES.resize(CRSCT);
//			COSTS.resize(CRSCT);
//			CRSADJ.resize(CRSCT);
//			//CRSWT.resize(CRSCT);
//		}
//		TIMES[crsid] = time;
//		COSTS[crsid] += time;
//		CRSADJ[crsid].push_back(stid);
//		//CRSWT[crsid].push_back(wt);
//	}
//	for (std::vector<unsigned int>& stadj : STADJ)
//		std::sort(stadj.begin(), stadj.end());
//	for (std::vector<unsigned int>& crsadj : CRSADJ)
//		std::sort(crsadj.begin(), crsadj.end());
//	STCTD = STCT;
//	CRSCTD = CRSCT;
//	REMOVED = std::vector<unsigned int>(CRSCT);
//	BETCA = std::vector<Ftype>(STCT + CRSCT);
//}

//unsigned int inclOld()
//{
//	unsigned int ret = 0;
//	for (unsigned int stid = 0; stid < STCT; stid++)
//		for (unsigned int crsid : STADJ[stid])
//			if (!REMOVED[crsid])
//			{
//				ret++;
//				break;
//			}
//	return ret;
//}

//unsigned int ccOld()
//{
//	unsigned int ret = 0;
//	unsigned int n = STCT + CRSCT;
//	// replace vis with vector<unsigned int> to label specific cc's.
//	std::vector<bool> vis(n, false);
//	std::vector<unsigned int> stck;
//	for (unsigned int i = 0; i < n; i++)
//	{
//		if (vis[i]) continue;
//		stck.push_back(i);
//		unsigned int ct = 0;
//		while (!stck.empty())
//		{
//			unsigned int v = stck.back();
//			stck.pop_back();
//			if (vis[v]) continue;
//			vis[v] = true;
//			if (v < STCT)
//			{
//				ct++;
//				for (unsigned int w : STADJ[v])
//					if (!REMOVED[w])
//						stck.push_back(w + STCT);
//			}
//			else
//				for (unsigned int w : CRSADJ[v - STCT])
//					stck.push_back(w);
//		}
//		if (ct > ret) ret = ct;
//	}
//	return ret;
//}

// I attempted to convert this into an iterative algorithm with a stack, but it was slower after compiler optimization.
//void bccHelperOld(unsigned int v, unsigned int u, unsigned int& ret, unsigned int& i, std::vector<unsigned int>& num, std::vector<unsigned int>& low, std::vector<std::pair<unsigned int, unsigned int>>& stck)
//{
//	bool vSt = v < STCT;
//	assert(num[v] == 0);
//	low[v] = num[v] = ++i;
//	const std::vector<unsigned int>& adj = vSt ? STADJ[v] : CRSADJ[v - STCT];
//	for (unsigned int w : adj)
//	{
//		if (vSt && REMOVED[w]) continue;
//		if (vSt) w += STCT;
//		if (num[w] == 0)
//		{
//			stck.emplace_back(v, w);
//			// Stack overflows here do not necessarily indicate a bug; this method recurses deeply but finitely.
//			// Use a large stack size in Visual Studio's project configuration (100MB is plenty)
//			bccHelper(w, v, ret, i, num, low, stck);
//
//			low[v] = std::min(low[v], low[w]);
//			if (low[w] >= num[v])
//			{
//				std::unordered_set<unsigned int> stSet;
//				unsigned int u1, u2;
//				while (std::tie(u1, u2) = stck.back(), stck.pop_back(), num[u1] >= num[w])
//					stSet.insert(u1 < STCT ? u1 : u2);
//				assert(u1 == v && u2 == w);
//				stSet.insert(vSt ? v : w);
//				ret = std::max(static_cast<unsigned int>(stSet.size()), ret);
//			}
//		}
//		else if (num[w] < num[v] && w != u)
//		{
//			stck.emplace_back(v, w);
//			low[v] = std::min(low[v], num[w]);
//		}
//	}
//}

//unsigned int bccOld()
//{
//	unsigned int i = 0;
//	unsigned int ret = 0;
//	unsigned int n = STCT + CRSCT;
//	std::vector<unsigned int> num(n, 0);
//	std::vector<unsigned int> low(n, 0);
//	std::vector<std::pair<unsigned int, unsigned int>> stck;
//	for (unsigned int j = 0; j < n; j++)
//	{
//		if (num[j] || j >= STCT && REMOVED[j - STCT]) continue;
//		bccHelper(j, 0, ret, i, num, low, stck);
//	}
//	assert(stck.empty());
//	return ret;
//}

//void testWCOld()
//{
//	std::fill(REMOVED.begin(), REMOVED.end(), false);
//	// Nonascending section size
//	const auto pred = [](unsigned int l, unsigned int r) -> bool
//	{
//		return CRSADJ[l].size() > CRSADJ[r].size();
//	};
//
//	std::vector<unsigned int> perm = mkIota(CRSCT);
//	std::sort(perm.begin(), perm.end(), pred);
//
//	unsigned int idx = 0;
//	unsigned int tc = 0;
//	for (unsigned int maxc : WC_MAXTC)
//	{
//		while (idx < CRSCT)
//		{
//			unsigned int crsid = perm[idx];
//			unsigned int cost = COSTS[crsid];
//			unsigned int newTC = tc + cost;
//			if (newTC > maxc) break;
//			REMOVED[crsid] = true;
//			tc = newTC;
//			idx++;
//		}
//		analyze(tc);
//	}
//}


// This is significantly less efficient than the recursive version. It is also outdated and would be incorrect in the current program.
//unsigned int iterativeBcc()
//{
//	typedef std::tuple<unsigned int, unsigned int, unsigned int, bool> RecData;
//	unsigned int i = 0;
//	unsigned int ret = 0;
//	unsigned int n = STCT + CRSCT;
//	std::vector<unsigned int> num(n, 0);
//	std::vector<unsigned int> low(n, 0);
//	std::vector<RecData> stck;
//	std::vector<Edge> eStck;
//	for (unsigned int j = 0; j < n; j++)
//	{
//		if (num[j]) continue;
//		stck.emplace_back(j, 0, 0, true);
//		while (!stck.empty())
//		{
//			auto& [v, u, idx, beg] = stck.back();
//			if (beg && idx == 0)
//			{
//				assert(num[v] == 0 && low[v] == 0);
//				low[v] = num[v] = ++i;
//			}
//			bool vSt = v < STCT;
//			std::vector<unsigned int>& adj = vSt ? STADJ[v] : CRSADJ[v - STCT];
//			if (idx == adj.size())
//			{
//				stck.pop_back();
//				continue;
//			}
//			assert(idx < adj.size());
//			unsigned int w = adj[idx];
//			if (vSt) w += STCT;
//			if (beg)
//			{
//				if (num[w] == 0)
//				{
//					eStck.emplace_back(v, w);
//					beg = false;
//					stck.emplace_back(w, v, 0, true);
//					continue;
//				}
//				else if (num[w] < num[v] && w != u)
//				{
//					eStck.emplace_back(v, w);
//					low[v] = std::min(low[v], num[w]);
//				}
//				idx++;
//			}
//			else
//			{
//				low[v] = std::min(low[v], low[w]);
//				if (low[w] >= num[v])
//				{
//					std::unordered_set<unsigned int> set;
//					unsigned int u1, u2;
//					while (std::tie(u1, u2) = eStck.back(), eStck.pop_back(), num[u1] >= num[w])
//						set.insert(u1 < STCT ? u1 : u2);
//					assert(u1 == v && u2 == w);
//					set.insert(vSt ? v : w);
//					ret = std::max(ret, static_cast<unsigned int>(set.size()));
//				}
//				idx++;
//				beg = true;
//			}
//		}
//	}
//	assert(eStck.empty());
//	return ret;
//}

//void simpleDataOld()
//{
//	STADJ.resize(STCTD = STCT = 3);
//	CRSADJ.resize(CRSCTD = CRSCT = 2);
//	COSTS.resize(CRSCT);
//	REMOVED.resize(CRSCT, false);
//	TIMES.resize(CRSCT, 1);
//	BETCA.resize(STCT + CRSCT);
//	STADJ[0] = { 0, 1 };
//	STADJ[1] = { 0 };
//	STADJ[2] = { 0, 1 };
//	COSTS[0] = (CRSADJ[0] = { 0, 1, 2 }).size();
//	COSTS[1] = (CRSADJ[1] = { 0, 2 }).size();
//}
//
//void randomDataOld()
//{
//	STADJ.resize(STCT = 1000000);
//	CRSADJ.resize(CRSCT = 500000);
//	STCTD = STCT;
//	CRSCTD = CRSCT;
//	std::default_random_engine randy;
//	std::uniform_int_distribution<unsigned int> stDist(0, STCT - 1);
//	std::uniform_int_distribution<unsigned int> crsDist(0, CRSCT - 1);
//	for (unsigned long long int i = 0; i < 1000000ull; i++)
//	{
//		unsigned int stid = stDist(randy);
//		unsigned int crsid = crsDist(randy);
//		STADJ[stid].push_back(crsid);
//		CRSADJ[crsid].push_back(stid);
//	}
//}
//inf 0 0
//100 174 36081
//75 272 44442
//50 541 60766
//40 715 68474
//30 1007 78458

	/*
	inline void slow(std::atomic<v_size_t>* bcSPtr, std::mutex* bcMutexPtr, std::vector<mask_t> src, e_size_t ect, e_size_t maxQ) const
	{
		std::atomic<v_size_t>& bcS = *bcSPtr;
		std::mutex& bcMutex = *bcMutexPtr;
		struct VData
		{
			bool src;
			bool vis;
			v_size_t mult;
			weight_t wt;
			v_size_t deg;
			v_size_t* adj;
			v_size_t* p;
			v_size_t ps;
			weight_t sp;
			weight_t d;
			weight_t dep;
			weight_t bc;

			inline VData(v_size_t v, bool SRC) :
				src(SRC),
				mult(MULT[v]),
				wt(WEIGHTS[v]),
				deg(0),
				bc(0)
			{
				if (MASK[v])
				{
					for (v_size_t w : ADJ[v])
						deg += MASK[w];
					adj = new v_size_t[deg];
					v_size_t adjIdx = 0;
					for (v_size_t w : ADJ[v])
						if (MASK[w]) adj[adjIdx++] = w;
					p = new v_size_t[deg];
				}
				else adj = nullptr;
			}

			inline constexpr void reset()
			{
				vis = false;
				ps = 0;
				sp = 0;
				d = -1;
				dep = 0;
			}

			inline constexpr void setStart()
			{
				vis = true;
				sp = mult;
				d = 0;
			}

			inline ~VData()
			{
				if (adj)
				{
					delete[] adj;
					delete[] p;
				}
			}
		};
		std::vector<VData> DATA;
		DATA.reserve(VCT);
		for (v_size_t v = 0; v < VCT; v++)
			DATA.emplace_back(v, src[v]);

		//std::vector<weight_t> threadBC(VCT, 0);
		//std::vector<v_size_t> S;
		std::vector<v_size_t> S;
		S.reserve(ect);
		//std::vector<v_size_t> P(VCT * maxD);
		//std::vector<v_size_t*> P(VCT);
		//for (v_size_t i = 0; i < VCT; i++)
		//	if (MASK[i])
		//		data[i].P = new v_size_t[ADJ[i].size()];
				//P[i] = new v_size_t[ADJ[i].size()];
		//std::vector<v_size_t> Ps(VCT, 0);
		//std::vector<weight_t> SP(VCT);
		//std::vector<weight_t> D(VCT);
		//std::vector<weight_t> DEP(VCT);
		typedef std::pair<weight_t, v_size_t> Entry;
		std::vector<Entry> Q(maxQ);
		e_size_t Qs = 0;
		v_size_t s;
		for (s = bcS++; s < VCT; s = bcS++)
		{
#if DO_BC_PROG
			if (s == 0 || (s - 1) * 100 / VCT < s * 100 / VCT) std::cout << '\r' << s * 100 / VCT << '%' << std::flush;
#endif
			VData& sd = DATA[s];

			if (!sd.src) continue;
			assert(sd.mult);
			assert(S.empty());

			for (VData& data : DATA)
				data.reset();
			sd.setStart();

			//std::fill(ps.begin(), ps.end(), 0);
			//std::fill(P.begin(), P.end(), std::vector<v_size_t>());
			//std::fill(SP.begin(), SP.end(), static_cast<weight_t>(0));
			//SP[s] = static_cast<weight_t>(MULT[s]);
			//std::fill(D.begin(), D.end(), static_cast<weight_t>(-1));
			//D[s] = static_cast<weight_t>(0);
			//std::fill(DEP.begin(), DEP.end(), static_cast<weight_t>(0));
			assert(Qs == 0);
			Q[Qs++] = { static_cast<weight_t>(0), s };
			while (Qs)
			{
				v_size_t v = Q[0].second;
				VData& vd = DATA[v];
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
				weight_t dv = vd.d + vd.wt;
				if (!vd.adj) continue;
				const v_size_t* wPtr = vd.adj;
				const v_size_t* end = wPtr + vd.deg;
				for (;
					wPtr < end;
					wPtr++)
				{
					v_size_t w = *wPtr;
					VData& wd = DATA[w];
					weight_t dw = dv + wd.wt;
					if (!wd.vis)
					//if (vd.d < 0)
					{
						wd.vis = true;
						for (c = Qs++; c > 0 && dw < Q[p = (c - 1) / 2].first; c = p)
							Q[c] = Q[p];
						Q[c] = { wd.d = dw, w };
						wd.sp += vd.sp * wd.mult;
						wd.p[wd.ps++] = v;
					}
					else if (wd.d == dw)
					{
						wd.sp += vd.sp * wd.mult;
						wd.p[wd.ps++] = v;
					}
				}
			}
			while (!S.empty())
			{
				v_size_t w = S.back();
				S.pop_back();
				VData& wd = DATA[w];
				weight_t f = (src[w] ? wd.mult + wd.dep : wd.dep) * wd.mult / wd.sp;
				for (v_size_t i = 0, v; i < wd.ps; i++)
				{
					VData& vd = DATA[wd.p[i]];
					vd.dep += vd.sp * f;
				}
				if (w != s)
					wd.bc += wd.dep * sd.mult;
			}
		}
#if DO_BC_PROG
		if (s == VCT)
			std::cout << '\r' << std::flush;
#endif
		bcMutex.lock();
		for (v_size_t v = 0; v < VCT; v++)
		{
			VData& vd = DATA[v];
			//if (vd.mask)
			if (MASK[v])
				BC[v] += vd.bc / vd.mult;
		}
		bcMutex.unlock();
	}
	*/
