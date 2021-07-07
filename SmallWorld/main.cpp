#include <algorithm>
#include <assert.h>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <random>
#include <stdint.h>
#include <string>
#include <unordered_set>
#include <vector>

#include "types.h"

template<typename T>
constexpr T NIL = std::numeric_limits<T>::max();

// Maps students to collapsed vertices
std::vector<v_size_t> STMAP;
// Number of collapsed student vertices
v_size_t STCT;
// Number of collapsed course vertices
v_size_t CRSCT;
// Number of collapsed vertices
v_size_t VCT;
// Number of uncollapsed student vertices
v_size_t USTCT;
// Number of uncollapsed vertices
v_size_t UVCT;

// Adjacency structure of collapsed graph
std::vector<std::vector<v_size_t>> ADJ;
// Number of students/courses mapping to each collapsed vertex
// TODO maybe decrease to 1 byte per vertex
std::vector<v_size_t> MULT;
// Number of students in each course.
std::vector<v_size_t> CRSDEGS;
// Student type (0 for courses) of each collapsed vertex
std::vector<v_type_t> TYPES;
// Vertex weight (0 for students) of each collapsed vertex
std::vector<weight_t> WEIGHTS;
// Cost for each course
std::vector<cost_t> COSTS;
// Whether each collapsed vertex has not been removed
// True for students with at least 1 remaining course, false for other students after a call to incl.
std::vector<bool> MASK;

// Connected components of the graph in order of nondescending numbers of students.
std::vector<std::vector<v_size_t>> CCS;
// Number of students in each connected component in nondescending order.
std::vector<v_size_t> CCSIZES;

// Biconnected components of the graph in order of nondescending numbers of students.
// Note that biconnectivity is defined in terms of the uncollapsed graph.
// Vertices are pairs containing their collapsed index and an identifier unique to other vertices that collapse the same vertex.
std::vector<std::vector<std::pair<v_size_t, v_size_t>>> BCCS;
// Number of students in each biconnected component in nondescending order.
std::vector<v_size_t> BCCSIZES;

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
		v_size_t p = stp[stidp - 1], c = stp[stidp];
		STMAP[c] = STCT += sttp[p] != sttp[c] || stadj[p] != stadj[c];
	}
	VCT = STCT + CRSCT;

	ADJ = std::vector<std::vector<v_size_t>>(VCT);
	MULT = std::vector<v_size_t>(VCT, 0);
	TYPES = std::vector<v_type_t>(VCT, 0);
	WEIGHTS = std::vector<weight_t>(VCT, 0);
	MASK = std::vector<bool>(VCT, true);

	std::vector<v_size_t> crsdeg(CRSCT, 0);
	std::vector<bool> repeat(USTCT, 0);

	for (v_size_t stid = 0; stid < USTCT; stid++)
	{
		v_size_t v = STMAP[stid];
		assert(MULT[v] < std::numeric_limits<decltype(MULT)::value_type>::max());
		if (repeat[stid] = MULT[v]++ != 0) continue;
		std::vector<v_size_t>& adj = ADJ[v];
		const std::vector<v_size_t>& uadj = stadj[stid];
		v_size_t deg = uadj.size();
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
		v_size_t deg = crsdeg[crsid], udeg = uadj.size();
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

void simpleData()
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

v_size_t cc()
{
	std::vector<v_size_t> ccidx(VCT, NIL<v_size_t>);
	std::vector<v_size_t> ccsiz;
	std::vector<v_size_t> stck;
	for (v_size_t i = 0; i < VCT; i++)
	{
		if (!MASK[i] || ccidx[i] != NIL<v_size_t>) continue;
		v_size_t idx = ccsiz.size();
		stck.push_back(i);
		ccsiz.push_back(0);
		while (!stck.empty())
		{
			v_size_t v = stck.back();
			stck.pop_back();
			if (ccidx[v] != NIL<v_size_t>) continue;
			ccidx[v] = idx;
			if (v < STCT) ccsiz[idx] += MULT[v];
			for (v_size_t w : ADJ[v])
				if (MASK[w]) stck.push_back(w);
		}
	}
	v_size_t num = ccsiz.size();
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

v_size_t bcc()
{
	typedef std::pair<v_size_t, v_size_t> Vert;
	typedef std::pair<Vert, Vert> Edge;

	struct Helper
	{
		v_size_t numI;
		v_size_t bccIdx;
		std::vector<std::vector<v_size_t>> num, low;
		std::vector<Edge> stck;
		std::vector<std::vector<Vert>> bccs;
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
								std::unordered_set<Vert> set;
								std::vector<Vert> verts;
								Vert u1, u2;
								v_size_t size = 0;
								while (std::tie(u1, u2) = stck.back(), stck.pop_back(), num[u1.first][u1.second] >= num[wv][wi])
									if (set.insert(u2).second)
									{
										size += u2.first < STCT;
										verts.push_back(u2);
									}
								assert(u1 == v && u2 == w);
								if (set.insert(v).second)
								{
									size += vv < STCT;
									verts.push_back(v);
								}
								if (set.insert(w).second)
								{
									size += wv < STCT;
									verts.push_back(w);
								}
								std::sort(verts.begin(), verts.end(), pair_lex_cmp<v_size_t>());
								bccs.push_back(std::move(verts));
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
	v_size_t num = helper.bccs.size();
	BCCS = std::vector<std::vector<Vert>>(num);
	BCCSIZES = std::vector<v_size_t>(num);
	std::vector<v_size_t> perm = mkPerm(num, idx_cmp(helper.bccs, size_cmp()));
	for (v_size_t i = 0; i < num; i++)
	{
		BCCS[i] = std::move(helper.bccs[perm[i]]);
		BCCSIZES[i] = helper.bccsizes[perm[i]];
	}
	return BCCSIZES.empty() ? 0 : BCCSIZES.back();
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
	using namespace std::chrono;
	typedef high_resolution_clock Clock;
	typedef Clock::time_point Time;
	const static Format<6, 6, true> ifmt;
	const static Format<1, 5, false> fmt;
	const static Format<5, 7, false> bfmt;

	weight_t f = static_cast<weight_t>(100) / USTCT;
	v_size_t inclSt = incl();
	v_size_t maxCC = cc();
	v_size_t maxBCC = bcc();
	Time start = Clock::now();
	//Ftype betc = betca<true>();
	Time end = Clock::now();
	//std::cout << duration_cast<seconds>(end - start).count() << " seconds" << std::endl;
	weight_t inclStd = inclSt * f;
	weight_t maxCCd = maxCC * f;
	weight_t maxBCCd = maxBCC * f;

	//std::cout << ifmt << tc << fmt << inclStd << fmt << maxCCd << fmt << maxBCCd << bfmt << betc << std::endl;
	std::cout << ifmt << tc << fmt << inclStd << fmt << maxCCd << fmt << maxBCCd << bfmt << std::endl;
}

void testWC()
{
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
	wcData(argv[1]);
	//simpleData();
	//randomData();

	//unsigned long long int ct = 0;
	//auto start = std::chrono::high_resolution_clock::now();
	//for (unsigned int i = 0; i < STCT; i++)
	//	for (unsigned int j = i + 1; j < STCT; j++)
	//	{
	//		std::vector<unsigned int> &ia = STADJ[i], &ja = STADJ[j];
	//		if (ia == ja)
	//		{
	//			ct++;
	//			//i++;
	//			//j = i + 1;
	//		}
	//	}
	//auto end = std::chrono::high_resolution_clock::now();
	//std::cout << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() << std::endl;
	//std::cout << ct << "/" << STCT;

	testWC();
	std::cout << std::endl;
}

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
