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

typedef float Ftype;
// cuda.cu functions
void compBetcA(
	unsigned int stct,
	unsigned int crsct,
	unsigned int ect,
	const unsigned int* stdeg,
	const unsigned int* crsdeg,
	const unsigned int* const* stadj,
	const unsigned int* const* crsadj,
	const Ftype* times,
	//double* betcaOut
	Ftype* betcaOut
);

unsigned int STCT = 0;
unsigned int CRSCT = 0;
double STCTD = 0;
double CRSCTD = 0;

// Edge weights are implicitly half the reciprocal of the TIMES value of their course.

// Indexed by stid
std::vector<unsigned int> MAJORS;
std::vector<unsigned int> GRADS;
std::vector<std::vector<unsigned int>> STADJ;
//std::vector<std::vector<double>> STWT;

// Indexed by crsid
std::vector<unsigned int> TIMES;
std::vector<unsigned int> COSTS;
std::vector<std::vector<unsigned int>> CRSADJ;
std::vector<unsigned int> REMOVED;
//std::vector<std::vector<double>> CRSWT;

std::vector<Ftype> BETCA;
//std::vector<double> BETCB;

void wcData(const char* location)
{
	std::ifstream csv(location);
	unsigned int stid, crsid, major, grad, time = 1;
	while (csv >> stid >> crsid >> major >> grad)
	{
		stid--;
		crsid--;
		major--;
		if (stid >= STCT)
		{
			STCT = stid + 1;
			MAJORS.resize(STCT);
			GRADS.resize(STCT);
			STADJ.resize(STCT);
			//STWT.resize(STCT);
		}
		MAJORS[stid] = major;
		GRADS[stid] = grad;
		STADJ[stid].push_back(crsid);
		//STWT[stid].push_back(wt);
		if (crsid >= CRSCT)
		{
			CRSCT = crsid + 1;
			TIMES.resize(CRSCT);
			COSTS.resize(CRSCT);
			CRSADJ.resize(CRSCT);
			//CRSWT.resize(CRSCT);
		}
		TIMES[crsid] = time;
		COSTS[crsid] += time;
		CRSADJ[crsid].push_back(stid);
		//CRSWT[crsid].push_back(wt);
	}
	for (std::vector<unsigned int>& stadj : STADJ)
		std::sort(stadj.begin(), stadj.end());
	for (std::vector<unsigned int>& crsadj : CRSADJ)
		std::sort(crsadj.begin(), crsadj.end());
	STCTD = STCT;
	CRSCTD = CRSCT;
	REMOVED = std::vector<unsigned int>(CRSCT);
	BETCA = std::vector<Ftype>(STCT + CRSCT);
}

void simpleData()
{
	STADJ.resize(STCTD = STCT = 3);
	CRSADJ.resize(CRSCTD = CRSCT = 2);
	COSTS.resize(CRSCT);
	REMOVED.resize(CRSCT, false);
	TIMES.resize(CRSCT, 1);
	BETCA.resize(STCT + CRSCT);
	STADJ[0] = { 0, 1 };
	STADJ[1] = { 0 };
	STADJ[2] = { 0, 1 };
	COSTS[0] = (CRSADJ[0] = { 0, 1, 2 }).size();
	COSTS[1] = (CRSADJ[1] = { 0, 2 }).size();
}

void randomData()
{
	STADJ.resize(STCT = 1000000);
	CRSADJ.resize(CRSCT = 500000);
	STCTD = STCT;
	CRSCTD = CRSCT;
	std::default_random_engine randy;
	std::uniform_int_distribution<unsigned int> stDist(0, STCT - 1);
	std::uniform_int_distribution<unsigned int> crsDist(0, CRSCT - 1);
	for (unsigned long long int i = 0; i < 1000000ull; i++)
	{
		unsigned int stid = stDist(randy);
		unsigned int crsid = crsDist(randy);
		STADJ[stid].push_back(crsid);
		CRSADJ[crsid].push_back(stid);
	}
}
//inf 0 0
//100 174 36081
//75 272 44442
//50 541 60766
//40 715 68474
//30 1007 78458

const std::vector<unsigned int> MAXC{ 0, 36081, 44442, 60766, 68474, 78458 };

unsigned int incl()
{
	unsigned int ret = 0;
	for (unsigned int stid = 0; stid < STCT; stid++)
		for (unsigned int crsid : STADJ[stid])
			if (!REMOVED[crsid])
			{
				ret++;
				break;
			}
	return ret;
}

unsigned int cc()
{
	unsigned int ret = 0;
	unsigned int n = STCT + CRSCT;
	// replace vis with vector<unsigned int> to label specific cc's.
	std::vector<bool> vis(n, false);
	std::vector<unsigned int> stck;
	for (unsigned int i = 0; i < n; i++)
	{
		if (vis[i]) continue;
		stck.push_back(i);
		unsigned int ct = 0;
		while (!stck.empty())
		{
			unsigned int v = stck.back();
			stck.pop_back();
			if (vis[v]) continue;
			vis[v] = true;
			if (v < STCT)
			{
				ct++;
				for (unsigned int w : STADJ[v])
					if (!REMOVED[w])
						stck.push_back(w + STCT);
			}
			else
				for (unsigned int w : CRSADJ[v - STCT])
					stck.push_back(w);
		}
		if (ct > ret) ret = ct;
	}
	return ret;
}

// I attempted to convert this into an iterative algorithm with a stack, but it was slower after compiler optimization.
void bccHelper(unsigned int v, unsigned int u, unsigned int& ret, unsigned int& i, std::vector<unsigned int>& num, std::vector<unsigned int>& low, std::vector<std::pair<unsigned int, unsigned int>>& stck)
{
	bool vSt = v < STCT;
	assert(num[v] == 0);
	low[v] = num[v] = ++i;
	const std::vector<unsigned int>& adj = vSt ? STADJ[v] : CRSADJ[v - STCT];
	for (unsigned int w : adj)
	{
		if (vSt && REMOVED[w]) continue;
		if (vSt) w += STCT;
		if (num[w] == 0)
		{
			stck.emplace_back(v, w);
			// Stack overflows here do not necessarily indicate a bug; this method recurses deeply but finitely.
			// Use a large stack size in Visual Studio's project configuration (100MB is plenty)
			bccHelper(w, v, ret, i, num, low, stck);

			low[v] = std::min(low[v], low[w]);
			if (low[w] >= num[v])
			{
				std::unordered_set<unsigned int> stSet;
				unsigned int u1, u2;
				while (std::tie(u1, u2) = stck.back(), stck.pop_back(), num[u1] >= num[w])
					stSet.insert(u1 < STCT ? u1 : u2);
				assert(u1 == v && u2 == w);
				stSet.insert(vSt ? v : w);
				ret = std::max(static_cast<unsigned int>(stSet.size()), ret);
			}
		}
		else if (num[w] < num[v] && w != u)
		{
			stck.emplace_back(v, w);
			low[v] = std::min(low[v], num[w]);
		}
	}
}

unsigned int bcc()
{
	unsigned int i = 0;
	unsigned int ret = 0;
	unsigned int n = STCT + CRSCT;
	std::vector<unsigned int> num(n, 0);
	std::vector<unsigned int> low(n, 0);
	std::vector<std::pair<unsigned int, unsigned int>> stck;
	for (unsigned int j = 0; j < n; j++)
	{
		if (num[j] || j >= STCT && REMOVED[j - STCT]) continue;
		bccHelper(j, 0, ret, i, num, low, stck);
	}
	assert(stck.empty());
	return ret;
}

template<bool N = false>
Ftype betca()
{
	unsigned int crsct = 0;
	unsigned int ect = 0;
	unsigned int* crsmap = new unsigned int[CRSCT];
	for (unsigned int crsid = 0; crsid < CRSCT; crsid++)
		if (!REMOVED[crsid])
			crsmap[crsid] = crsct++;

	unsigned int* stdeg = new unsigned int[STCT];
	unsigned int** stadj = new unsigned int*[STCT];
	for (unsigned int stid = 0; stid < STCT; stid++)
	{
		unsigned int& deg = stdeg[stid] = 0;
		for (unsigned int crsid : STADJ[stid])
			deg += !REMOVED[crsid];
		ect += deg;
		unsigned int*& adj = stadj[stid] = new unsigned int[deg];
		unsigned int j = 0;
		for (unsigned int crsid : STADJ[stid])
			if (!REMOVED[crsid])
				adj[j++] = crsmap[crsid];
	}

	unsigned int* crsdeg = new unsigned int[crsct];
	const unsigned int** crsadj = new const unsigned int*[crsct];
	Ftype* weights = new Ftype[crsct];
	for (unsigned int crsid = 0; crsid < CRSCT; crsid++)
		if (!REMOVED[crsid])
		{
			unsigned int mapped = crsmap[crsid];
			crsdeg[mapped] = CRSADJ[crsid].size();
			crsadj[mapped] = CRSADJ[crsid].data();
			weights[mapped] = 0.5 / TIMES[crsid];
		}
	unsigned int n = STCT + crsct;
	Ftype* betca = new Ftype[n];

	compBetcA(STCT, crsct, ect, stdeg, crsdeg, stadj, crsadj, weights, betca);

	Ftype factor = N ? 2. / ((n - 1) * (n - 2)) : 1;

	Ftype ret = 0;
	for (unsigned int stid = 0; stid < STCT; stid++)
		ret = std::max(ret, BETCA[stid] = factor * betca[stid]);
	for (unsigned int crsid = 0; crsid < CRSCT; crsid++)
		if (!REMOVED[crsid])
			ret = std::max(ret, BETCA[STCT + crsid] = factor * betca[STCT + crsmap[crsid]]);

	delete[] crsmap;
	delete[] stdeg;
	for (unsigned int stid = 0; stid < STCT; stid++) delete[] stadj[stid];
	delete[] stadj;
	delete[] crsdeg;
	delete[] crsadj;
	delete[] weights;
	delete[] betca;
	return ret;
}

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

	unsigned int inclSt = incl();
	unsigned int maxCC = cc();
	unsigned int maxBCC = bcc();
	const auto start = std::chrono::high_resolution_clock::now();
	Ftype betc = betca<true>();
	const auto end = std::chrono::high_resolution_clock::now();
	std::cout << std::chrono::duration_cast<std::chrono::seconds>(end - start).count() << " seconds" << std::endl;
	double inclStd = inclSt * 100. / STCTD;
	double maxCCd = maxCC * 100. / STCTD;
	double maxBCCd = maxBCC * 100. / STCTD;

	std::cout << ifmt << tc << fmt << inclStd << fmt << maxCCd << fmt << maxBCCd << bfmt << betc << std::endl;
}

void testWC()
{
	std::fill(REMOVED.begin(), REMOVED.end(), false);
	// Nonascending section size
	const auto pred = [](unsigned int l, unsigned int r) -> bool
	{
		return CRSADJ[l].size() > CRSADJ[r].size();
	};

	std::vector<unsigned int> perm(CRSCT);
	std::iota(perm.begin(), perm.end(), 0);
	std::sort(perm.begin(), perm.end(), pred);

	unsigned int idx = 0;
	unsigned int tc = 0;
	for (unsigned int maxc : MAXC)
	{
		while (idx < CRSCT)
		{
			unsigned int crsid = perm[idx];
			unsigned int cost = COSTS[crsid];
			unsigned int newTC = tc + cost;
			if (newTC > maxc) break;
			REMOVED[crsid] = true;
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
	testWC();
	std::cout << std::endl;
}


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
