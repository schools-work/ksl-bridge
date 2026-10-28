// CPU reference port of ml/decoding/ctc_beam.py (no LM; LM scores are passed in per extension by the caller).
// Build with: python -m ml.kernels.ctc_beam_cuda.build   Acceptance: identical N-best on tests/golden/cslr/cases.json.
#include <torch/extension.h>
#include <map>
#include <vector>
#include <cmath>
#include <algorithm>

static inline double lse2(double a, double b) {
    if (a == -INFINITY) return b; if (b == -INFINITY) return a;
    double m = std::max(a, b); return m + std::log(std::exp(a - m) + std::exp(b - m));
}

std::vector<std::pair<std::vector<int64_t>, double>> beam_search(torch::Tensor log_probs, int64_t beam, double beta,
                                                                 double prune, std::vector<int64_t> skip) {
    auto lp = log_probs.accessor<float, 2>();
    int64_t T = log_probs.size(0), C = log_probs.size(1);
    using Prefix = std::vector<int64_t>;
    struct B { double pb = -INFINITY, pnb = -INFINITY; };
    std::map<Prefix, B> beams; beams[{}].pb = 0.0;
    std::map<Prefix, double> bonus; bonus[{}] = 0.0;
    for (int64_t t = 0; t < T; ++t) {
        std::map<Prefix, B> nxt;
        for (auto& [prefix, b] : beams) {
            for (int64_t c = 0; c < C; ++c) {
                double p = lp[t][c];
                if (p < prune || std::find(skip.begin(), skip.end(), c) != skip.end()) continue;
                if (c == 0) { auto& nb = nxt[prefix]; nb.pb = lse2(nb.pb, lse2(b.pb + p, b.pnb + p)); continue; }
                Prefix np = prefix; np.push_back(c);
                if (!bonus.count(np)) bonus[np] = bonus[prefix] + beta;
                auto& nn = nxt[np];
                if (!prefix.empty() && prefix.back() == c) {
                    nn.pnb = lse2(nn.pnb, b.pb + p);
                    auto& same = nxt[prefix]; same.pnb = lse2(same.pnb, b.pnb + p);
                } else { nn.pnb = lse2(nn.pnb, lse2(b.pb + p, b.pnb + p)); }
            }
        }
        std::vector<std::pair<Prefix, B>> v(nxt.begin(), nxt.end());
        std::sort(v.begin(), v.end(), [&](auto& x, auto& y) {
            return lse2(x.second.pb, x.second.pnb) + bonus[x.first] > lse2(y.second.pb, y.second.pnb) + bonus[y.first]; });
        if ((int64_t)v.size() > beam) v.resize(beam);
        beams = std::map<Prefix, B>(v.begin(), v.end());
    }
    std::vector<std::pair<std::vector<int64_t>, double>> out;
    for (auto& [p, b] : beams) out.push_back({p, lse2(b.pb, b.pnb) + bonus[p]});
    std::sort(out.begin(), out.end(), [](auto& a, auto& b) { return a.second > b.second; });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("beam_search", &beam_search, "CTC prefix beam search (no LM)"); }
