#include <queue>
#include "bsgda_cpu.h"

using namespace std;
using namespace torch::indexing;

inline void clear(std::queue<int64_t> &q) {
    std::queue<int64_t> empty_q;
    std::swap(q, empty_q);
}


template<typename dtype>
int64_t argmax(const unordered_map<int64_t, dtype> &dict) {
    int64_t idx_max = dict.begin()->first;
    dtype val_max = dict.begin()->second;
    for (auto it : dict) {
        if (it.second > val_max) {
            idx_max = it.first;
            val_max = it.second;
        }
    }
    return idx_max;
}


std::tuple<std::unordered_map<int64_t, std::vector<int64_t>>, std::vector<int64_t>>
computing_sets_cpu(torch::Tensor &rowptr, torch::Tensor &col, torch::Tensor &wgt, double_t T,
                   double_t mu, int64_t p_hops){

        int64_t n = rowptr.size(-1) - 1;
    auto deg = rowptr.index({Slice(1, None)}) - rowptr.index({Slice(None, -1)});
    auto deg_data=deg.data_ptr<int64_t>();
    auto rowptr_data=rowptr.data_ptr<int64_t>();
    auto col_data=col.data_ptr<int64_t>();


    std::unordered_map<int64_t, std::vector<int64_t>> sets;
    std::vector<int64_t> set_lengths;
    set_lengths.reserve(n);

    auto scales = torch::ones({n}, wgt.dtype());
    auto scales_tmp = torch::ones({n}, wgt.dtype());
    auto in_queue = torch::zeros(n, {torch::kBool});
    auto hops = torch::zeros({n}, {torch::kLong});
    std::queue<int64_t> queue;

    int64_t nbr, nbr_begin, nbr_end, is_sampled, ripple_size, i;
    for (int64_t set_idx = 0; set_idx < n; set_idx++) {
        ripple_size = 0;
        clear(queue);
        scales.index_put_({"..."}, 1);
        in_queue.index_put_({"..."}, 0);
        hops.index_put_({"..."}, 0);

        is_sampled = 1;
        queue.push(set_idx);
        while (!queue.empty()) {
            auto idx = queue.front();
            queue.pop();
            in_queue[idx] = true;
            scales_tmp.index_put_({"..."}, scales);
            scales_tmp.clamp_(1);

            nbr_begin = rowptr_data[idx];
            nbr_end = rowptr_data[idx+1];
            torch::Tensor nbrs = col.index({Slice(nbr_begin, nbr_end)});

            auto dominator = mu * torch::sum(wgt.index({Slice(nbr_begin, nbr_end)}) / scales_tmp.index({nbrs}));
            auto s_i = ((is_sampled + mu * deg_data[idx] - T) / dominator).item<double_t>();
            scales[idx] = s_i;
            is_sampled = 0;

            if (s_i >= 1. && hops[idx].item<int64_t>() < p_hops) {
                sets[set_idx].push_back(idx);
                ripple_size++;

                for (i = nbr_begin; i < nbr_end; i++) {
                    nbr = col_data[i];
                    if (!in_queue[nbr].item<bool>()) {
                        queue.push(nbr);
                        in_queue[nbr] = true;
                        hops[nbr] += 1;
                    }
                }
            }

        }
        set_lengths.push_back(ripple_size);
    }
    return {sets, set_lengths};
}


std::tuple<std::vector<int64_t>, bool>
solving_set_covering_cpu(const std::unordered_map<int64_t, std::vector<int64_t>> &sets,
                         const std::vector<int64_t> &set_lengths, int64_t K) {

    assert(K > 0);
    auto n = (int64_t) set_lengths.size();
    int selected_num = 0;
    int64_t idx2sample = -1;
    int64_t temp_idx = -1;
    int64_t temp = 0;

    std::unordered_map<int64_t, int64_t> sets2choose;
    for (int64_t i = 0; i < n; i++)
        sets2choose[i] = (int64_t) sets.at(i).size();

    std::vector<int64_t> S;
    S.reserve(K);
    bool vf, uncovered_flag;

    torch::Tensor uncovered = torch::ones(n, torch::dtype(torch::kBool).device(torch::kCPU));
    auto* raw_uncovered = uncovered.data_ptr<bool>();

    // select the first node
    idx2sample = argmax(sets2choose);

    S.push_back(idx2sample);
    sets2choose.erase(idx2sample);
    selected_num += 1;
    for (auto i:sets.at(idx2sample)) raw_uncovered[i] = false;

    // select 2->K nodes
    uncovered_flag = torch::any(uncovered).item<bool>();
    while (selected_num < K && uncovered_flag) {
        for (auto it = sets2choose.begin(); it != sets2choose.end();) {
            temp = 0;
            temp_idx = it->first;

            for (int64_t j:sets.at(temp_idx)) {
                if (raw_uncovered[j]) temp += 1;
            }
            if (temp == 0) {
                it = sets2choose.erase(it);
            } else {
                sets2choose[temp_idx] = temp;
                ++it;
            };
        }

        idx2sample = argmax(sets2choose);
        sets2choose.erase(idx2sample);
        for (auto i:sets.at(idx2sample)) raw_uncovered[i] = false;
        selected_num += 1;
        S.push_back(idx2sample);
        uncovered_flag = torch::any(uncovered).item<bool>();
    }
    vf = !uncovered_flag;
    return {S, vf};
}


std::tuple<std::vector<int64_t>, bool>
greedy_gda_sampling_cpu(torch::Tensor &rowptr, torch::Tensor &col, torch::Tensor &wgt,
                    int64_t K,
                    double_t T,
                    double_t mu, int64_t p_hops) {
    std::unordered_map<int64_t, std::vector<int64_t>> sets;
    std::vector<int64_t> set_lengths;
    std::tie(sets, set_lengths) = computing_sets_cpu(rowptr, col, wgt, T, mu, p_hops);
    std::vector<int64_t> S;
    bool vf;
    std::tie(S, vf) = solving_set_covering_cpu(sets, set_lengths, K);
    return {S, vf};
}