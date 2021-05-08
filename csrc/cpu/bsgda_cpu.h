#pragma once

#include <torch/extension.h>

std::tuple<std::unordered_map<int64_t, std::vector<int64_t>>, std::vector<int64_t>>
computing_sets_cpu(torch::Tensor &rowptr, torch::Tensor &col, torch::Tensor &wgt, double_t T,
                   double_t mu = 0.01, int64_t p_hops = 12);

std::tuple<std::vector<int64_t>, bool>
solving_set_covering_cpu(std::unordered_map<int64_t, std::vector<int64_t>> & sets,
        std::vector<int64_t> & set_lengths, int64_t K);


std::tuple<std::vector<int64_t>,bool>greedy_gda_sampling_cpu(torch::Tensor &rowptr, torch::Tensor &col, torch::Tensor &wgt,
                                                     int64_t K,
                                                     double_t T,
                                                     double_t mu = 0.01, int64_t p_hops = 12);