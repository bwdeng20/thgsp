#include <torch/script.h>
#include "cpu/bsgda_cpu.h"

std::tuple<std::unordered_map<int64_t, std::vector<int64_t>>, std::vector<int64_t>>
computing_sets(torch::Tensor &rowptr, torch::Tensor &col, torch::Tensor &wgt, double_t T,
                   double_t mu = 0.01, int64_t p_hops = 12){
    if (rowptr.device().is_cuda()){
    #ifdef WITH_CUDA
         AT_ERROR("No CUDA version supported");
    #else
         AT_ERROR("Not compiled with CUDA support");
    #endif
    } else{
        return computing_sets_cpu(rowptr,col,wgt,T,mu,p_hops);
    }
}


std::tuple<std::vector<int64_t>,bool>greedy_gda_sampling(torch::Tensor &rowptr, torch::Tensor &col, torch::Tensor &wgt,
                                                     int64_t K,
                                                     double_t T,
                                                     double_t mu = 0.01, int64_t p_hops = 12){
    if (rowptr.device().is_cuda()){
    #ifdef WITH_CUDA
         AT_ERROR("No CUDA version supported");
    #else
         AT_ERROR("Not compiled with CUDA support");
    #endif
    } else{
        return greedy_gda_sampling_cpu(rowptr,col,wgt,K,T,mu,p_hops);
    }
}

std::tuple<std::vector<int64_t>, bool>
solving_set_covering(const std::unordered_map<int64_t, std::vector<int64_t>> & sets,
        const std::vector<int64_t> & set_lengths, int64_t K){
        return solving_set_covering_cpu(sets,set_lengths,K);
    }

static auto registry = torch::RegisterOperators().op("thgsp::computing_sets", &computing_sets)
                                                 .op("thgsp::solving_set_covering", &solving_set_covering)
                                                 .op("thgsp::greedy_gda_sampling",&greedy_gda_sampling);
