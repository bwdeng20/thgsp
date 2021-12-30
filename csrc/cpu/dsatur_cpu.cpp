#include "dsatur_cpu.h"

using namespace std;
using namespace torch::indexing;

int64_t pick_color(const unordered_set<int64_t> &used_colors, int64_t num_node) {
    int64_t i = 0;
    for (; i < num_node; i++) {
        if (!used_colors.count(i))
            break;
    }
    return i;
}


int64_t argmax(const unordered_map<int64_t, int64_t> &dict, int64_t* degree) {
    int64_t idx_max = dict.begin()->first;
    int64_t val_max = dict.begin()->second;
    int64_t deg_max = degree[idx_max];
    int64_t idx,val;
    for (auto it : dict) {
        idx = it.first;
        val = it.second;

        if (val> val_max)
        {
            idx_max = idx;
            val_max = val;
        } else if (val==val_max)
        {
            if (degree[idx]>deg_max)
            {
                idx_max = idx;
                val_max = val;
            }
        }
    }
    return idx_max;
}


torch::Tensor dsatur_coloring_cpu(const torch::Tensor& rowptr, const torch::Tensor& col) {
    int64_t u,v,n = rowptr.size(-1) - 1;
    int64_t nbr_begin, nbr_stop;
    if(n<1){
        torch::Tensor tmp= torch::arange(n,torch::dtype(torch::kLong).device(torch::kCPU));
        return tmp;
    }
    torch::Tensor vtx_color = -torch::ones(n, {torch::kLong});
    auto vtx_color_data=vtx_color.data_ptr<int64_t>();
    auto rowptr_data=rowptr.data_ptr<int64_t>();
    auto col_data=col.data_ptr<int64_t>();

    unordered_map<int64_t, unordered_set<int64_t>> distinct_colors;
    unordered_map<int64_t, int64_t> saturation_level;
    for (int64_t i = 0; i < n; i++) {
        distinct_colors[i] = unordered_set<int64_t>{-1};
        saturation_level[i] = 1; /* unclored as -1*/
    }

    auto deg = rowptr.index({Slice(1, None)}) - rowptr.index({Slice(None, -1)});
    auto deg_data=deg.data_ptr<int64_t>();

    u = torch::argmax(deg).item().toLong();
    vtx_color_data[u] = 0;
    saturation_level.erase(u);
    distinct_colors.erase(u);

    nbr_begin=rowptr_data[u];
    nbr_stop=rowptr_data[u+1];
    for(int64_t j=nbr_begin;j<nbr_stop;j++){
        v=col_data[j];
        distinct_colors[v].insert(vtx_color_data[u]);
        if (vtx_color_data[v]==-1) {
            saturation_level[v] +=1;
        }
    }

    unordered_set<int64_t> nbr_colors;
    for (int64_t i = 1; i < n; i++) {
        u = argmax(saturation_level,deg_data);
        saturation_level.erase(u);
        nbr_begin=rowptr_data[u];
        nbr_stop=rowptr_data[u+1];

        nbr_colors.clear();
        for(int64_t j=nbr_begin;j<nbr_stop;j++){
            v=col_data[j];
            nbr_colors.insert(vtx_color_data[v]);
        }

        int64_t color4u = pick_color(nbr_colors, n);
        vtx_color_data[u] = color4u;

        for(int64_t j=nbr_begin;j<nbr_stop;j++){
            v=col_data[j];
            if (vtx_color_data[v]==-1) {
                distinct_colors[v].insert(vtx_color_data[u]);
                saturation_level[v] = (int64_t) distinct_colors[v].size();

            }
        }
    }
    return vtx_color;
}
