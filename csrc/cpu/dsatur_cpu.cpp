#include "dsatur_cpu.h"

using namespace std;
using namespace torch::indexing;

long pick_color(const unordered_set<long> &used_colors, long num_node) {
    long i = 0;
    for (; i < num_node; i++) {
        if (!used_colors.count(i))
            break;
    }
    return i;
}


long argmax(const unordered_map<long, long> &dict, const long* degree) {
    long idx_max = dict.begin()->first;
    long val_max = dict.begin()->second;
    long deg_max = degree[idx_max];
    long idx,val;
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
    long u,v,n = rowptr.size(-1) - 1;
    long nbr_begin, nbr_stop;
    if(n<2){
        torch::Tensor tmp= torch::arange(n,torch::dtype(torch::kLong).device(torch::kCPU));
        return tmp;
    }
    torch::Tensor vtx_color = -torch::ones(n, {torch::kLong});
    auto vtx_color_data=vtx_color.data_ptr<int64_t>();
    auto rowptr_data=rowptr.data_ptr<int64_t>();
    auto col_data=col.data_ptr<int64_t>();

    unordered_map<long, unordered_set<long>> distinct_colors;
    unordered_map<long, long> saturation_level;
    for (long i = 0; i < n; i++) {
        distinct_colors[i] = unordered_set<long>{-1};
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
    for(long j=nbr_begin;j<nbr_stop;j++){
        v=col_data[j];
        distinct_colors[v].insert(vtx_color_data[u]);
        if (vtx_color_data[v]==-1) {
            saturation_level[v] +=1;
        }
    }

    unordered_set<long> nbr_colors;
    for (long i = 1; i < n; i++) {
        u = argmax(saturation_level,deg_data);
        saturation_level.erase(u);
        nbr_begin=rowptr_data[u];
        nbr_stop=rowptr_data[u+1];

        nbr_colors.clear();
        for(long j=nbr_begin;j<nbr_stop;j++){
            v=col_data[j];
            nbr_colors.insert(vtx_color_data[v]);
        }

        long color4u = pick_color(nbr_colors, n);
        vtx_color_data[u] = color4u;

        for(long j=nbr_begin;j<nbr_stop;j++){
            v=col_data[j];
            if (vtx_color_data[v]==-1) {
                distinct_colors[v].insert(vtx_color_data[u]);
                saturation_level[v] = (long) distinct_colors[v].size();

            }
        }
    }
    return vtx_color;
}
