#include "dsatur_cpu.h"

using namespace torch::indexing;
using namespace std;

int pick_color(const unordered_set<int> &used_colors, int &num_node) {
    int i = 0;
    for (; i < num_node+1; i++) {
        if (!used_colors.count(i))
            break;
    }
    return i;
}

int argmax(const unordered_map<int, int> &dict) {
    int idx_max = dict.begin()->first;
    int val_max = dict.begin()->second;
    for (auto it : dict) {
        if (it.second > val_max) {
            idx_max = it.first;
            val_max = it.second;
        }
    }
    return idx_max;
}

torch::Tensor dsatur_cpu(torch::Tensor& rowptr, torch::Tensor& col) {
    int n = rowptr.size(-1) - 1;

    auto deg = rowptr.index({Slice(1, None)}) - rowptr.index({Slice(None, -1)});


    unordered_map<int, unordered_set<int>> distinct_colors;
    unordered_map<int, int> saturation_level;
    for (int i = 0; i < n; i++) {
        distinct_colors[i] = unordered_set<int>{};
        saturation_level[i] = 0;
    }

    torch::Tensor vtx_color = -torch::ones(n, {torch::kLong});

    int u = torch::argmax(deg).item().toInt();
    torch::Tensor nbr = col.index({Slice(rowptr[u].item().toInt(), rowptr[u + 1].item().toInt())});
    int64_t *raw_nbr = nbr.data_ptr<int64_t>();
    int num_nbr = nbr.size(0);

    for (int j = 0; j < num_nbr; j++) {
        int v = *(raw_nbr + j);
        distinct_colors[v].insert(0);
        saturation_level[v] += 1;
    }

    vtx_color[u] = 0;
    saturation_level.erase(u);  // only the saturation of uncolored nodes are in need.

    for (int i = 1; i < n; i++) {
        u = argmax(saturation_level);
        nbr = col.index({Slice(rowptr[u].item().toInt(), rowptr[u + 1].item().toInt())});
        num_nbr = nbr.size(0);
        raw_nbr = nbr.data_ptr<int64_t>();

        unordered_set<int> nbr_colors = {};
        for (int j = 0; j < num_nbr; j++) {
            int v = *(raw_nbr + j);
            nbr_colors.insert(vtx_color[v].item<int64_t>());
        }

        int color4u = pick_color(nbr_colors, n);

        vtx_color[u] = color4u;
        saturation_level.erase(u);

        for (int j = 0; j < num_nbr; j++) {
            int v = *(raw_nbr + j);
            unordered_set<int> &temp = distinct_colors[v];
            temp.insert(color4u);
            if (saturation_level.count(v))
                saturation_level[v] = (int64_t) temp.size();
        }
    }
    return vtx_color;
}
