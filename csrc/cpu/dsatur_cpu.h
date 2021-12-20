#pragma once

#include <torch/extension.h>

torch::Tensor dsatur_coloring_cpu(const torch::Tensor& rowptr, const torch::Tensor& col);