#pragma once

#include <torch/extension.h>

torch::Tensor dsatur_cpu(torch::Tensor& rowptr, torch::Tensor& col);