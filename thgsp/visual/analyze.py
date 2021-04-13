import numpy as np
import torch as th
from scipy.sparse.linalg import eigsh
from sklearn.cluster import KMeans
from thgsp.convert import to_np
from thgsp.graphs import GraphBase
import plotly.graph_objects as go
import plotly.express as px


def fast_interpolate(x, num_sample=50):
    dx = (x[1:] - x[:-1]) / (num_sample - 1)
    xall = dx.reshape(-1, 1) * np.arange(num_sample - 1) + x[:-1].reshape(-1, 1)  # len(x)-1 x num_segment
    xall = np.append(xall.reshape(-1), x[-1])  # the last point
    return xall


class Y2:
    def __init__(self, y1points, y2points):
        if len(y1points) != len(y2points):
            raise RuntimeError("The numbers of calibration points from two y-axes don't equal")
        length1 = y1points[1] - y1points[0]
        length2 = y2points[1] - y2points[0]
        self.scale_ratio21 = length2 / length1
        self.axis1_base = y1points[0]
        self.axis2_base = y2points[0]

    def y1toy2(self, y1):
        y2 = self.scale_ratio21 * (y1 - self.axis1_base) + self.axis2_base
        return y2

    def y2toy1(self, y2):
        y1 = 1 / self.scale_ratio21 * (y2 - self.axis2_base) + self.axis1_base
        return y1


def parse_band(band, fs):
    assert band.shape[-1] == 2
    num_band = band.shape[0]
    band_indices = [[] for _ in range(num_band)]
    i = 0  # band index
    j = 0  # fs index
    while j < len(fs) and i < num_band:
        f = fs[j]
        left, right = band[i]
        if f < left:
            j = j + 1
        elif f == left:
            band_indices[i].append(j)
            j = j + 1
        else:  # compare right terminal
            if f < right:
                band_indices[i].append(j)
                j = j + 1
            elif f == right:
                band_indices[i].append(j)
                i = i + 1
            else:
                i = i + 1
    return band_indices


def band_index2boundary(band_indices, ylimits, spectral_spacing):
    last_basis_of_previous_band = ylimits[0]  # set to y-axis limit by default
    split_lines = []
    for i, band in enumerate(band_indices):
        if len(band) == 0:
            split_lines.append(None)
            continue
        first_of_this_band = band[0]
        spectral_split = (spectral_spacing[first_of_this_band] + last_basis_of_previous_band) / 2
        split_lines.append(spectral_split)  # record the left split line of i-th band
        last_basis_of_previous_band = spectral_spacing[band[-1]]
    spectral_split = last_basis_of_previous_band + (
            ylimits[-1] - last_basis_of_previous_band) / 2  # separate the last band and maximum left y
    split_lines.append(spectral_split)
    return split_lines


def band_bound2y(split_lines, band, axis_translator=None):
    low_lines = []
    high_lines = []
    for i in range(len(band)):
        if split_lines[i] is None:
            continue
        low_split = split_lines[i] if axis_translator is None else axis_translator.y1toy2(split_lines[i])
        high_split = next((x for x in iter(split_lines[i + 1:]) if x))
        high_split = high_split if axis_translator is None else axis_translator.y1toy2(high_split)
        y_low_split = [low_split, low_split, band[i][0], band[i][0]]
        y_high_split = [high_split, high_split, band[i][1], band[i][1]]
        low_lines.append(y_low_split)
        high_lines.append(y_high_split)
    return np.asarray(low_lines), np.asarray(high_lines)


def compute_eigen_of_rw(G: GraphBase, k=2, which="SM"):
    L = G.L("comb").to_scipy("csr")
    D = G.D().to_scipy("csr")
    vals, vecs = eigsh(L, k=k, M=D, which=which)
    return vals, -vecs


def plot_basis(x, y, spectral_spacing, num=150, size=None):
    if y.ndim == 1:
        y = y[None, ...]
    assert len(y) == len(spectral_spacing)
    xall = fast_interpolate(x, num)
    bases = []
    for el, basis in enumerate(y):
        yall = fast_interpolate(basis, num)
        series_color = np.abs(yall)
        max_scale = series_color.max()
        basis = go.Scatter(x=xall, y=yall + spectral_spacing[el], mode='markers+lines',
                           marker=dict(color=series_color + max_scale / 4, colorscale='greys', size=size,
                                       cmax=max_scale, cmin=0),
                           showlegend=False, line=dict(color="lightgrey"), hoverinfo='skip')
        bases.append(basis)
    return bases


def plot_dots(x, all_series, spectral_spacing, threshold, labels=None, highlight_entry=None, size=None):
    if all_series.ndim == 1:
        all_series = all_series[None, ...]  # (N) ->(1,N)
    if labels is None:
        labels = np.arange(len(x))
    if highlight_entry is not None:
        print(all_series.shape)
        assert highlight_entry.shape == all_series.shape
    assert len(labels) == len(x)
    assert len(spectral_spacing) == len(all_series)  # == M

    mask = np.abs(all_series) > threshold
    if np.sum(mask) == 0:
        raise RuntimeError("Please choose a smaller threshold such that at least one point is shown")
    alldots = []
    highlights = []
    for el, series in enumerate(all_series):
        x2show = x[mask[el]]
        y2show = series[mask[el]]
        labels2show = labels[mask[el]]
        dots = go.Scatter(x=x2show, y=y2show + spectral_spacing[el],
                          mode='markers', marker=dict(color='black', size=size),
                          showlegend=False, name=f"{el}-th basis",
                          text=[f'node {label} [{y2show[i]:.3f}]' for i, label in enumerate(labels2show)],
                          hovertemplate="%{text}<br>" + "embed: %{x:.2f}<br>")
        if highlight_entry is not None:
            lth_sampled = highlight_entry[el].nonzero()[1]
            highlight = go.Scatter(x=x[lth_sampled], y=series[lth_sampled] + spectral_spacing[el],
                                   mode="markers", marker=dict(color='red', size=size * 1.8, symbol='circle-open'),
                                   showlegend=False, name=f"{el}-th sampled node",
                                   text=[f'node {labels[lth_sampled]}', ],
                                   hovertemplate="%{text}<br>" + "embed: %{x:.2f}<br>")
            highlights.append(highlight)
        alldots.append(dots)
    return alldots, highlights


def show_transform(G: GraphBase, transform_matrix, fs, highlight_entry=None, cluster=None,
                   embedding=None, amplitude_scale=1.5, amplitude_norm="max_abs", epsilon_support=0.05,
                   support_scatter_size=3, bands=None, bands_colors=None, bands_opacity=0.1, verbose=True):
    assert amplitude_norm in ("max_abs", "l2", "overall_max_abs")
    transform_matrix = to_np(transform_matrix)
    M, N = transform_matrix.shape
    if len(fs) != M:  # transform matrix and frequencies
        raise RuntimeError(f"{M} graph frequencies are to be assigned to {M} bases")

    if highlight_entry is not None:  # highlight entry
        if highlight_entry.shape != (M, N):
            raise RuntimeError(f" 'highlight_entry' should have the same size with 'transform_matrix'")

    vecs = None
    vals = None
    # Clusters
    if cluster is None:
        num_clusters = 1  # all nodes belong to one cluster
        cluster = np.ones(N)
    elif isinstance(cluster, int):
        assert cluster > 1
        num_clusters = cluster
        if verbose:
            print("Clustering the nodes with applying K-means on many eigenvectors of random walk Laplacian ...")
        vals, vecs = compute_eigen_of_rw(G, cluster + 1)
        kmeans = KMeans(n_clusters=cluster).fit(vecs)
        if verbose:
            print("Clustering Done.")
        cluster = kmeans.labels_
    else:
        if len(cluster) != N:
            raise RuntimeError(f"The cluster indicator should be of length {N}")
        num_clusters = len(np.unique(cluster))

    # Vertex embedding
    if embedding is None:
        if verbose:
            print("Computing 1-D embedding based on the 2nd random walk Laplacian ...")
        if vecs is None:
            vals, vecs = compute_eigen_of_rw(G, 2)
        idx2nd = vals.argsort()[1]  # the second smallest
        embedding = vecs[:, idx2nd]
        if verbose:
            print("Embedding Done.")

    elif embedding == "equispaced":  # for a regular embedding
        raise NotImplemented

    elif isinstance(embedding, (np.ndarray, th.Tensor, list, tuple)):
        if len(embedding) != N:
            raise RuntimeError(f"Incorrect 1-D embedding size, {len(embedding)}!={N}")
    else:
        raise TypeError(f"{type(embedding)} is not a supported type of arg 'embedding'")

    # Check embedding order with clusters
    emd_order = np.argsort(embedding)
    embedding_ordered = embedding[emd_order]
    cluster_boundaries = None
    if num_clusters > 1:
        cluster_ordered = cluster[emd_order]
        jump_nodes = (cluster_ordered[1:] - cluster_ordered[:-1]).nonzero()[0]
        if len(jump_nodes) != num_clusters - 1:
            raise RuntimeError("Vertex embedding and clusters are inconsistent")

        #  Vertical lines to separate the clusters
        cluster_boundaries = np.zeros(num_clusters + 1)
        cluster_boundaries[[0, -1]] = np.min(embedding), np.max(embedding)
        jump2nodes = jump_nodes + 1
        split_interval = np.stack((embedding_ordered[jump_nodes], embedding_ordered[jump2nodes + 1]))
        cluster_boundaries[1:-1] = np.mean(split_interval, axis=0)

    # Frequencies y axis coordinates
    spectral_spacing = np.arange(M)

    # Bands and bands_colors  Note that this can cope with more general band information.
    if bands is not None:
        if bands.ndim == 1:
            bands = np.hstack([bands[:-1, None], bands[1:, None]])
        assert bands.ndim == 2 and bands.shape[-1] == 2  # num_band, 2
        bands_colors = px.colors.qualitative.Safe if bands_colors is None else bands_colors
        if len(bands_colors) < 3:
            raise RuntimeError(f"`bands_colors({len(bands_colors)})` is not enough !');")

    # Modes scaling
    modes = transform_matrix  # each row is a basis vector(different from that in grasp)
    if amplitude_norm == "l2":
        modes = modes / np.linalg.norm(modes, axis=1, keepdims=True)
    elif amplitude_norm == "max_abs":
        modes = modes / np.abs(modes).max(1, keepdims=True)
    elif amplitude_norm == "overall_max_abs":
        modes = modes / np.abs(modes).max()
    else:
        raise TypeError(f"amplitude normalization strategy{amplitude_norm} is invalid or not supported at present")
    modes_ordered = modes[:, emd_order]

    # ---> Plot
    fig = go.Figure()
    cur_amplitude_scale = amplitude_scale * (spectral_spacing[-1] - spectral_spacing[0]) / M
    cur_curve_scale = cur_amplitude_scale / 2
    all_series = cur_curve_scale * modes_ordered

    xlim = embedding_ordered[[0, -1]]
    xlim = xlim + (xlim[-1] - xlim[0]) / 100 * np.array([-1, 1])
    # Add Horizontals and vericals
    horizontals = [go.Scatter(x=xlim, y=[i, i], mode="lines",
                              line=dict(color='lightgrey', width=support_scatter_size / 2), hoverinfo='skip') for i in
                   spectral_spacing]
    fig.add_traces(horizontals)

    yleftlimits = spectral_spacing[[0, -1]] + cur_amplitude_scale * np.array([-1, 1])
    verticals = [go.Scatter(x=[j, j], y=yleftlimits, mode="lines", hoverinfo='skip',
                            line=dict(color='lightgrey', width=support_scatter_size / 2)) for j in embedding_ordered]
    fig.add_traces(verticals)

    # plot basis vectors
    bases = plot_basis(embedding_ordered, all_series, spectral_spacing, size=support_scatter_size / 2)
    fig.add_traces(bases)

    # plot dots and highlight entries
    dots, highs = plot_dots(embedding_ordered, all_series, spectral_spacing, threshold=epsilon_support,
                            labels=emd_order,
                            highlight_entry=highlight_entry,
                            size=support_scatter_size * 1.5)
    fig.add_traces(dots)
    fig.add_traces(highs)

    # plot cluster boundaries
    if cluster_boundaries is not None:
        clusters = [go.Scatter(x=[j, j], y=yleftlimits, mode="lines", hoverinfo='skip',
                               line=dict(color='black', width=support_scatter_size * 1.5)) for j in
                    cluster_boundaries[1:-1]]
        fig.add_traces(clusters)

    translator = Y2(spectral_spacing[[0, -1]], fs[[0, -1]])
    Dx = xlim[-1] - xlim[0]
    new_xlim = [xlim[0], xlim[-1] + 0.1 * Dx]
    x_turning_point = xlim[-1] + 0.075 * Dx

    x_split = [xlim[0], xlim[-1], x_turning_point, new_xlim[-1]]
    # plot bands
    if bands is not None:
        band_indices = parse_band(bands, fs)
        bb = band_index2boundary(band_indices, yleftlimits, spectral_spacing)
        low_lines, high_lines = band_bound2y(bb, bands, translator)

        for i, low in enumerate(low_lines):
            color = bands_colors[i % len(bands_colors)]
            color = "rgba" + color[3:-1] + f",{bands_opacity})"
            if low is None:
                continue
            low_line = go.Scatter(x=x_split, y=low, line=dict(color=color, width=0.001), name=f"{i}-th band",
                                  yaxis="y2", mode='lines')

            high_line = go.Scatter(x=x_split, y=high_lines[i], fill='tonexty', fillcolor=color, name=f"{i}-th band",
                                   yaxis="y2", mode='lines', line=dict(color=color),
                                   hovertemplate="%{y:.4f}<br>")

            fig.add_trace(low_line)
            fig.add_trace(high_line)

    # plot frequencies
    spectral_spacing_y_right = translator.y1toy2(spectral_spacing)
    basis2fs = [go.Scatter(x=x_split[1:], y=[y, fs[i], fs[i]], mode="lines", yaxis='y2',
                           line=dict(color='lightgrey', width=support_scatter_size / 2),
                           name=f"{i}-th freq", hovertemplate="%{y}")
                for i, y in
                enumerate(spectral_spacing_y_right)]

    fig.add_traces(basis2fs)

    fig.update_layout(showlegend=False, template="simple_white",
                      xaxis=dict(mirror=True, ticks="inside", title_text="Vertex embedding", range=new_xlim),
                      yaxis=dict(title_text=r"$\text{Graph Frequency index  }l$", range=yleftlimits, ticks="inside"),
                      yaxis2=dict(title_text=r"$\text{Graph Frequency  }\lambda_l$", anchor="x", ticks="inside",
                                  overlaying="y", range=translator.y1toy2(yleftlimits), side="right"),
                      )

    return fig, embedding, cluster
