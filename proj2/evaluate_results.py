import json
import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pathlib as pl
import seaborn as sns


def _str_to_float(str_num):
    return float(
        ('.' + str_num[1:]) 
            if str_num[0] == '0' and len(str_num) > 1 
            else str_num
    )

def _get_matched_files_with_props(model, n, beta, f, path):
    matched_files = []

    for file in pl.Path(path).iterdir():
        if file.suffix != '.out':
            continue

        file_model, file_n, _, file_beta, file_f =  \
            file.stem.split('_')[0:5]

        if (
            model ==  file_model and 
            n == int(file_n) and
            math.isclose(beta, _str_to_float(file_beta)) and
            math.isclose(f, _str_to_float(file_f))
        ):
            matched_files.append(str(file))
    
    return matched_files

def _avg_results_in_files(files, n, f):
    results = {}
    iters = {}
    s = int(round(n * f))

    for file in files:
        f = open(file, 'r')
        line = f.readline()

        while line and line != '\n':
            values = [val.strip() for val in line.split(',')]

            if values[2] not in results:
                if len(values) == 3:
                    results[values[2]] = []
                    iters[values[2]] = 0
                elif len(values) == 4:
                    results[values[2]] = {}
                    iters[values[2]] = {}

            if len(values) == 4:
                param = int(values[3])

                if param not in results[values[2]]:
                    results[values[2]][param] = []
                    iters[values[2]][param] = 0

                results[values[2]][param].append(n - int(values[1]) - s)
                iters[values[2]][param] += 1
            else:
                results[values[2]].append(n - int(values[1]) - s)
                iters[values[2]] += 1

            line = f.readline()

        f.close()

    _avg_results(results, iters)
    return results

def _avg_results(results, iters):    
    for key in results:
        if isinstance(results[key], dict):
            _avg_results(results[key], iters[key])
        else:
            results[key] = sum(results[key]) / iters[key]

def _results_same_props(model, n, beta, f, path):
    files = _get_matched_files_with_props(model, n, beta, f, path)
    return _avg_results_in_files(files, n, f)

def get_results_by_model(
    models, 
    ns, 
    betas, 
    fs, 
    vacc_prots, 
    vacc_prots_w_params, 
    path='./results/'
):
    results = {}

    for model in models:
        results[model] = {}

        for n in ns:
            results[model][n] = _get_results_from_files(
                model, 
                n, 
                betas, 
                fs,
                vacc_prots,
                vacc_prots_w_params,
                path
            )

    return results

def _get_results_from_files(
    model, 
    n, 
    betas, 
    fs, 
    vacc_prots, 
    vacc_prots_w_params,
    path='./results/'
):
    results_b_f = {v:{b:[] for b in betas} for v in vacc_prots + ['NO']}

    for v in vacc_prots_w_params:
        results_b_f[v] = {b:{} for b in betas}


    for i in range(len(betas)):
        if 0 in fs:
            res = _results_same_props(
                model,
                n,
                betas[i],
                0,
                path
            )

            results_b_f['NO'][betas[i]].append(res['NO']) 

        for j in range(len(fs)):
            if fs[j] == 0:
                continue

            res = _results_same_props(
                model, 
                n, 
                betas[i], 
                fs[j], 
                path
            )

            for v in res:
                if v not in vacc_prots_w_params:
                    results_b_f[v][betas[i]].append(res[v])
                else:
                    for p in res[v]:
                        if p not in results_b_f[v][betas[i]]:
                            results_b_f[v][betas[i]][p] = []

                        results_b_f[v][betas[i]][p].append(res[v][p])

    return results_b_f

def coords_fv_fr_relation(
    results, 
    fs, 
    vacc_with_params=['RW', 'TS'], 
    store_path='./f_relations/'
):
    for model in results:
        for n in results[model]:
            for v in results[model][n]:
                if v == 'NO':
                    continue

                for beta in results[model][n][v]:
                    if v in vacc_with_params:
                        for p in results[model][n][v][beta]:
                            f_name = (
                                store_path + 
                                '{}_{}_{}_{}_{}.coords'.format(
                                    model,
                                    n,
                                    beta,
                                    v,
                                    p
                                )
                            )

                            f = open(f_name, 'w')
                            f.write('{} {}\n'.format(
                                0,
                                results[model][n]['NO'][beta][0] / n)
                            )

                            for i in range(len(
                                results[model][n][v][beta][p]
                            )):
                                f.write('{} {}\n'.format(
                                    fs[i+1],
                                    results[model][n][v][beta][p][i] / n
                                ))

                            f.close()
                    else:
                        f_name = (
                            store_path + '{}_{}_{}_{}.coords'.format(
                                model,
                                n,
                                beta,
                                v
                            )
                        )

                        f = open(f_name, 'w')
                        f.write('{} {}\n'.format(
                                0,
                                results[model][n]['NO'][beta][0] / n
                            )
                        )

                        for i in range(len(results[model][n][v][beta])):
                            f.write('{} {}\n'.format(
                                fs[i+1],
                                results[model][n][v][beta][i] / n
                            ))

                        f.close()

def write_results(
    results, 
    betas, 
    fs, 
    c_vacc_rng, 
    c_info_rng, 
    vacc_with_params,
    path='./matrices/'
):
    for model in results:
        for n in results[model]:
            for beta in betas:
                xs = {}

                for v in results[model][n]:
                    if v == 'NO':
                        continue

                    x, f, p = None, None, None

                    if v in vacc_with_params:
                        x, f, p = compute_x_with_vacc_params(
                            model,
                            n,
                            v,
                            list(results[model][n][v][beta].keys()),
                            results[model][n]['NO'][beta][0],
                            results[model][n][v][beta],
                            fs,
                            c_vacc_rng,
                            c_info_rng
                        )

                    else:
                        x, f = compute_x(
                            model,
                            n,
                            v,
                            results[model][n]['NO'][beta][0],
                            results[model][n][v][beta],
                            fs,
                            c_vacc_rng,
                            c_info_rng,
                        )
                    # heatmap(x, [], [])
                    # ax = plt.gca()
                    # ax.set_title(v + ' - x')
                    # plt.show()
                    # heatmap(f, [], [])
                    # ax.set_title(v + ' - f')
                    # plt.show()
                    # if p is not None:
                    #     heatmap(x, [], [])
                    #     ax.set_title(v + ' - p')
                    #     plt.show()
                    write_in_file(
                        model, 
                        n, 
                        v, 
                        beta, 
                        c_vacc_rng, 
                        c_info_rng, 
                        x, 
                        f, 
                        p, 
                        path
                    )

                    xs[v] = x

                b_x = compute_best_vacc_prot(xs, c_vacc_rng, c_info_rng)
                # print(b_x)
                # vmap = {
                #     0: 'Random',
                #     1: 'BFS',
                #     2: 'DFS',
                #     3: 'Acquaintance',
                #     4: 'Random Walk',
                #     5: 'TSH',
                #     6: 'Degree',
                #     7: 'Coreness'
                # }

                # n_v = len(vmap)
                # numt = 4
                # nticks = np.linspace(0, len(c_info_rng) - 1, numt, dtype=np.int)
                # ticklabels = ["{:.1e}".format(c_info_rng[idx]) for idx in nticks]
                # cmap = sns.color_palette("deep", n_v)
                # ax = sns.heatmap(b_x, cmap=cmap, yticklabels=ticklabels, xticklabels=ticklabels)
                # ax.set_yticks(nticks)
                # ax.set_xticks(nticks)
                # ax.invert_yaxis()
                # # Get the colorbar object from the Seaborn heatmap
                # colorbar = ax.collections[0].colorbar
                # # The list comprehension calculates the positions to place the labels to be evenly distributed across the colorbar
                # r = colorbar.vmax - colorbar.vmin
                # colorbar.set_ticks([colorbar.vmin + 0.5 * r / (n_v) + r * i / (n) for i in range(n_v)])
                # colorbar.set_ticklabels(list(vmap.values()))

                # plt.show()
                write_model(
                    (
                        path + model + '_' + str(n) + '_' + str(beta) + 
                        '_best.csv'
                    ), 
                    b_x,
                    c_vacc_rng,
                    c_info_rng
                )

def compute_x_with_vacc_params(
    model,
    n,
    v,
    params,
    beta_f_zero,
    beta_f_col_per_param,
    fs,
    c_vacc_rng,
    c_info_rng,
    max_type='local'
):
    x_aux = np.zeros((
        len(c_vacc_rng), 
        len(c_info_rng), 
        len(beta_f_col_per_param), 
        len(fs)
    ))

    for i in range(len(c_vacc_rng)):
        for j in range(len(c_info_rng)):
            for k in range(len(params)):
                for l in range(len(fs)):
                    x_aux[i, j, k, l] = (
                        beta_f_zero - 
                        beta_f_col_per_param[params[k]][l] -
                        n * fs[l] * c_vacc_rng[i] -
                        compute_cost(v, model, n, fs[l], params[k]) * 
                        c_info_rng[j]
                    ) / n
    
    x = np.zeros((len(c_vacc_rng), len(c_info_rng)))
    f = np.zeros((len(c_vacc_rng), len(c_info_rng)))
    p = np.zeros((len(c_vacc_rng), len(c_info_rng)))

    for i in range(len(c_vacc_rng)):
        for j in range(len(c_info_rng)):
            _, _, p_idx, f_idx = np.unravel_index(
                np.argmax(x_aux[i, j], axis=None), 
                x_aux.shape
            )

            p[i, j] = params[p_idx]
            f[i, j] = fs[f_idx]
            x[i, j] = np.max(x_aux[i, j])

    return x, f, p

def compute_x(
    model,
    n, 
    v,
    beta_f_zero,
    beta_f_col,
    fs, 
    c_vacc_rng, 
    c_info_rng, 
    max_type='local'
):
    x = np.ones((len(c_vacc_rng), len(c_info_rng), len(beta_f_col)))

    for i in range(len(c_vacc_rng)):
        for j in range(len(c_info_rng)):
            for k in range(len(fs)):
                x[i, j, k] = (
                    beta_f_zero - beta_f_col[k] - 
                    n * fs[k] * c_vacc_rng[i] -
                    compute_cost(v, model, n, fs[k]) * c_info_rng[j]
                ) / n

    if max_type == 'global':
        max_idx = np.unravel_index(np.argmax(x), x.shape)
        return x[:,:,max_idx[-1]], fs[1+max_idx[-1]]
    else:
        return (
            np.max(x, axis=2), 
            np.choose(np.argmax(x, axis=2), fs)
        )

def compute_cost(vacc_prot, model, n, f, *args):
    if vacc_prot == 'RA' or vacc_prot == 'NO':
        return 0
    elif vacc_prot == 'AC' or vacc_prot == 'BF' or vacc_prot == 'DF':
        return n * f
    elif vacc_prot == 'RW':
        if model == 'configuration-model':
            return args[0] * (1 - 3 / (2.5 + 3))
        else:
            return args[0] * (1 - 3 / (3 + 3))
    elif vacc_prot == 'TS':
        return 2 * args[0]
    elif vacc_prot == 'DE' or vacc_prot == 'CO' or vacc_prot == 'CI':
        return n

def compute_best_vacc_prot(
    xs, 
    c_v_rng, 
    c_i_rng, 
    map={
        'RA': 0,
        'BF': 1,
        'DF': 2,
        'AC': 3,
        'RW': 4,
        'TS': 5,
        'DE': 6,
        'CO': 7,
        'CI': 8,
    }
):
    best_vacc = np.zeros((len(c_v_rng), len(c_i_rng)), dtype=int)

    for i in range(len(c_v_rng)):
        for j in range(len(c_i_rng)):
            best_vacc[i, j] = map['RA']
            best_val = xs['RA'][i, j]

            for vacc, x in xs.items():
                if vacc == 'RA':
                    continue

                if x[i, j] > best_val:
                    best_val = x[i, j]
                    best_vacc[i, j] = map[vacc]

    return best_vacc

def write_in_file(model, n, v, beta, c_v_rng, c_i_rng, x, f, p, path):
    common_name = (
        model + '_' + str(n) + '_' + str(beta).replace('.', '') + 
        '_' + v + '.csv'
    )

    x_f_name = path + 'x_' + common_name
    f_f_name = path + 'f_' + common_name
    p_f_name = path + 'p_' + common_name

    for arr, f_name in ((x, x_f_name), (f, f_f_name), (p, p_f_name)):
        if arr is None:
            continue

        write_model(f_name, arr, c_v_rng, c_i_rng)

def write_model(file_name, arr, c_v_rng, c_i_rng):
    f = open(file_name, 'w')

    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            f.write('{} {} {}\n'.format(
                i, j, arr[i, j]
            )) 

    f.close()

def plot_avg_results(
    n,
    beta_f_matrix, 
    betas, 
    fs, 
    title, 
    c_vacc_rng, 
    c_info_rng,
    vacc_protocol
):
    for i in range(len(betas)):
        x, f_opt = compute_x(
            n,
            beta_f_matrix[i, :], 
            fs, 
            c_vacc_rng, 
            c_info_rng,
            vacc_protocol,
            'local'
        )

        if betas[i] == 2:
            print(x)

        heatmap(x, [], [], cmap='PuOr', cbarlabel=r'$\chi_{opt}$')
        plt.show()
        heatmap(f_opt, [], [], cmap='PuOr', cbarlabel=r'$f_{opt}$')
        plt.show()


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def check_best_vacc_protocol(
    n, 
    beta_f_matrix, 
    betas,
    beta_idx, 
    fs, 
    c_vacc, 
    c_info, 
    vacc_protocols=[
        # random, 
        # acquaintance, 
        # two_step_heuristic, 
        # degree, 
        # coreness, 
        # collective_influence
    ]
):
    max_val = -math.inf
    best_vacc_prot = None

    for vacc_prot in vacc_protocols:
        results, _ = compute_x(
            n, 
            beta_f_matrix[beta_idx, :], 
            fs, 
            [c_vacc], 
            [c_info],
            vacc_prot,
            'local'
        )

        print(results[0, 0])
        print(vacc_prot.__name__)

        if results[0, 0] > max_val:
            max_val = results[0, 0]
            best_vacc_prot = vacc_prot.__name__

    return max_val, best_vacc_prot

if __name__ == "__main__":
    betas = [0.5, 1, 2, 4, 8, 16, 32]
    fs = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    # results = get_results_from_files(
    #     'configuration-model', 
    #     10000, 
    #     betas, 
    #     fs
    # )

    # plot_avg_results(
    #     10000,
    #     results, 
    #     betas, 
    #     fs, 
    #     '', 
    #     np.logspace(-6, 0, num=20), 
    #     np.logspace(-6, 0, num=20), 
    #     collective_influence
    # )

    # print(check_best_vacc_protocol(
    #     10000, 
    #     results, 
    #     betas, 
    #     2, 
    #     fs,
    #     1e-4, 
    #     0.1
    # ))

    results = get_results_by_model(
        ['ba', 'configuration-model', 'dms'],
        [625, 1250], 
        betas, 
        fs, 
        ['RA', 'BF', 'DF', 'AC', 'DE', 'CO'], 
        ['RW', 'TS']
    )

    # print(json.dumps(results, indent=4))
    # print(results)

    write_results(
        results, 
        betas, 
        fs[1:], 
        np.logspace(-6, 0, num=20), 
        np.logspace(-6, 0, num=20), 
        ['RW', 'TS']
    )

    # coords_fv_fr_relation(results, fs)
