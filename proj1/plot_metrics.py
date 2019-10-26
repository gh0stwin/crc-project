import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import sys


DC = 'Degree Centrality'

def graph_metrics_to_plots_both_models():
    models = get_metrics_by_graph_and_n('./results/metrics.out', [1])
    models = avg_metrics(models) 

    ba = {}
    dms = {}

    for model_name in models:
        if 'ba' in model_name:
            for metric in models[model_name].keys():
                ba[metric] = models[model_name][metric]
        elif 'dms' in model_name:
            for metric in models[model_name].keys():
                dms[metric] = models[model_name][metric]

    # Delete gcc
    ba.pop('Global Clustering Coefficient', None)
    dms.pop('Global Clustering Coefficient', None)

    print(ba)
    print(dms)

    funcs = [{
        'Average Degree': lambda x: (4 * x - 6) / x,
        'Variance': square_root,
        'Local Clustering Coefficient': ba_clsut,
        'Degree Assortativity': lambda x: np.full(len(x), 0)
    }, {
        'Average Degree': lambda x: (4 * x - 6) / x,
        'Variance': square_root,
        'Local Clustering Coefficient': constant,
        'Degree Assortativity': lambda x: np.full(len(x), 0)
    }]

    fit = [[True, True, False, False], [True, True, True, False]]

    legends = [[
        # ('BA - Observed Average Degree', 'BA, DMS - Expected Average Degree: ' + r'$\frac{2(2(N-2)+1)}{N}$'),
        ('BA - Observed variance', 'BA - Quadratic function fit: a=%5.4f'),
        ('BA - Observed Average Local Clustering Coefficient', r'BA - Expected Average Clustering Coefficient: $\frac{4}{m}\frac{\left(\ln{N} \right)^2}{N}=\frac{\left(\ln{N} \right)^2}{2N}$'),
        ('BA - Observed Degree Assortativity', 'BA - Expected Degree Assortativity'),
    ], [
        # ('DMS - Observed Average Degree', 'DMS - Expected Average Degree: ' + r'$\frac{2(2(N-2)+1)}{N}$'),
        ('DMS - Observed variance', 'DMS - Quadratic function fit: a=%5.4f'),
        ('DMS - Observed Average Local Clustering Coefficient', r'DMS - Constant function fit: $k=\frac{5}{6}$'),
        ('DMS - Observed Degree Assortativity', 'DMS - Expected Degree Assortativity'),
    ]]

    xlabels = [r'$N$'] * 4
    ylabels = [r'$\sigma^2$', r'$<C>$',r'$r$']
    y_maxlim = [[2500, 0.08, 0.2], [60000, 1, 0.2]]
    y_minlim = [[0, 0, 0, -0.2], [0, 0, 0, -0.4]]

    plot_both_avg_metrics(ba, dms, funcs, fit, legends, xlabels, ylabels, y_minlim, y_maxlim)

def graph_metrics_to_plots():
    models = get_metrics_by_graph_and_n('./results/metrics.out', [1, 2, 3, 4, 5])
    models = avg_metrics(models)
    ba = {}
    dms = {}

    for model_name in models:
        if 'ba' in model_name:
            for metric in models[model_name].keys():
                ba[metric] = models[model_name][metric]
        elif 'dms' in model_name:
            for metric in models[model_name].keys():
                dms[metric] = models[model_name][metric]

    # Delete gcc
    ba.pop('Global Clustering Coefficient', None)
    dms.pop('Global Clustering Coefficient', None)

    print(ba)
    print(dms)

    funcs = {
        'Average Degree': lambda x: (4 * x - 6) / x,
        'Variance': quadratic,
        'Local Clustering Coefficient': ba_clsut,
        'Degree Assortativity': lambda x: np.full(len(x), 0)
    }

    fit = [False, True, False, False]

    # titles = [
    #     'Barabási-Albert Model - Average Degree when \nm (number of ' + 
    #     'links that connect the new node \nto the existing nodes in ' + 
    #     'the network) is 2',
    #     'Barabási-Albert Model - Variance when \nm (number of ' + 
    #     'links that connect the new node \nto the existing nodes in ' + 
    #     'the network) is 2',
    #     'Barabási-Albert Model - Average Local Clustering\n Coefficient when m (number of ' + 
    #     'links that connect the\n new node to the existing nodes in ' + 
    #     'the network) is 2',
    #     'Barabási-Albert Model - Pearson Correlation Coefficient when \nm (number of ' + 
    #     'links that connect the new node \nto the existing nodes in ' + 
    #     'the network) is 2',
    # ]

    legends = [
        ('Observed Average Degree', 'Expected Average Degree: ' + r'$\frac{2(2(N-2)+1)}{N}$'),
        ('Observed variance', 'Quadratic function fit: a=%5.3f, b=%5.3f, c=%5.3f'),
        ('Observed Average Local Clustering Coefficient', r'Expected Average Clustering Coefficient: $\frac{4}{m}\frac{\left(\ln{N} \right)^2}{N}=\frac{\left(\ln{N} \right)^2}{2N}$'),
        ('Observed Degree Assortativity', 'Expected Degree Assortativity'),
    ]

    xlabels = [r'Network Size $N$'] * 4
    ylabels = [r'$<k>$', r'$\sigma^2$', r'$<C>$',r'$r$']
    y_maxlim = [5, 8000000, 0.08, 0.2]
    y_minlim = [0, 0, 0, -0.2]
    plot_avg_metrics(ba, funcs, fit, legends, xlabels, ylabels, y_minlim, y_maxlim)

    funcs = {
        'Average Degree': lambda x: (4 * x - 6) / x,
        'Variance': quadratic,
        'Local Clustering Coefficient': constant,
        'Degree Assortativity': lambda x: np.full(len(x), 0)
    }

    fit = [False, True, True, False]

    # titles = [
    #     'Dorogovtsev-Mendes-Samukhin Model - Average Degree when \nm (number of ' + 
    #     'links that connect the new node \nto the existing nodes in ' + 
    #     'the network) is 2',
    #     'Dorogovtsev-Mendes-Samukhin Model - Variance when \nm (number of ' + 
    #     'links that connect the new node \nto the existing nodes in ' + 
    #     'the network) is 2',
    #     'Dorogovtsev-Mendes-Samukhin Model - Average Local Clustering\n Coefficient when m (number of ' + 
    #     'links that connect the\n new node to the existing nodes in ' + 
    #     'the network) is 2',
    #     'Dorogovtsev-Mendes-Samukhin Model - Pearson Correlation Coefficient when \nm (number of ' + 
    #     'links that connect the new node \nto the existing nodes in ' + 
    #     'the network) is 2'
    # ]

    legends = [
        ('Observed Average Degree', 'Expected Average Degree: ' + r'$\frac{2(2(N-2)+1)}{N}$'),
        ('Observed variance', 'Quadratic function fit: a=%5.3f, b=%5.3f, c=%5.3f'),
        ('Observed Average Local Clustering Coefficient', r'Constant function fit: k=%5.3f'),
        ('Observed Degree Assortativity', 'Expected Degree Assortativity'),
    ]

    y_maxlim = [5, 600000, 1, 0.2]
    y_minlim = [0, 0, 0, -0.3]
    plot_avg_metrics(dms, funcs, fit, legends, xlabels, ylabels, y_minlim, y_maxlim)

def square_root(x, a, b):
    return np.sqrt(x * a) + b * x
    
def constant(x, a):
    return np.full(len(x), a)

def inv(x, a):
    return a / x

def inv_power_law(x, a):
    return a ** (-1 * x)

def neg_power(x, a, b, c):
    return a * np.exp(-1 * x) + c

def neg_inv_power(x, a, b, c):
    return a * np.exp(-b / x) + c

def avg_dist_func(x, gamma):
    return x ** ((2 - gamma) / (gamma - 1))

def linear(x, m):
    return m * x

def ba_clsut(x):
    return (np.log(x) ** 2) / (2 * x)

def quadratic(x, a, b, c):
    return a * x ** 2 + b * x + c

def expon(x, a, b, c):
    return a ** (x + b) + c

def plot_dists(models, title='a'):
    for model_name in models.keys():
        for metric in models.keys():
            if metric == DC:
                continue
            
            plt.plot(models[model_name][DC], models[model_name][metric], '.')
            plt.title(title)

        plt.savefig('./plots/' + title + '.png')
        plt.show()


        name, n = model_name.split('-')
        plt.plot(np.arange(int(n)), models[model_name][DC], '.')
        plt.savefig('./plots/Degree Distribution for ' + model_name.split('-')[0].upper() + ' with ' + n + 'vertices.png')
        plt.show()

def plot_avg_metrics(models, funcs, fit, legends, xlabels, ylabels, y_minlim, y_maxlim):
    i = 0

    for metric in models.keys():
        popt = ()

        if fit[i]:
            popt, _ = curve_fit(funcs[metric], models[metric]['x'], models[metric]['y'], maxfev=5000000)

        xdata = np.linspace(models[metric]['x'][0], models[metric]['x'][-1], 50)
        plt.plot(models[metric]['x'], models[metric]['y'], '.', xdata, funcs[metric](xdata, *popt))
        plt.ylim(ymin=y_minlim[i], ymax=y_maxlim[i])
        plt.xlim(xmin=0)

        if fit[i]:
            plt.legend((legends[i][0], legends[i][1] % tuple(popt)))
        else:
            plt.legend(legends[i])

        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])
        plt.show()
        i += 1

def plot_both_avg_metrics(ba, dms, funcs, fit, legends, xlabels, ylabels, y_minlim, y_maxlim):
    plt.figure(figsize=(4.7, 4.7))
    i = 0

    for metric in ba.keys():
        popt1, popt2 = (), ()

        if fit[0][i]:
            popt1, _ = curve_fit(funcs[0][metric], ba[metric]['x'], ba[metric]['y'], maxfev=5000000)

        if fit[1][i]:
            popt2, _ = curve_fit(funcs[1][metric], dms[metric]['x'], dms[metric]['y'], maxfev=5000000)

        xdata1 = np.linspace(ba[metric]['x'][0], ba[metric]['x'][-1], 50)
        xdata2 = np.linspace(dms[metric]['x'][0], dms[metric]['x'][-1], 50)
        # _1, = plt.plot(ba[metric]['x'], ba[metric]['y'], 'ko', label=legends[0][i][0])

        label=''
        # label = legends[0][i][1]
        # if fit[0][i]:
        #     label = legends[0][i][1] % tuple(popt1)

        # _2, = plt.plot(xdata1, funcs[0][metric](xdata1, *popt1), 'r-', label=label)
        _3 = plt.scatter(dms[metric]['x'], dms[metric]['y'], s=85, facecolors='none', edgecolors='k', label=legends[1][i][0])

        # label = legends[1][i][1]
        # if fit[1][i]:
        #     label = legends[1][i][1] % tuple(popt2)

        _4, = plt.plot(xdata2, funcs[1][metric](xdata2, *popt2), 'b-', label=label)

        # plt.semilogy()
        # plt.semilogx()
        plt.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
        plt.ylim(ymin=min(y_minlim[0][i], y_minlim[1][i]), ymax=max(y_maxlim[0][i], y_maxlim[1][i]))
        plt.xlim(xmin=0)
        plt.xlabel(xlabels[i])
        plt.ylabel(ylabels[i])
        # if metric == 'Local Clustering Coefficient':
        #     plt.legend(handles=[_1, _2, _3, _4], loc=7)
        # elif metric == 'Average Degree':
        #     plt.legend(handles=[_1, _3, _2], loc=4)

        # else:
        #     plt.legend(handles=[_1, _2, _3, _4], loc='best')
        plt.savefig('./plots/aa.png', dpi=800)
        plt.show()
        print(popt1)
        print(popt2)
        i += 1

def avg_metrics(models):
    graph_models = {}

    for model_name in models.keys():
        graph_model_name, k = model_name.split('-')
        k = int(k)

        if graph_model_name not in graph_models:
            graph_models[graph_model_name] = {
                metric:{'x':[], 'y':[]} 
                for metric in list(models[model_name].keys())
            }

        for metric in models[model_name].keys():
            if metric == 'Variance':
                for i in range(len(models[model_name][metric])):
                    models[model_name][metric][i] = models[model_name][metric][i] ** 2

            graph_models[graph_model_name][metric]['x'].append(k)            
            graph_models[graph_model_name][metric]['y'].append(
                np.average(models[model_name][metric])
            )

    return graph_models

def avg_dist_samples(models):
    models_per_n = {}
    group_samples = {}

    for sample_name in models.keys():
        model_name = sample_name.rsplit('-', 1)[0]

        if model_name not in group_samples:
            group_samples[model_name] = []

        group_samples[model_name].append(models[sample_name])

    for model_name in group_samples.keys():
        models_per_n[model_name] = avg_dists(
            group_samples[model_name], 
            DC
        )

    return models_per_n

def avg_dists(sample_models, dc_dist):
    n_model = {}
    max_val = -1

    for sample_model in sample_models:
        aux_val = np.max(sample_model[dc_dist])

        if aux_val > max_val:
            max_val = aux_val

    n_model['#'] = np.zeros(max_val + 1)

    for sample_model in sample_models:
        np.add.at(n_model['#'], sample_model[dc_dist], 1)

        for i in range(max_val + 1):
            k_val_idxs = np.where(sample_model[dc_dist] == i)[0]
            
            if k_val_idxs.size == 0:
                continue

            for dist in sample_model.keys():
                if dist == dc_dist:
                    continue
                elif dist not in n_model:
                    n_model[dist] = np.zeros(max_val + 1)

                n_model[dist][i] += np.sum(sample_model[dist][k_val_idxs])

    for dist in n_model.keys():
        if dist == '#':
            continue

        n_model[dist] = np.divide(
            n_model[dist], 
            n_model['#'], 
            out=np.zeros_like(n_model[dist]),
            where= n_model['#'] != 0
        )
    return n_model

def get_dist_metrics(f_names, metric_idxs):
    models = {}

    for f_name in f_names:
        f = open(f_name, 'r')
        metrics = get_metric_names(f.readline(), metric_idxs)
        models.update(dist_samples(f, metrics, metric_idxs))

    return models

def dist_samples(f, metric_names, metric_idxs):
    models = {}
    line = f.readline()
    prev_model = None
    file_line_idx = 1

    while line:
        model, values = line.split(',', 1)
        model = model.rsplit('-', 1)[0]

        if prev_model is None:
            prev_model = model
        elif prev_model != model:
            prev_model = model
            file_line_idx = 1

        if model not in models:
            models[model] = {}

        try:
            metric_idx = metric_idxs.index(file_line_idx)
            models[model][metric_names[metric_idx]] = np.array(
                [float(val) for val in values.split(',') if is_float(val)],
                dtype=np.float if metric_names[metric_idx] != DC else np.int
            )
        except ValueError:
            pass

        line = f.readline()
        file_line_idx += 1

    return models

def is_float(val):
    try:
        float(val)
        return True
    except ValueError:
        return False

def get_metrics_by_graph_and_n(f_name, cols):
    f = open(f_name, 'r')
    col_names = get_metric_names(f.readline(), cols)
    models = {}
    cont = True

    while cont:
        models_aux, cont = avg_samples(f, col_names, cols)
        models.update(models_aux)

    return models
        
def get_metric_names(header, cols):
    metrics = []
    header_vals = header.split(',')

    for i in cols:
        metrics.append(header_vals[i].strip())
    
    return metrics

def avg_samples(f, col_names, col_idxs):
    models = {}
    line = f.readline()
    first_model = None
    n = None
    pointer = None

    while line:
        cells = line.split(',')
        model, new_n = cells[0].split('-')[0:2]
        model = model + '-' + new_n

        if n == None:
            n = new_n
        elif new_n != n:
            f.seek(pointer)
            break

        if model not in models:
            models[model] = {name:[] for name in col_names}

        for i in range(len(col_idxs)):
            models[model][col_names[i]].append(float(cells[col_idxs[i]]))

        pointer = f.tell()
        line = f.readline()

    return models, line

if __name__ == '__main__':
    graph_metrics_to_plots_both_models()
