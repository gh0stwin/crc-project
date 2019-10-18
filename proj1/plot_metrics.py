import matplotlib.pyplot as plt
import numpy as np


def plot_avg_metrics(models):
    for model_name in models.keys():
        for metric in models[model_name].keys():
            plt.plot(models[model_name][metric]['x'], models[model_name][metric]['y'], '.')
            title = model_name.upper() + ' - ' + metric
            plt.title(title)
            plt.savefig('./plots/' + title + '.png')
            plt.show()

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
            graph_models[graph_model_name][metric]['x'].append(k)            
            graph_models[graph_model_name][metric]['y'].append(
                np.average(models[model_name][metric])
            )

    return graph_models

def get_dist_metrics(f_name, metric_idxs):
    f = open(f_name, 'r')
    metrics = get_metric_names(f.readline(), metric_idxs)
    models = {}
    cont = True

    while cont:
        models_aux, cont = dist_samples(f, 8, metrics, metric_idxs)
        models.update(models_aux)

    return models

def dist_samples(f, n_metrics, metric_names, metric_idxs):
    models = {}
    line = f.readline()
    first_model = None
    n = None
    metric_idx = 0
    pointer = None

    while line:
        cells = line.split(',', 1)
        model, new_n = cells[0].split('-')[0:2]
        model = model + '-' + new_n

        if n == None:
            n = new_n
        elif new_n != n:
            f.seek(pointer)
            break

        if model not in models:
            models[model] = {name:[] for name in metric_names}

        idx = None
        try:
            idx = metric_idxs.index(metric_idx)
        except ValueError:
            pass

        if idx is not None:
            models[model][metric_names[idx]] = np.array(cells[1].split(','))

        pointer = f.tell()
        line = f.readline()
        metric_idx += 1

        if metric_idx == n_metrics - 1:
            metric_idx = 0

    return models, line

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

# variance, degree dist, avg degree, avg gb cc, avg lcl cc, assortativity
