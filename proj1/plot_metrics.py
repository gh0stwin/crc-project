import matplotlib.pyplot as plt
import numpy as np


DC = 'Degree Centrality'

def plot_dists(models, title='a'):
    for model_name in models.keys():
        for metric in models.keys():
            if metric == DC:
                continue

            plt.plot(models[model_name][DC], models[model_name][metric], '.')
            plt.title(title)
            
        plt.savefig('./plots/' + title + '.png')
        plt.show()

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
    # variance, avg degree, avg gb cc, avg lcl cc, assortativity
    models = get_metrics_by_graph_and_n(
        './results/metrics.out', 
        [4, 6, 9, 10, 18]
    )

    #degree dist
