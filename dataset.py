import torch
import math
import numpy as np

def get_dataset(seed=None):
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    def get_toy_dataset(n_samples, anomaly_centers=False):
        z = torch.randn(n_samples, 2)  # 2d
        scale_centers = 4
        sq2 = 1 / math.sqrt(2)
        centers = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        if anomaly_centers:
            centers = [(0.5, 0.5), (-0.5, 0.5), (0.5, -0.5), (-0.5, -0.5), (0, 0)]
            centers += [(1.7, 0), (-1.7, 0), (0, 1.7), (0, -1.7)]
        centers = torch.tensor([(scale_centers * x, scale_centers * y) for x, y in centers])
        return sq2 * (0.3 * z + centers[torch.randint(len(centers), size=(n_samples,))])

    N = 10000

    data_train_id = get_toy_dataset(N)
    data_test_id = get_toy_dataset(N // 2)
    data_test_ood = get_toy_dataset(N // 2, anomaly_centers=True)

    data_test = np.vstack([data_test_id, data_test_ood])
    labels_test = np.hstack([np.zeros(len(data_test_id)), np.ones(len(data_test_ood))])
    labels_test = labels_test.astype(np.uint8)

    id_to_type = {
        0: "normal",
        1: "anomaly"
    }

    return data_train_id, np.zeros(len(data_train_id)).astype(np.uint8), data_test, labels_test, id_to_type


def create_meshgrid_from_data(data, n_points=100, meshgrid_offset=1):
    x_min, x_max = data[:, 0].min() - meshgrid_offset, data[:, 0].max() + meshgrid_offset
    y_min, y_max = data[:, 1].min() - meshgrid_offset, data[:, 1].max() + meshgrid_offset
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, n_points), np.linspace(y_min, y_max, n_points))
    return xx, yy