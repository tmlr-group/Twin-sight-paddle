import numpy as np
import paddle
import paddle.nn.functional as F


def inference_by_SSFL_model(SSFL_model, dataloader, device):
    feature_vector = []
    labels_vector = []
    SSFL_model.to(device)
    SSFL_model.eval()
    for step, (x, y) in enumerate(dataloader):
        x = x.to(device)

        # get encoding
        with paddle.no_grad():
            h = SSFL_model(x)

        h = h.squeeze()
        h = h.detach()

        feature_vector.extend(h.cpu().detach().numpy())
        labels_vector.extend(y.numpy())

    feature_vector = np.array(feature_vector)
    labels_vector = np.array(labels_vector)
    return feature_vector, labels_vector





def info_nce_loss( features, batch_size, device, n_views=2, temperature=0.07):
    labels = paddle.concat([paddle.arange(batch_size) for i in range(n_views)], axis=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).astype('float')
    labels = labels.to(device)

    features = F.normalize(features, axis=1)

    similarity_matrix = paddle.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     n_views * self.conf.batch_size, n_views * self.conf.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
    mask = paddle.cast(paddle.eye(labels.shape[0]), 'bool').to(device)
    labels = labels[~mask].reshape([labels.shape[0], -1])
    similarity_matrix = similarity_matrix[~mask].reshape([similarity_matrix.shape[0], -1])
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
    positives = similarity_matrix[paddle.cast(labels, 'bool')].reshape([labels.shape[0], -1])

        # select only the negatives the negatives
    negatives = similarity_matrix[~paddle.cast(labels, 'bool')].reshape([similarity_matrix.shape[0], -1])

    logits = paddle.concat([positives, negatives], axis=1)
    labels = paddle.zeros(logits.shape[0], dtype='int64').to(device)

    logits = paddle.cast(logits / temperature, dtype='float64')
    return logits, labels