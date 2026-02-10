import torch
import numpy as np

def calculate_per_class_metrics(predictions, labels, num_classes=3):
    """
    计算每个类别的精确率(Precision)、召回率(Recall)和F1分数(F1-Score)

    Args:
        predictions: 模型输出的预测类别张量，形状为 [batch, patch_num, sensor_num]
        labels:      真实的标签张量，形状与 predictions 相同
        num_classes: 类别数量，默认为3

    Returns:
        class_metrics: 一个字典，包含每个类别的TP, FP, FN, Precision, Recall, F1
        global_metrics: 一个字典，包含宏平均、微平均、加权平均的 Precision, Recall, F1
    """
    # 确保输入在CPU上并转换为numpy数组以便计算
    pred_flat = predictions.cpu().numpy().flatten()
    label_flat = labels.cpu().numpy().flatten()

    # 初始化每个类别的TP, FP, FN
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for class_id in range(num_classes):
        # 对于当前类别class_id，计算TP, FP, FN
        tp[class_id] = np.sum((pred_flat == class_id) & (label_flat == class_id))
        fp[class_id] = np.sum((pred_flat == class_id) & (label_flat != class_id))
        fn[class_id] = np.sum((pred_flat != class_id) & (label_flat == class_id))

    # 计算每个类别的Precision, Recall, F1
    precision_per_class = np.zeros(num_classes)
    recall_per_class = np.zeros(num_classes)
    f1_per_class = np.zeros(num_classes)

    for i in range(num_classes):
        # 避免除零，如果TP+FP==0，则Precision设为0或NaN（根据需求）
        if tp[i] + fp[i] > 0:
            precision_per_class[i] = tp[i] / (tp[i] + fp[i])
        else:
            precision_per_class[i] = 0.0  # 或 np.nan

        if tp[i] + fn[i] > 0:
            recall_per_class[i] = tp[i] / (tp[i] + fn[i])
        else:
            recall_per_class[i] = 0.0  # 或 np.nan

        if precision_per_class[i] + recall_per_class[i] > 0:
            f1_per_class[i] = 2 * (precision_per_class[i] * recall_per_class[i]) / (
                        precision_per_class[i] + recall_per_class[i])
        else:
            f1_per_class[i] = 0.0  # 或 np.nan

    # 组织每个类别的指标返回结果
    class_metrics = {}
    for i in range(num_classes):
        class_metrics[f'Class_{i}'] = {
            'TP': tp[i],
            'FP': fp[i],
            'FN': fn[i],
            'Precision': precision_per_class[i],
            'Recall': recall_per_class[i],
            'F1': f1_per_class[i]
        }

    return class_metrics