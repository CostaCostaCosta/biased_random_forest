import seaborn as sns
import matplotlib.pyplot as plt


def save_prc_curve(recall, precision, name):
    plt.figure(figsize=(4, 4))
    ax = sns.lineplot(recall, precision)
    ax.set_title('PRC Curve ' + name)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')

    output_name = 'prc' + name
    plt.savefig(output_name)
    plt.close()
    return None


def save_roc_curve(fp_rate, tp_rate, name):
    plt.figure(figsize=(4, 4))
    ax = sns.lineplot(fp_rate, tp_rate)
    ax.set_title('ROC Curve ' + name)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')

    output_name = 'roc' + name
    plt.savefig(output_name)
    plt.close()
    return None
