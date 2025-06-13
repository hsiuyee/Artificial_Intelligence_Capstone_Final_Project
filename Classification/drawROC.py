import matplotlib
import matplotlib.pyplot as plt
from sklearn.metrics import auc

matplotlib.use('Agg')

def read_roc_data(filename):
    roc_curves = []
    with open(filename, 'r') as file:
        for i in range(4):
            line = file.readline()
            line = line[:-2].split(', ')
            fpr = [float(x) for x in line]
            line = file.readline()
            line = line[:-2].split(', ')
            tpr = [float(x) for x in line]
            roc_curves.append((fpr, tpr))
    return roc_curves

def plot_roc_curves(roc_curves):
    plt.figure(figsize=(8, 6))
    legend = [None, 'Alex_Oneshot', 'Res_Oneshot', 'Alex_Fiveshot', 'Res_Fiveshot']
    for i, (fpr, tpr) in enumerate(roc_curves, start=1):
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label='%s (area = %0.2f)' % (legend[i], roc_auc))
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./Classification/fewshot_ROC.png')

if __name__ == '__main__':
    roc_curves = read_roc_data('./Classification/roc_data.txt')
    plot_roc_curves(roc_curves)
