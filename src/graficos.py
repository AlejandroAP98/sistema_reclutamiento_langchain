import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
from math import pi

from src.config import GRAFICOS_DIR


def generar_graficos_knn(modelo, X_train, y_train, X_test, y_test, features,
                         top_df, accs=None, precs=None, recs=None, f1s=None, cv_folds=5):
    GRAFICOS_DIR.mkdir(exist_ok=True)
    y_pred = modelo.predict(X_test)
    y_prob = modelo.predict_proba(X_test)[:, 1]

    sns.set_theme(style="darkgrid", palette="muted")
    plt.rcParams.update({"figure.dpi": 120, "savefig.dpi": 150})

    _confusion_matrix(y_test, y_pred)
    _metricas_cv_barras(accs, precs, recs, f1s, cv_folds)
    _curva_roc(y_test, y_prob)
    _top_scores(top_df)

    print(f"\nGraficos generados en: {GRAFICOS_DIR.resolve()}")


def _confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["No Apto", "Apto"],
                yticklabels=["No Apto", "Apto"])
    ax.set_title("Matriz de Confusion - KNN (full dataset)", fontsize=12, fontweight="bold")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Real")
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "confusion_matrix.png")
    plt.close(fig)


def _metricas_cv_barras(accs, precs, recs, f1s, folds):
    if not all([accs, precs, recs, f1s]):
        return

    nombres = ["Accuracy", "Precision", "Recall", "F1-Score"]
    medias = [np.mean(accs), np.mean(precs), np.mean(recs), np.mean(f1s)]
    stds = [np.std(accs), np.std(precs), np.std(recs), np.std(f1s)]
    colores = ["#6366f1", "#a855f7", "#10b981", "#f59e0b"]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(nombres))
    barras = ax.bar(x, medias, yerr=stds, capsize=6, color=colores,
                    edgecolor="white", linewidth=0.8, error_kw={"linewidth": 1.5})

    for i, (media, std) in enumerate(zip(medias, stds)):
        ax.text(i, media + std + 1.5, f"{media:.1f}%", ha="center",
                fontsize=11, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(nombres)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Valor (%)")
    ax.set_title(f"Metricas KNN - CV {folds}-fold (+/- 1 std)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "metricas_knn.png")
    plt.close(fig)


def _curva_roc(y_test, y_prob):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="#6366f1", lw=2.5, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", lw=1.5)
    ax.fill_between(fpr, tpr, alpha=0.15, color="#6366f1")
    ax.set_xlabel("Tasa de Falsos Positivos (FPR)", fontsize=10)
    ax.set_ylabel("Tasa de Verdaderos Positivos (TPR)", fontsize=10)
    ax.set_title("Curva ROC - KNN (full dataset)", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "curva_roc.png")
    plt.close(fig)


def _top_scores(top_df):
    fig, ax = plt.subplots(figsize=(7, 3))
    nombres = top_df["nombre"].tolist()
    scores = top_df["score_final"].tolist()

    max_s = max(scores) if scores else 1
    colores = ["#6366f1" if s == max_s else "#a5b4fc" for s in scores]
    barras = ax.barh(nombres, scores, color=colores, edgecolor="white")

    for bar, s in zip(barras, scores):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2,
                f"{s:.1f}", va="center", fontsize=10, fontweight="bold")

    ax.set_xlabel("Score Final (%)")
    ax.set_title("Top Candidatos - Scoring KNN", fontsize=12, fontweight="bold")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(GRAFICOS_DIR / "top_scores.png")
    plt.close(fig)
