#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Análise de Complexidade-Entropia para Detecção de Imagens Sintéticas - Versão Com Foco em Bordas

Objetivo:
    Implementamos extração de features baseadas em bordas e entropia ordinal
    (Complexidade-Entropia) para diferenciar imagens reais de sintéticas.
    O foco principal são representações de borda/gradiente e séries
    extraídas de contornos, linhas/colunas e patches de alta transição.

Principais capacidades:
    - Extração multi-escala de entropia (H) e complexidade (C) via ordpy.
    - Versão alternativa H_bits = H * log2(d!) para interpretação em bits.
    - Múltiplas representações de borda (Canny, Sobel, Laplacian, Scharr).
    - Séries orientadas por bordas (contornos, linhas de alto gradiente, patches).
    - Treino e avaliação com RandomForest; geração de visualizações (plano CH, ROC).

    Extraímos ~70-120 séries por imagem
    Calculamos (H, C) para cada série em 4 escalas → ~280 pares (H, C)
    Agregamos esses 280 valores em 112 estatísticas (média, std, quartis, etc.)
    Treinamos RF com 112 features (não apenas 2!)
    Visualizamos apenas H_mean vs C_mean no Plano CH (2D)

Autor:
    Fábio Linhares — Universidade Federal de Alagoas (UFAL)
"""

import os
import sys
import math
import numpy as np
import pandas as pd

# Configurar matplotlib para usar backend sem interface gráfica (evita erro Qt/Wayland)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
import cv2
from PIL import Image
import ordpy
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from sklearn.preprocessing import StandardScaler
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Configuração visual
plt.style.use('default')
sns.set_palette("husl")

class EdgeFocusedDeepfakeAnalyzer:
    """Analisa imagens focando em bordas para detectar imagens sintéticas.

    Parâmetros principais:
        embedding_dim (int): dimensão base para ordinal patterns.
        delay (int): defasagem temporal entre elementos do padrão ordinal.
        multi_scale (bool): se True, usa múltiplos valores de embedding_dim.
        verbose (bool): controla mensagens de progresso.

    Comportamento:
        - Extrai mapas de borda e séries orientadas por bordas.
        - Calcula H (normalizado pelo ordpy), C e H_bits.
        - Agrega estatísticas das séries e metadados das bordas.
    """
    def __init__(self, embedding_dim=5, delay=1, multi_scale=True, verbose=True):
        self.embedding_dim = embedding_dim
        self.delay = delay
        self.multi_scale = multi_scale
        self.embedding_dims = [3, 4, 5, 6] if multi_scale else [embedding_dim]
        self.features_df = None
        self.model_rf = None
        self.model_knn = None
        self.scaler = StandardScaler()
        self.verbose = verbose
        if verbose:
            print("Configuração:")
            print(f"   - Dimensões de embedding: {self.embedding_dims}")
            print(f"   - Delay: {self.delay}")
            print(f"   - Multi-escala: {multi_scale}")

    def load_and_preprocess_image(self, image_path):
        """Carrega e pré-processa imagem (grayscale, 256x256, [0,1])."""
        try:
            p = str(image_path)
            if not p.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                return None
            image = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
            if image is None:
                try:
                    image = np.array(Image.open(image_path).convert('L'))
                except Exception:
                    return None
            if image is None or image.size == 0:
                return None
            image = cv2.resize(image, (256, 256))
            image = image.astype(np.float64) / 255.0
            return image
        except Exception:
            return None

    def extract_edge_regions(self, image):
        """Extrai múltiplas representações de bordas."""
        if image is None:
            return None
        edge_info = {}
        img_uint8 = (image * 255).astype(np.uint8)

        # 1) Canny
        edges_canny = cv2.Canny(img_uint8, 50, 150)
        edge_info['canny'] = edges_canny.astype(np.float64) / 255.0

        # 2) Sobel
        sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        edge_info['sobel_magnitude'] = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_info['sobel_direction'] = np.arctan2(sobel_y, sobel_x)

        # 3) Laplacian
        lap = cv2.Laplacian(image, cv2.CV_64F, ksize=3)
        edge_info['laplacian'] = np.abs(lap)

        # 4) Scharr
        scharr_x = cv2.Scharr(image, cv2.CV_64F, 1, 0)
        scharr_y = cv2.Scharr(image, cv2.CV_64F, 0, 1)
        edge_info['scharr_magnitude'] = np.sqrt(scharr_x**2 + scharr_y**2)

        # 5) Alta frequência
        blurred = cv2.GaussianBlur(image, (5, 5), 0)
        edge_info['high_frequency'] = np.abs(image - blurred)

        # 6) Transições de iluminação
        grad_mag = np.sqrt(
            np.gradient(image, axis=0)**2 + np.gradient(image, axis=1)**2
        )
        edge_info['illumination_transitions'] = grad_mag

        return edge_info

    def extract_edge_focused_series(self, image, edge_info):
        """Extrai séries temporais focadas em bordas/gradientes."""
        if image is None or edge_info is None:
            return []
        series_list = []
        h, w = image.shape

        # 1) Contornos do Canny
        contours, _ = cv2.findContours(
            (edge_info['canny'] * 255).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
        )
        need_len = max(self.embedding_dims) + (max(self.embedding_dims) - 1) * self.delay
        for contour in contours[:20]:
            if len(contour) >= need_len:
                vals = []
                for pt in contour:
                    x, y = pt[0]
                    if 0 <= y < h and 0 <= x < w:
                        vals.append(image[y, x])
                if len(vals) >= need_len:
                    series_list.append(np.array(vals))

        # 2) Regiões de alto gradiente (Sobel)
        sobel_mag = edge_info['sobel_magnitude']
        thr = np.percentile(sobel_mag, 75)
        high_mask = sobel_mag > thr
        step_row = max(1, h // 15)
        step_col = max(1, w // 15)
        for i in range(0, h, step_row):
            if high_mask[i, :].sum() > w * 0.1:
                s = image[i, :].ravel()
                if len(s) >= need_len:
                    series_list.append(s)
        for j in range(0, w, step_col):
            if high_mask[:, j].sum() > h * 0.1:
                s = image[:, j].ravel()
                if len(s) >= need_len:
                    series_list.append(s)

        # 3) Patches em transições de iluminação
        illum = edge_info['illumination_transitions']
        thr_illum = np.percentile(illum, 70)
        patch, stride = 16, 12
        for i in range(0, h - patch + 1, stride):
            for j in range(0, w - patch + 1, stride):
                p_illum = illum[i:i+patch, j:j+patch]
                if p_illum.mean() > thr_illum:
                    p_img = image[i:i+patch, j:j+patch].ravel()
                    if len(p_img) >= need_len:
                        series_list.append(p_img)

        # 4) Alta frequência
        hf = edge_info['high_frequency']
        thr_hf = np.percentile(hf, 75)
        step_hf = max(1, h // 12)
        for i in range(0, h, step_hf):
            if hf[i, :].mean() > thr_hf * 0.5:
                s = hf[i, :].ravel()
                if len(s) >= need_len:
                    series_list.append(s)

        # 5) Direção do gradiente
        sobel_dir = edge_info['sobel_direction']
        step_dir = max(1, h // 10)
        for i in range(0, h, step_dir):
            if high_mask[i, :].sum() > w * 0.15:
                s = sobel_dir[i, :].ravel()
                if len(s) >= need_len:
                    series_list.append(s)

        return series_list

    def _extract_traditional_series(self, image):
        """Linhas/colunas para baseline."""
        series = []
        h, w = image.shape
        for i in range(0, h, max(1, h // 8)):
            series.append(image[i, :].ravel())
        for j in range(0, w, max(1, w // 8)):
            series.append(image[:, j].ravel())
        return series

    def _compute_edge_statistics(self, edge_info):
        """Estatísticas globais das representações de borda."""
        stats = {}
        for key, edge_map in edge_info.items():
            stats[f'edge_{key}_mean'] = float(np.mean(edge_map))
            stats[f'edge_{key}_std']  = float(np.std(edge_map))
            stats[f'edge_{key}_max']  = float(np.max(edge_map))
            hist, _ = np.histogram(edge_map.ravel(), bins=50, density=True)
            hist = hist[hist > 0]
            stats[f'edge_{key}_entropy'] = float(-np.sum(hist * np.log2(hist + 1e-10)))
        return stats

    def _get_default_features(self):
        """Features nulas quando falha a extração."""
        feats = {}
        for dx in self.embedding_dims:
            p = f'd{dx}_'
            for metric in ['H', 'Hbits', 'C']:
                for stat in ['mean', 'std', 'max', 'min', 'q25', 'q75', 'skew', 'kurt']:
                    feats[f'{p}{metric}_{stat}'] = 0.0
            feats[f'{p}HC_corr'] = 0.0
            feats[f'{p}CH_dist'] = 0.0
            feats[f'{p}n_series'] = 0
        return feats

    def compute_ordinal_features_corrected(self, image):
        """Calcula features ordinais H, H_bits e C a partir de séries focadas em borda.

        Observações rápidas:
            - H é obtido normalizado pelo ordpy (já no intervalo [0,1]).
            - H_bits = H * log2(d!) fornece interpretação em bits para cada dx.
            - Séries com variância muito baixa são descartadas.
        """
        edge_info = self.extract_edge_regions(image)
        if edge_info is None:
            return self._get_default_features()

        edge_series = self.extract_edge_focused_series(image, edge_info)
        all_series = edge_series + self._extract_traditional_series(image)
        if not all_series:
            return self._get_default_features()

        measures = {dx: {'H': [], 'C': [], 'H_bits': []} for dx in self.embedding_dims}

        for ts in all_series:
            if np.std(ts) < 1e-10:
                continue
            tsn = (ts - np.mean(ts)) / np.std(ts)
            for dx in self.embedding_dims:
                min_len = dx + (dx - 1) * self.delay
                if len(tsn) < min_len:
                    continue
                H_norm, C = ordpy.complexity_entropy(tsn, dx=dx, taux=self.delay)
                if any(map(lambda v: np.isnan(v) or np.isinf(v), [H_norm, C])):
                    continue
                measures[dx]['H'].append(H_norm)
                measures[dx]['C'].append(C)
                max_bits = math.log2(math.factorial(dx))
                measures[dx]['H_bits'].append(H_norm * max_bits)

        # Agrega estatísticas
        feats = {}
        for dx in self.embedding_dims:
            H_list = measures[dx]['H']
            C_list = measures[dx]['C']
            Hb_list = measures[dx]['H_bits']
            if not H_list:
                continue
            p = f'd{dx}_'
            # H normalizado
            feats[f'{p}H_mean'] = float(np.mean(H_list))
            feats[f'{p}H_std']  = float(np.std(H_list))
            feats[f'{p}H_max']  = float(np.max(H_list))
            feats[f'{p}H_min']  = float(np.min(H_list))
            feats[f'{p}H_q25']  = float(np.percentile(H_list, 25))
            feats[f'{p}H_q75']  = float(np.percentile(H_list, 75))
            feats[f'{p}H_skew'] = float(skew(H_list))
            feats[f'{p}H_kurt'] = float(kurtosis(H_list))
            # H em bits
            feats[f'{p}Hbits_mean'] = float(np.mean(Hb_list))
            feats[f'{p}Hbits_std']  = float(np.std(Hb_list))
            # C
            feats[f'{p}C_mean'] = float(np.mean(C_list))
            feats[f'{p}C_std']  = float(np.std(C_list))
            feats[f'{p}C_max']  = float(np.max(C_list))
            feats[f'{p}C_min']  = float(np.min(C_list))
            feats[f'{p}C_q25']  = float(np.percentile(C_list, 25))
            feats[f'{p}C_q75']  = float(np.percentile(C_list, 75))
            feats[f'{p}C_skew'] = float(skew(C_list))
            feats[f'{p}C_kurt'] = float(kurtosis(C_list))
            # Relação H–C e distância do centróide
            feats[f'{p}HC_corr'] = float(np.corrcoef(H_list, C_list)[0, 1]) if len(H_list) > 1 else 0.0
            feats[f'{p}CH_dist'] = float(np.sqrt(np.mean(H_list)**2 + np.mean(C_list)**2))
            feats[f'{p}n_series'] = int(len(H_list))

        # Estatísticas globais das bordas
        feats.update(self._compute_edge_statistics(edge_info))

        # Limpeza numérica
        for k, v in list(feats.items()):
            if np.isnan(v) or np.isinf(v):
                feats[k] = 0.0
        return feats

    def process_dataset(self, dataset_path, label, max_samples=200):
        """Processa um dataset e retorna DataFrame de features."""
        features_list = []
        dataset_path = Path(dataset_path)
        exts = ['.jpg', '.jpeg', '.png', '.bmp']
        files = []
        for e in exts:
            files += list(dataset_path.glob(f'**/*{e}'))
            files += list(dataset_path.glob(f'**/*{e.upper()}'))

        if self.verbose:
            print(f"📂 {label}: {len(files)} arquivos | processando {min(len(files), max_samples)}")

        if len(files) > max_samples:
            np.random.seed(42)
            files = np.random.choice(files, max_samples, replace=False)
        else:
            files = files[:max_samples]

        ok = 0
        for i, path in enumerate(files):
            if self.verbose and i and i % 50 == 0:
                print(f"   ⏳ {i}/{len(files)} (válidas: {ok})")
            img = self.load_and_preprocess_image(path)
            if img is None:
                continue
            feats = self.compute_ordinal_features_corrected(img)
            if any(v != 0 for k, v in feats.items() if k.endswith('n_series')):
                feats['label'] = label
                feats['filename'] = Path(path).name
                features_list.append(feats)
                ok += 1

        if self.verbose:
            print(f"Concluído: {ok}/{len(files)} válidas\n")
        return pd.DataFrame(features_list)

    def train_and_evaluate(self, test_size=0.25, optimize=False):
        """Treina RandomForest e KNN (com opção de GridSearch) e avalia ambos."""
        if self.features_df is None or len(self.features_df) < 40:
            print("Dados insuficientes para treino.")
            return None

        print("🤖 Treinando classificadores (RF e KNN)...")
        feats = [c for c in self.features_df.columns if c not in ['label', 'filename', 'dataset']]
        X = self.features_df[feats]
        y = self.features_df['label']
        X = X.loc[:, X.var() != 0]
        if X.shape[1] == 0:
            print("Nenhuma feature com variância > 0.")
            return None

        print(f"Features utilizadas: {X.shape[1]}")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        self.scaler.fit(X_train)
        X_train_s = self.scaler.transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        # ========== RANDOM FOREST ==========
        print("\n" + "="*60)
        print("RANDOM FOREST")
        print("="*60)
        
        if optimize:
            print("GridSearchCV (RF)...")
            grid_rf = GridSearchCV(
                RandomForestClassifier(random_state=42),
                {
                    'n_estimators': [100, 200, 300],
                    'max_depth': [10, 20, None],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2]
                },
                cv=5, scoring='accuracy', n_jobs=-1, verbose=0
            )
            grid_rf.fit(X_train_s, y_train)
            self.model_rf = grid_rf.best_estimator_
            print(f"Best params RF: {grid_rf.best_params_}")
        else:
            self.model_rf = RandomForestClassifier(
                n_estimators=200, max_depth=20, random_state=42, n_jobs=-1
            )
            self.model_rf.fit(X_train_s, y_train)

        # Métricas RF
        tr_acc_rf = self.model_rf.score(X_train_s, y_train)
        te_acc_rf = self.model_rf.score(X_test_s,  y_test)
        print(f"Acurácia RF: Treino={tr_acc_rf:.4f} ({tr_acc_rf*100:.1f}%) | Teste={te_acc_rf:.4f} ({te_acc_rf*100:.1f}%)")

        cv_scores_rf = cross_val_score(self.model_rf, X_train_s, y_train, cv=5)
        print(f"CV 5-fold RF: {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")

        # Importância RF
        imp_rf = pd.DataFrame({
            'feature': X_train.columns,
            'importance': self.model_rf.feature_importances_
        }).sort_values('importance', ascending=False)
        print("\nTop 15 features (RF):")
        print(imp_rf.head(15).to_string(index=False))

        # Confusion matrix RF
        y_pred_rf = self.model_rf.predict(X_test_s)
        cm_rf = confusion_matrix(y_test, y_pred_rf, labels=['fake','real'])
        print("\nMatriz de Confusão RF (linhas=verdadeiro, cols=predito):")
        print("                Pred Fake  Pred Real")
        print(f"True Fake:      {cm_rf[0,0]:9d} {cm_rf[0,1]:10d}")
        print(f"True Real:      {cm_rf[1,0]:9d} {cm_rf[1,1]:10d}")

        print("\nRelatório de Classificação RF:")
        print(classification_report(y_test, y_pred_rf, labels=['fake','real'], target_names=['fake','real']))

        # ========== K-NEAREST NEIGHBORS ==========
        print("\n" + "="*60)
        print("K-NEAREST NEIGHBORS")
        print("="*60)
        
        if optimize:
            print("GridSearchCV (KNN)...")
            grid_knn = GridSearchCV(
                KNeighborsClassifier(),
                {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'metric': ['euclidean', 'manhattan', 'minkowski']
                },
                cv=5, scoring='accuracy', n_jobs=-1, verbose=0
            )
            grid_knn.fit(X_train_s, y_train)
            self.model_knn = grid_knn.best_estimator_
            print(f"Best params KNN: {grid_knn.best_params_}")
        else:
            self.model_knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
            self.model_knn.fit(X_train_s, y_train)

        # Métricas KNN
        tr_acc_knn = self.model_knn.score(X_train_s, y_train)
        te_acc_knn = self.model_knn.score(X_test_s,  y_test)
        print(f"Acurácia KNN: Treino={tr_acc_knn:.4f} ({tr_acc_knn*100:.1f}%) | Teste={te_acc_knn:.4f} ({te_acc_knn*100:.1f}%)")

        cv_scores_knn = cross_val_score(self.model_knn, X_train_s, y_train, cv=5)
        print(f"CV 5-fold KNN: {cv_scores_knn.mean():.4f} ± {cv_scores_knn.std():.4f}")

        # Confusion matrix KNN
        y_pred_knn = self.model_knn.predict(X_test_s)
        cm_knn = confusion_matrix(y_test, y_pred_knn, labels=['fake','real'])
        print("\nMatriz de Confusão KNN (linhas=verdadeiro, cols=predito):")
        print("                Pred Fake  Pred Real")
        print(f"True Fake:      {cm_knn[0,0]:9d} {cm_knn[0,1]:10d}")
        print(f"True Real:      {cm_knn[1,0]:9d} {cm_knn[1,1]:10d}")

        print("\nRelatório de Classificação KNN:")
        print(classification_report(y_test, y_pred_knn, labels=['fake','real'], target_names=['fake','real']))

        # ========== COMPARAÇÃO ==========
        print("\n" + "="*60)
        print("COMPARAÇÃO DE MODELOS")
        print("="*60)
        print(f"{'Modelo':<20} {'Acurácia Treino':<20} {'Acurácia Teste':<20} {'CV Mean ± Std'}")
        print("-"*80)
        print(f"{'Random Forest':<20} {tr_acc_rf:.4f} ({tr_acc_rf*100:.1f}%){'':<6} {te_acc_rf:.4f} ({te_acc_rf*100:.1f}%){'':<6} {cv_scores_rf.mean():.4f} ± {cv_scores_rf.std():.4f}")
        print(f"{'KNN':<20} {tr_acc_knn:.4f} ({tr_acc_knn*100:.1f}%){'':<6} {te_acc_knn:.4f} ({te_acc_knn*100:.1f}%){'':<6} {cv_scores_knn.mean():.4f} ± {cv_scores_knn.std():.4f}")

        return {
            'X_train': X_train, 'X_test': X_test,
            'y_train': y_train, 'y_test': y_test,
            'y_pred_rf': y_pred_rf,
            'y_pred_knn': y_pred_knn,
            'feature_importance': imp_rf,
            'X_train_scaled': X_train_s,
            'X_test_scaled': X_test_s,
            'cm_rf': cm_rf,
            'cm_knn': cm_knn
        }

    def create_visualizations(self, save_dir, eval_results=None):
        """Gera visualizações: planos CH multi-escala e curvas ROC comparativas (RF vs KNN)."""
        if self.features_df is None or len(self.features_df) == 0:
            print("Sem dados para visualizar.")
            return

        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)

        real = self.features_df[self.features_df['label'] == 'real']
        fake = self.features_df[self.features_df['label'] == 'fake']

        # Plano CH
        valid_dims = [dx for dx in self.embedding_dims if f'd{dx}_H_mean' in self.features_df.columns]
        if valid_dims:
            fig, axes = plt.subplots(1, len(valid_dims), figsize=(6*len(valid_dims), 5))
            if len(valid_dims) == 1:
                axes = [axes]
            for ax, dx in zip(axes, valid_dims):
                h_col, c_col = f'd{dx}_H_mean', f'd{dx}_C_mean'
                ax.scatter(real[h_col], real[c_col], s=50, alpha=0.6,
                           label='Real', color='blue', edgecolors='darkblue')
                ax.scatter(fake[h_col], fake[c_col], s=50, alpha=0.6,
                           label='Fake', color='red', edgecolors='darkred')
                ax.set_xlabel(f'Entropia H (d={dx}) [0,1]')
                ax.set_ylabel(f'Complexidade C (d={dx}) [0,1]')
                ax.set_title(f'Plano CH (d={dx})', fontweight='bold')
                ax.grid(True, alpha=0.3); ax.legend()
            plt.suptitle('Plano Complexidade-Entropia Multi-Escala (Foco em Bordas)', fontweight='bold')
            plt.tight_layout()
            plt.savefig(save_dir / 'plano_ch_multiescala.png', dpi=300, bbox_inches='tight')
            plt.close()

        # ROC comparativa (RF vs KNN)
        if eval_results is not None and self.model_rf and self.model_knn:
            y_test = eval_results['y_test']
            X_test_s = eval_results['X_test_scaled']
            
            # RF
            proba_rf = self.model_rf.predict_proba(X_test_s)
            idx_real_rf = list(self.model_rf.classes_).index('real')
            y_score_rf = proba_rf[:, idx_real_rf]
            
            # KNN
            proba_knn = self.model_knn.predict_proba(X_test_s)
            idx_real_knn = list(self.model_knn.classes_).index('real')
            y_score_knn = proba_knn[:, idx_real_knn]
            
            y_true = (y_test == 'real').astype(int)
            
            fpr_rf, tpr_rf, _ = roc_curve(y_true, y_score_rf)
            auc_rf = roc_auc_score(y_true, y_score_rf)
            
            fpr_knn, tpr_knn, _ = roc_curve(y_true, y_score_knn)
            auc_knn = roc_auc_score(y_true, y_score_knn)

            plt.figure(figsize=(10, 8))
            plt.plot(fpr_rf, tpr_rf, color='darkorange', lw=2, label=f'RF (AUC={auc_rf:.3f})')
            plt.plot(fpr_knn, tpr_knn, color='green', lw=2, label=f'KNN (AUC={auc_knn:.3f})')
            plt.plot([0,1], [0,1], color='navy', lw=2, linestyle='--', label='Chance')
            plt.xlim([0,1]); plt.ylim([0,1.05])
            plt.xlabel('FPR (Taxa de Falsos Positivos)', fontsize=12)
            plt.ylabel('TPR (Taxa de Verdadeiros Positivos)', fontsize=12)
            plt.title('Curvas ROC Comparativas - Deepfake Detection', fontsize=14, fontweight='bold')
            plt.grid(True, alpha=0.3); plt.legend(loc='lower right', fontsize=11)
            plt.savefig(save_dir / 'roc_curve_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"\nAUC-ROC RF:  {auc_rf:.4f}")
            print(f"AUC-ROC KNN: {auc_knn:.4f}")
            
            # Gráfico de barras comparativo
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Matrizes de confusão
            cm_rf = eval_results['cm_rf']
            cm_knn = eval_results['cm_knn']
            
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                       xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
            axes[0].set_title('Matriz de Confusão - Random Forest', fontweight='bold')
            axes[0].set_ylabel('Verdadeiro')
            axes[0].set_xlabel('Predito')
            
            sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Greens', ax=axes[1],
                       xticklabels=['Fake', 'Real'], yticklabels=['Fake', 'Real'])
            axes[1].set_title('Matriz de Confusão - KNN', fontweight='bold')
            axes[1].set_ylabel('Verdadeiro')
            axes[1].set_xlabel('Predito')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'confusion_matrices_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()

        print(f"Visualizações salvas em: {save_dir}")

    def analyze_single_image(self, image_path, save_dir=None, use_knn=False):
        """Analisa uma imagem individual usando RF ou KNN e plota no plano CH.
        
        Parâmetros:
            image_path (str|Path): caminho da imagem a analisar.
            save_dir (str|Path|None): diretório para salvar a visualização.
            use_knn (bool): se True, usa KNN; caso contrário, usa RF.
        """
        model = self.model_knn if use_knn else self.model_rf
        model_name = "KNN" if use_knn else "Random Forest"
        
        if model is None:
            print(f"Modelo {model_name} não treinado. Execute train_and_evaluate() primeiro.")
            return None
        
        print(f"\n{'='*60}")
        print(f"Analisando imagem com {model_name}: {Path(image_path).name}")
        print("="*60)
        
        # Carrega e extrai features
        img = self.load_and_preprocess_image(image_path)
        if img is None:
            print("Falha ao carregar imagem.")
            return None
        
        feats = self.compute_ordinal_features_corrected(img)
        if not any(v != 0 for k, v in feats.items() if k.endswith('n_series')):
            print("Falha na extração de features (séries inválidas).")
            return None
        
        # Prepara features para predição
        feature_cols = [c for c in self.features_df.columns 
                       if c not in ['label', 'filename', 'dataset']]
        X_single = pd.DataFrame([feats])[feature_cols]
        X_single = X_single.reindex(columns=feature_cols, fill_value=0)
        X_single_scaled = self.scaler.transform(X_single)
        
        # Predição
        pred = model.predict(X_single_scaled)[0]
        proba = model.predict_proba(X_single_scaled)[0]
        classes = model.classes_
        proba_dict = {cls: prob for cls, prob in zip(classes, proba)}
        
        print(f"\nPredição ({model_name}): {pred.upper()}")
        print(f"Probabilidades: Fake={proba_dict.get('fake', 0):.3f} | Real={proba_dict.get('real', 0):.3f}")
        
        # Plota nos planos CH multi-escala
        valid_dims = [dx for dx in self.embedding_dims 
                     if f'd{dx}_H_mean' in self.features_df.columns]
        
        if not valid_dims:
            print("Sem dimensões válidas para plotar.")
            return {'prediction': pred, 'probabilities': proba_dict, 'features': feats}
        
        fig, axes = plt.subplots(1, len(valid_dims), figsize=(6*len(valid_dims), 5))
        if len(valid_dims) == 1:
            axes = [axes]
        
        real = self.features_df[self.features_df['label'] == 'real']
        fake = self.features_df[self.features_df['label'] == 'fake']
        
        marker_color = 'darkgreen' if use_knn else 'darkred' if pred == 'fake' else 'darkblue'
        
        for ax, dx in zip(axes, valid_dims):
            h_col, c_col = f'd{dx}_H_mean', f'd{dx}_C_mean'
            
            ax.scatter(real[h_col], real[c_col], s=30, alpha=0.4,
                      label='Real (ref)', color='blue', edgecolors='none')
            ax.scatter(fake[h_col], fake[c_col], s=30, alpha=0.4,
                      label='Fake (ref)', color='red', edgecolors='none')
            
            h_val = feats.get(h_col, 0)
            c_val = feats.get(c_col, 0)
            marker_style = 's' if pred == 'fake' else 'D'
            
            ax.scatter(h_val, c_val, s=300, alpha=0.9,
                      label=f'Imagem ({pred})', color=marker_color,
                      edgecolors='black', linewidths=2, marker=marker_style, zorder=10)
            
            ax.set_xlabel(f'Entropia H (d={dx}) [0,1]')
            ax.set_ylabel(f'Complexidade C (d={dx}) [0,1]')
            ax.set_title(f'Plano CH (d={dx})', fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=9)
        
        plt.suptitle(f'{model_name}: {Path(image_path).name} → {pred.upper()}', 
                    fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        fig_path = None
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(exist_ok=True)
            model_suffix = "knn" if use_knn else "rf"
            fig_name = f"analise_{Path(image_path).stem}_{model_suffix}.png"
            fig_path = save_dir / fig_name
            plt.savefig(fig_path, dpi=300, bbox_inches='tight')
            print(f"\nVisualização salva: {fig_path}")
        
        plt.show()
        
        return {
            'prediction': pred,
            'probabilities': proba_dict,
            'features': feats,
            'figure_path': fig_path,
            'model_used': model_name
        }

def main():
    """Execução principal do script.

    Fluxo:
        1. Validamos ordpy com um teste rápido.
        2. Percorre datasets configurados, extrai features e salva CSV.
        3. Treina classificadores RandomForest e KNN (optionally grid-search).
        4. Gera visualizações comparativas (plano CH, ROC RF vs KNN) e salva em results/.
        5. Oferece modo interativo para análise de imagem individual com ambos os modelos.
    """

    # Smoke-test do ordpy
    print("\nValidando ordpy...")
    try:
        H, C = ordpy.complexity_entropy(np.random.randn(1000), dx=5)
        print(f"ordpy OK: H={H:.4f} (normalizado), C={C:.4f} | 0≤H≤1? {0<=H<=1}")
    except Exception as e:
        print(f"Erro ordpy: {e}")
        return

    # Caminhos base
    base_path = Path("/home/zerocopia/Projetos/mestrado/Datasets")
    datasets = [
        (base_path / "1/Dataset/Train/Real", 'real', 'dataset1'),
        (base_path / "1/Dataset/Train/Fake", 'fake', 'dataset1'),
        (base_path / "2/AI-face-detection-Dataset/real", 'real', 'dataset2'),
        (base_path / "2/AI-face-detection-Dataset/AI",   'fake', 'dataset2'),
    ]

    # Inicializa analisador
    analyzer = EdgeFocusedDeepfakeAnalyzer(embedding_dim=5, delay=1, multi_scale=True, verbose=True)

    # Processa datasets
    all_dfs = []
    for path, label, dsname in datasets:
        if not path.exists():
            print(f"Pulando (não existe): {path}")
            continue
        print(f"\n{'='*60}\nProcessando: {dsname} / {label}\n{'='*60}")
        df = analyzer.process_dataset(path, label, max_samples=200)
        if not df.empty:
            df['dataset'] = dsname
            all_dfs.append(df)

    if not all_dfs:
        print("\nNenhum dataset processado!")
        return

    # Combina e salva features
    analyzer.features_df = pd.concat(all_dfs, ignore_index=True)
    print("\n" + "="*60)
    print("RESUMO")
    print("="*60)
    print(f"Total de amostras: {len(analyzer.features_df)}")
    print("\nDistribuição por classe:")
    print(analyzer.features_df['label'].value_counts())

    results_dir = Path("/home/zerocopia/Projetos/mestrado/results")
    results_dir.mkdir(exist_ok=True)
    csv_path = results_dir / "features_final_bordas.csv"
    analyzer.features_df.to_csv(csv_path, index=False)
    print(f"\nFeatures salvas em: {csv_path}")

    # Estatísticas resumidas (por dimensão)
    for dx in analyzer.embedding_dims:
        h_col, c_col = f'd{dx}_H_mean', f'd{dx}_C_mean'
        if h_col in analyzer.features_df.columns:
            print(f"\n--- d={dx} ---")
            print("H_mean por classe:\n", analyzer.features_df.groupby('label')[h_col].describe())
            print("C_mean por classe:\n", analyzer.features_df.groupby('label')[c_col].describe())

    # Treino/avaliação (RF e KNN)
    eval_results = analyzer.train_and_evaluate(test_size=0.25, optimize=False)

    # Visualizações comparativas
    print("\nGerando visualizações comparativas...")
    analyzer.create_visualizations(results_dir, eval_results=eval_results)

    print("\n" + "="*60)
    print("ANÁLISE CONCLUÍDA")
    print("="*60)
    print("Arquivos gerados em:", results_dir)
    print("  - features_final_bordas.csv")
    print("  - plano_ch_multiescala.png")
    print("  - roc_curve_comparison.png (RF vs KNN)")
    print("  - confusion_matrices_comparison.png")
    
    # Modo interativo para análise individual
    print("\n" + "="*60)
    print("ANÁLISE DE IMAGEM INDIVIDUAL")
    print("="*60)
    print("Você pode analisar uma imagem específica com RF ou KNN.")
    print("Digite o caminho da imagem ou 'sair' para encerrar.\n")
    
    while True:
        user_input = input("Caminho da imagem (ou 'sair'): ").strip()
        if user_input.lower() in ['sair', 'exit', 'q', '']:
            break
        
        img_path = Path(user_input)
        if not img_path.exists():
            print(f"Arquivo não encontrado: {img_path}\n")
            continue
        
        model_choice = input("Usar qual modelo? (rf/knn) [rf]: ").strip().lower() or 'rf'
        use_knn = model_choice == 'knn'
        
        result = analyzer.analyze_single_image(img_path, save_dir=results_dir, use_knn=use_knn)
        if result:
            print(f"\nAnálise concluída para: {img_path.name} usando {result['model_used']}\n")
        print("-"*60 + "\n")
    
    print("\nEncerrando análise interativa.")

if __name__ == '__main__':
    main()
