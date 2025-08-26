# Sobre
Repositório para armazenar estudos, projetos e materiais relacionados ao Mestrado em Informática na Universidade Federal de Alagoas (UFAL). Inclui códigos-fonte, documentos, apresentações e outros recursos desenvolvidos durante o curso.

---  

# 🎓 **Projeto de Mestrado**  
## Universidade Federal de Alagoas (UFAL) - Instituto de Computação  
### Programa de Pós-Graduação em Informática  

## 📌 **Título**  
**Detecção Avançada de Mídias Sintéticas em Vídeos mediante Análise de Complexidade-Entropia**  

👨‍🎓 **Aluno:** Fábio Sant'Anna Linhares  
👩‍🏫 **Orientadora:** Prof.ª Dr.ª Fabiane da Silva Queiroz  
🔬 **Linha de Pesquisa:** Computação Visual e Inteligente  
🎯 **Tema de Pesquisa:** Visão Computacional: Análise, Caracterização e Classificação de Padrões Dinâmicos e Estruturais em Mídias Sintéticas

---

## 📝 **Resumo**  

A proliferação de **mídias sintéticas**, popularmente conhecidas como deepfakes, representa um desafio crescente para a segurança da informação e a confiança no ecossistema digital. A rápida evolução dos modelos generativos, como **Redes Adversariais Generativas (GANs)** e **Modelos de Difusão**, torna os métodos de detecção baseados em artefatos específicos rapidamente obsoletos.

Este projeto propõe uma **mudança de paradigma** na detecção de mídias sintéticas. Em vez de tratar imagens geradas por IA como imagens autênticas com defeitos, hipotetizamos que elas são o produto de um **sistema dinâmico complexo e determinístico**. Argumentamos que tais sistemas imprimem uma **"textura estatística"** única e mensurável, caracterizada por uma assinatura específica no espaço de complexidade-entropia.

Propomos o **Plano Causalidade Entropia-Complexidade (Plano CH)** como a ferramenta principal para capturar essa assinatura fundamental, visando criar um detector que seja, por construção, mais generalizável e interpretável. Esta abordagem combina a robustez teórica da **Teoria da Informação** com a capacidade de representação dos modelos de **aprendizado profundo**.

---

## 🎯 **Objetivos do Projeto**  

### **🔹 Objetivo Geral**
Desenvolver e validar um **framework híbrido e generalizável** para a detecção de vídeos deepfake, fundamentado na sinergia entre a análise de complexidade estatística e a extração de features de aprendizado profundo.

### **🔹 Objetivos Específicos**  
1. **Pipeline de Extração:** Implementar um pipeline robusto para a extração das coordenadas (H,C) do Plano CH a partir de frames de vídeo, incluindo uma análise de sensibilidade aos parâmetros de embedding dx e dy.

2. **Mapeamento de Assinaturas:** Mapear e caracterizar as "assinaturas de complexidade" de vídeos reais e falsos de múltiplos datasets (e.g., FaceForensics++, Celeb-DF) no Plano CH, validando empiricamente a Hipótese de Separação.

3. **Análise de Robustez:** Avaliar a robustez das features (H,C) a perturbações comuns do mundo real, como compressão de vídeo, adição de ruído e variações de iluminação.

4. **Modelo Híbrido:** Construir, treinar e validar um modelo híbrido que combine F_CH e F_ViT, testando sua capacidade de generalização contra um modelo baseline.

5. **Interpretabilidade:** Oferecer explicações e insights sobre os mecanismos de detecção, interpretando como as medidas capturam as anomalias.

---

## 🔬 **Hipóteses de Pesquisa**

### **H1 (Hipótese de Separação):**
Imagens geradas por diferentes modelos de IA (e.g., GANs, Modelos de Difusão) e imagens autênticas ocuparão regiões estatisticamente separáveis no Plano Causalidade Entropia-Complexidade.

### **H2 (Hipótese de Eficiência Informacional):**
O vetor de features bidimensional F_CH=[H,C], derivado do Plano CH, constitui um estimador estatisticamente mais eficiente da classe da imagem (real vs. falsa) do que features baseadas em artefatos, como as derivadas da Análise de Nível de Erro (ELA).

### **H3 (Hipótese de Sinergia Híbrida):**
Um modelo de classificação que funde as features interpretáveis do Plano CH (F_CH) com as features de representação global aprendidas por um Vision Transformer (F_ViT) exibirá desempenho superior em acurácia e generalização.

---

## 🛠 **Metodologia Proposta**

### **1️⃣ Pipeline de Extração de Features Estatísticas (F_CH)**
- **Implementação:** Conversão de frames para escala de cinza e varredura por janela deslizante de tamanho dx×dy
- **Parâmetros:** Investigação de dimensões de embedding dx e dy (e.g., 2×2, 3×2) respeitando (dx⋅dy)!≪W⋅H
- **Saída:** Vetor [H,C] para cada frame, constituindo features de baixa dimensão, computacionalmente eficientes e interpretáveis

### **2️⃣ Pipeline de Extração de Features de Deep Learning (F_ViT)**
- **Arquitetura:** Vision Transformer (ViT) pré-treinado (ViT-Base/16) como extrator "congelado"
- **Extração:** Vetor de embedding do token `[CLS]` da última camada para formar F_ViT
- **Justificativa:** Complementaridade conceitual entre padrões ordinais locais (PE2D) e dependências globais (ViT)

### **3️⃣ Fusão de Features e Classificação**
- **Método:** Concatenação simples: F_hybrid = [F_CH, F_ViT]
- **Classificador:** Gradient Boosting (XGBoost/LightGBM) para dados tabulares heterogêneos
- **Baseline:** Modelo utilizando apenas F_ViT para validação da Hipótese de Sinergia

### **4️⃣ Protocolo Experimental**
- **Datasets:** 
  - Treinamento/Validação: FaceForensics++ (FF++)
  - Teste Zero-Shot: Celeb-DF (v2)
- **Métricas:** AUC-ROC, EER (vídeo-level), Acurácia/Precisão/Recall/F1 (frame-level)
- **Robustez:** Degradações controladas (compressão JPEG, ruído Gaussiano)

---

## 🔧 **Ambiente de Desenvolvimento**

### **🐍 Python com Anaconda**
O projeto utiliza **Python** como linguagem principal, gerenciado através do **Anaconda** para garantir reprodutibilidade e isolamento de dependências.

#### **Instalação do Ambiente:**
```bash
# Criar ambiente conda
conda create -n deepfake-detection python=3.9
conda activate deepfake-detection

# Instalar dependências principais
conda install numpy pandas matplotlib scikit-learn
conda install pytorch torchvision torchaudio -c pytorch
pip install transformers ordpy
```

### **📊 Pacote ordpy**
O projeto utiliza intensivamente o pacote **ordpy** para análise de entropia de permutação e complexidade estatística.

#### **Sobre o ordpy:**
- **Repositório:** [arthurpessa/ordpy](https://github.com/arthurpessa/ordpy)
- **Documentação:** [ordpy.readthedocs.io](https://ordpy.readthedocs.io/)
- **Referência:** Pessa, A. A. B., & Ribeiro, H. V. (2021). ordpy: A Python package for data analysis with permutation entropy and ordinal network methods. *Chaos*, 31, 063110.

#### **Funcionalidades Utilizadas:**
- `ordpy.complexity_entropy()` - Cálculo do Plano Complexidade-Entropia
- `ordpy.permutation_entropy()` - Entropia de permutação para séries temporais e imagens
- `ordpy.two_by_two_patterns()` - Padrões ordinais 2×2 para análise de imagens
- `ordpy.ordinal_distribution()` - Distribuições ordinais para análise estatística

#### **Instalação:**
```bash
pip install ordpy
```

#### **Exemplo de Uso:**
```python
import ordpy
import numpy as np

# Análise de complexidade-entropia para imagem
H, C = ordpy.complexity_entropy(image_data, dx=2, dy=2)
print(f"Entropia: {H:.4f}, Complexidade: {C:.4f}")

# Padrões ordinais 2x2
patterns = ordpy.two_by_two_patterns(image_data, 
                                   taux=1, tauy=1, 
                                   overlapping=True, 
                                   tie_patterns=True)
```

---

## 📊 **Cronograma**

O projeto está planejado para execução ao longo de **24 meses**, dividido em quatro fases:

### **📚 Fase 1 (Meses 1-6): Fundamentação e Implementação**
- Revisão aprofundada da literatura
- Configuração do ambiente computacional (Anaconda + ordpy)
- Implementação dos pipelines F_CH e F_ViT
- Familiarização com datasets

### **🔬 Fase 2 (Meses 7-12): Experimentação**
- Extração de features nos datasets FF++ e Celeb-DF
- Análise de sensibilidade dos parâmetros PE2D
- Caracterização das assinaturas de complexidade
- Validação da Hipótese de Separação (H1)

### **🤖 Fase 3 (Meses 13-18): Desenvolvimento**
- Desenvolvimento do modelo híbrido
- Implementação do modelo baseline
- Treinamento e otimização
- Validação das hipóteses H2 e H3

### **📊 Fase 4 (Meses 19-24): Validação e Documentação**
- Protocolo de validação final
- Testes de generalização e robustez
- Análise dos resultados
- Redação da dissertação

---

## 📈 **Resultados Esperados**  

- **Validação Empírica:** Confirmação das três hipóteses centrais do projeto
- **Framework Inovador:** Desenvolvimento de um detector híbrido fundamentado em teoria
- **Generalização Superior:** Desempenho robusto em datasets não vistos durante treinamento
- **Interpretabilidade:** Explicações claras dos mecanismos de detecção
- **Contribuição Científica:** Publicações em conferências e periódicos de alto impacto
- **Código Aberto:** Disponibilização do framework para a comunidade científica

---

## 🔬 **Principais Referências**

📄 **Ribeiro, H. V. et al. (2012).** *Complexity-Entropy Causality Plane as a Complexity Measure for Two-Dimensional Patterns.* PLOS ONE, 7, e40689.

📄 **Pessa, A. A. B., & Ribeiro, H. V. (2021).** *ordpy: A Python package for data analysis with permutation entropy and ordinal network methods.* Chaos, 31, 063110.

📄 **Bandt, C., & Pompe, B. (2002).** *Permutation entropy: A Natural Complexity Measure for Time Series.* Physical Review Letters, 88, 174102.

📄 **Celeb-DF (2020).** *Celeb-DF: A Large-scale Challenging Dataset for DeepFake Forensics.* CVPR.

📄 **FaceForensics++ (2019).** *FaceForensics++: Learning to Detect Manipulated Facial Images.* ICCV.

📄 **Vaswani, A. et al. (2017).** *Attention is all you need.* Advances in Neural Information Processing Systems.

📄 **Martin, M. T., Plastino, A., & Rosso, O. A. (2006).** *Generalized Statistical Complexity Measures: Geometrical and Analytical Properties.* Physica A, 369, 439–462.

---

## 📬 **Contato**  
📩 **E-mail:** fl@ic.ufal.br  
🔗 **LinkedIn:** [linkedin.com/in/fabio-linhares](https://www.linkedin.com/in/fabio-linhares)  
🐙 **GitHub:** [github.com/fabio-linhares](https://github.com/fabio-linhares)
🌐 **Site do Projeto:** [fabiolinhares.com.br/ufal/orientacao/preprojeto](https://www.fabiolinhares.com.br/ufal/orientacao/preprojeto/preprojeto.html)
