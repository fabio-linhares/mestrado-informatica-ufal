<div align="center">
  <img src="docs/img/logo.png" alt="Logo da UFAL" width="200"/>
</div>


# 🎓 **Projeto de Mestrado**  

Repositório para armazenar estudos, projetos e materiais relacionados ao Mestrado em Informática na Universidade Federal de Alagoas (UFAL). Inclui códigos-fonte, documentos, apresentações e outros recursos desenvolvidos durante o curso.


## Universidade Federal de Alagoas (UFAL) - Instituto de Computação  
### Programa de Pós-Graduação em Informática  

## 📌 **Título**  
**Detecção Avançada de Mídias Sintéticas em Vídeos mediante Análise de Complexidade-Entropia**  

👨‍🎓 **Aluno:** Fábio Sant'Anna Linhares  
👩‍🏫 **Orientadora:** Prof.ª Dr.ª Fabiane da Silva Queiroz  
🔬 **Linha de Pesquisa:** Computação Visual e Inteligente  
🎯 **Tema de Pesquisa:** Visão Computacional: Análise, Caracterização e Classificação de Padrões Dinâmicos e Estruturais em Mídias Sintéticas

---

## 📝 **Introdução**  

A proliferação de **mídias sintéticas**, popularmente conhecidas como deepfakes, representa um desafio crescente para a segurança da informação e a confiança no ecossistema digital. A rápida evolução dos modelos generativos, como **Redes Adversariais Generativas (GANs)** e **Modelos de Difusão**, torna os métodos de detecção baseados em artefatos específicos rapidamente obsoletos. A comunidade de pesquisa enfrenta a necessidade premente de desenvolver detectores que não apenas apresentem alta acurácia, mas que também generalizem para métodos de manipulação desconhecidos e não vistos durante o treinamento.

A literatura atual é dominada por abordagens de aprendizado profundo, como **Redes Neurais Convolucionais (CNNs)** e **Vision Transformers (ViTs)**, que, apesar de seu desempenho notável, frequentemente operam como "caixas-pretas". Esses modelos podem aprender correlações espúrias nos dados de treinamento, o que limita sua robustez em cenários do mundo real. Existe uma lacuna significativa na literatura no que tange a métodos de detecção fundamentados em princípios teóricos que explorem a natureza intrínseca do conteúdo gerado por IA.

### **🎯 Mudança de Paradigma Proposta**

Este projeto de pesquisa propõe uma **mudança de paradigma**. Em vez de tratar imagens geradas por IA como imagens autênticas com defeitos, hipotetizamos que elas são o produto de um **sistema dinâmico complexo e determinístico**. Argumentamos que tais sistemas imprimem uma **"textura estatística"** única e mensurável, caracterizada por uma assinatura específica no espaço de complexidade-entropia, análoga à de sistemas caóticos.

Propomos o **Plano Causalidade Entropia-Complexidade (Plano CH)** como a ferramenta principal para capturar essa assinatura fundamental, visando criar um detector que seja, por construção, mais generalizável e interpretável. Esta abordagem combina a robustez teórica da **Teoria da Informação** com a capacidade de representação dos modelos de **aprendizado profundo**, oferecendo uma solução híbrida e inovadora para o problema da detecção de mídias sintéticas.

---

## 📚 **Justificativa**

A era da informação digital é marcada por um fluxo massivo de conteúdo cuja veracidade é frequentemente questionada. Imagens e vídeos não naturais — ou seja, gerados parcial ou totalmente por algoritmos de inteligência artificial, contendo um ou mais rostos humanos trocados ou não — constituem um novo tipo de artefato comunicacional: o que chamaremos **produtos de IA**.

A popularização de algoritmos generativos, como as **Redes Adversariais Generativas (GANs)** e os **modelos de difusão**, tem permitido a criação de conteúdo sintético visualmente consistente, muitas vezes indistinguível, a olho nu, de conteúdo natural e autêntico. Isso levanta sérias preocupações sobre **desinformação**, **manipulação de opinião pública** e **danos à imagem pessoal e coletiva**.

### **Limitações das Abordagens Atuais**

Pesquisas voltadas à detecção desses produtos sintéticos concentradas, em grande parte, em abordagens baseadas em **Deep Learning (DL)**, como Redes Neurais Convolucionais (CNNs) e Vision Transformers (ViTs) têm demonstrado resultados promissores. No entanto, muitos desses métodos se concentram na análise de artefatos espaciais e na detecção de anomalias em quadros individuais.

A **natureza temporal dos vídeos**, onde a evolução dos padrões e correlações ao longo do tempo é crucial, nos parece menos explorada. Produtos de IA em vídeo frequentemente carregam **traços dinâmicos atípicos**, exibem **inconsistências temporais sutis**, como falhas em padrões de piscar, movimentos de cabeça não naturais, ou transições abruptas entre expressões faciais, que podem não ser evidentes em um único quadro, mas se tornam detectáveis ao analisar a série temporal de características extraídas.

### **Fundamentação Teórica**

É neste ponto que as ferramentas da **Teoria da Informação** e da **Análise de Sistemas Dinâmicos Complexos** se mostram particularmente adequadas. A **entropia de Shannon** quantifica a incerteza de um sistema, enquanto a **complexidade estatística** mede o grau de estrutura e padrões, complementando a entropia.

O **Plano Complexidade-Entropia (CECP)**, e sua extensão **Multivariada (MvCECP)**, provaram ser eficazes na distinção de sistemas com dinâmicas variadas — periódicas, caóticas e estocásticas — ao mapear as características de suas séries temporais em um espaço bidimensional.

A **entropia de permutação** (Bandt e Pompe) é uma medida robusta e computacionalmente eficiente para extrair padrões ordinais de séries temporais. O parâmetro **embedding delay (τ)**, por sua vez, permite investigar as séries temporais em diferentes escalas de tempo, revelando dinâmicas ocultas ou anômalas.

### **Potencial de Detecção**

Acreditamos que a aplicação dessas ferramentas aos produtos de IA permitirá capturar as **"digitais" dinâmicas da manipulação** de forma mais precisa. Por exemplo, a suavidade excessiva de certas áreas manipuladas ou a ausência de padrões ordinais esperados em movimentos faciais podem ser detectadas como desvios em medidas de complexidade-entropia.

Além disso, a **Teoria da Estimação Estatística**, particularmente o **princípio da máxima entropia de Jaynes**, fornecerá a base formal para inferir as distribuições de probabilidade que melhor representam os dados, garantindo que as inferências sobre a natureza das mídias sintéticas sejam as menos preconceituosas e mais objetivas possíveis.

---

## 📋 **Protocolo PICOC**

Para estruturar sistematicamente a revisão da literatura, utilizaremos o protocolo **PICOC (Population, Intervention, Comparison, Outcomes, Context)**, que fornece um framework robusto para a formulação de questões de pesquisa e busca bibliográfica:

### **🎯 Population (População)**
- **Imagens e vídeos digitais** gerados por algoritmos de inteligência artificial
- **Mídias sintéticas** (deepfakes) criadas por GANs, modelos de difusão e outras técnicas generativas
- **Datasets de referência**: FaceForensics++, Celeb-DF, DFDC, etc.

### **🔬 Intervention (Intervenção)**
- **Análise de complexidade-entropia** baseada em entropia de permutação
- **Plano Causalidade Entropia-Complexidade (Plano CH)**
- **Extração de features estatísticas** usando padrões ordinais bidimensionais
- **Fusão com features de Vision Transformers** para detecção híbrida

### **⚖️ Comparison (Comparação)**
- **Métodos tradicionais** baseados em CNNs (ResNet, EfficientNet)
- **Abordagens de análise de artefatos** (ELA, análise espectral)
- **Detectores baseados em ViTs** puros
- **Métodos ensemble** convencionais

### **📊 Outcomes (Resultados)**
- **Acurácia de detecção** (AUC-ROC, EER)
- **Capacidade de generalização** cross-dataset
- **Robustez** a perturbações (compressão, ruído)
- **Interpretabilidade** dos mecanismos de detecção
- **Eficiência computacional**

### **🌍 Context (Contexto)**
- **Detecção de deepfakes** em ambiente controlado e real
- **Aplicações de segurança da informação**
- **Cenários de forense digital**
- **Mitigação de desinformação**

---

## ❓ **Questões de Pesquisa (QA)**

### **🔍 Questão Principal (QP)**
**"Como a análise de complexidade-entropia pode aprimorar a detecção de mídias sintéticas em vídeos, superando as limitações de generalização dos métodos atuais baseados em deep learning?"**

### **📋 Questões Secundárias (QS)**

**QS1:** Quais são as assinaturas estatísticas distintivas de vídeos sintéticos no espaço complexidade-entropia comparadas às de vídeos autênticos?

**QS2:** Como a fusão de features de complexidade-entropia com representações de Vision Transformers impacta na capacidade de generalização cross-dataset?

**QS3:** Qual é a robustez das features baseadas em entropia de permutação contra degradações comuns (compressão, ruído) em vídeos?

**QS4:** Como os parâmetros de embedding (dx, dy) influenciam na separabilidade entre classes no Plano CH?

**QS5:** Qual é o trade-off entre interpretabilidade e performance dos detectores híbridos propostos comparados aos métodos estado-da-arte?

**QS6:** Como as características temporais dos vídeos deepfake se manifestam através da análise de séries temporais de complexidade-entropia?

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

## � **Datasets Utilizados**

O projeto incorpora múltiplos datasets especializados para garantir robustez e generalização na detecção de deepfakes:

### **🗂️ Dataset 1: Deepfake and Real Images**
- **Localização:** `/Datasets/1/Deepfake and real images.zip`
- **Tipo:** Imagens estáticas (deepfake vs. reais)
- **Aplicação:** Treinamento inicial e validação de features de complexidade-entropia
- **Características:** Dataset balanceado para análise de padrões ordinais em imagens sintéticas

### **🗂️ Dataset 2: Detect AI-Generated Faces High-Quality**
- **Localização:** `/Datasets/2/Detect AI-Generated Faces High-Quality Dataset.zip`
- **Fonte:** Kaggle - `shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset`
- **Tipo:** Faces de alta qualidade geradas por IA
- **Aplicação:** Teste de robustez e validação cross-dataset
- **Instalação:**
```python
import kagglehub
path = kagglehub.dataset_download("shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset")
```

### **🎯 Datasets de Referência Acadêmica**
Conforme protocolo PICOC, o projeto também utiliza datasets consolidados:
- **FaceForensics++:** Dataset principal para treinamento/validação
- **Celeb-DF v2:** Avaliação zero-shot de generalização
- **DFDC:** Validação adicional em cenários desafiadores

---

## 📚 **Base Teórica e Artigos Fundamentais**

### **🔬 Artigos Teóricos de Base**
Localizados em `/docs/artigos/`:

#### **📄 Complexity-Entropy Causality Plane as a Complexity.pdf**
- **Referência:** Ribeiro, H. V. et al. (2012)
- **Contribuição:** Fundamentação teórica do Plano CH para análise bidimensional
- **Aplicação:** Base matemática para extração de features F_CH

#### **📄 Distinguishing noise from chaos.pdf**
- **Contribuição:** Metodologia para separação de dinâmicas determinísticas e estocásticas
- **Aplicação:** Validação da Hipótese de Separação (H1)

#### **📄 Theory of Statistical Estimation.pdf**
- **Contribuição:** Princípio da máxima entropia de Jaynes
- **Aplicação:** Inferência estatística objetiva sobre mídias sintéticas

#### **📄 How-to_conduct_a_systematic_literature_review.pdf**
- **Contribuição:** Metodologia PICOC para revisão sistemática
- **Aplicação:** Estruturação da pesquisa bibliográfica

---

## 🔍 **Protocolo PICOC: Implementação e Resultados**

### **📋 Preparação da Revisão Sistemática**
Localizada em `/docs/picoc/preparacao/`:

#### **🎯 Bases de Dados Utilizadas**
- **Web of Science:** Coleção Principal (1945-presente) - 9.000+ periódicos indexados
- **IEEE Xplore:** Biblioteca Digital completa (1988-presente) - 6M+ documentos
- **Scopus (Elsevier):** Base multidisciplinar abrangente
- **ScienceDirect:** 3.800+ periódicos e 37.000+ títulos de livros

#### **📝 Artigos Selecionados (25 Principais)**
Conforme lista em `/docs/picoc/preparacao/artigos selecionados`:

**Surveys e Reviews Fundamentais:**
- Khan A.A. (2025): "A survey on multimedia-enabled deepfake detection" - *Discover Computing*
- Kadha V. (2025): "Unravelling Digital Forgeries: A Systematic Survey" - *ACM Computing Surveys*

**Métodos de Análise Temporal:**
- Zhang Y. (2025): "Exploring coordinated motion patterns of facial landmarks" - *Applied Soft Computing*
- Zhu C. (2024): "Deepfake detection via inter-frame inconsistency recomposition" - *Pattern Recognition*

**Abordagens de Análise de Frequência:**
- Qiusong L. (2025): "Joint spatial-frequency deepfake detection network" - *Applied Intelligence*
- Shi Z. (2025): "Customized Transformer Adapter With Frequency Masking" - *IEEE TIFS*

**Métodos Baseados em Teoria da Informação:**
- Sheng Z. (2025): "SUMI-IFL: An Information-Theoretic Framework" - *AAAI 2025*
- Sudarsan M. (2025): "LEAD-AI: Lightweight Entropy Analysis" - *SPIE*

### **✅ Artigos Aprovados para Revisão**
Localizados em `/docs/picoc/aprovados/1-11/`:
- **11 artigos** selecionados após aplicação dos critérios de QA
- Cada pasta contém: PDF completo, arquivo .bib, e metadados HTML
- Critérios de aprovação baseados nas 7 questões de avaliação (Q1-Q7)

---

## ❓ **Questões de Avaliação (QA) - Refinadas**

### **📊 Critérios de Qualidade dos Estudos**
Baseados em análise em `/docs/picoc/preparacao/perguntas_avaliacao`:

#### **🔬 Rigor Metodológico**
**Q1:** O estudo reporta métricas de avaliação claras e apropriadas para a tarefa (ex: Acurácia, AUC-ROC, EER)?

**Q2:** O estudo utiliza datasets públicos e bem conhecidos para validação (ex: FaceForensics++, Celeb-DF)?

**Q3:** O método proposto é comparado com pelo menos um outro método de detecção já existente (baseline)?

#### **🎯 Robustez e Aplicabilidade**
**Q4:** O estudo avalia a robustez do detector contra perturbações comuns (ex: compressão, ruído, variações de iluminação)?

**Q5:** A metodologia proposta é descrita com detalhes suficientes para permitir a sua replicação?

#### **📈 Credibilidade Científica**
**Q6:** Os autores discutem as limitações do estudo e as ameaças à validade dos resultados?

**Q7:** Os objetivos da pesquisa, as contribuições e as questões de pesquisa do estudo estão claramente definidos?

### **📚 Artigos em Análise Detalhada (QA)**
Localizados em `/docs/picoc/qa/`:

#### **Surveys e Estado-da-Arte:**
- "A survey on multimedia-enabled deepfake detection state-of-the-art tools and techniques..."
- "Unravelling Digital Forgeries A Systematic Survey on Image Manipulation Detection..."

#### **Métodos Baseados em Transformers:**
- "Customized Transformer Adapter With Frequency Masking for Deepfake Detection"
- "WaveConViT: Wavelet-Based Convolutional Vision Transformer..."

#### **Abordagens de Análise Temporal e Espacial:**
- "Exploring coordinated motion patterns of facial landmarks for deepfake video detection"
- "Joint spatial-frequency deepfake detection network based on dual-domain attention..."

#### **Métodos Baseados em Teoria da Informação:**
- "LEAD-AI lightweight entropy analysis for distinguishing AI-generated images..."
- "SUMI-IFL An Information-Theoretic Framework for Image Forgery Localization..."

#### **Análise de Robustez:**
- "DPL Cross-quality DeepFake Detection via Dual Progressive Learning"
- "Detecting face tampering in videos using deepfake forensics"

---

## �🔧 **Ambiente de Desenvolvimento**

### **🐍 Python com Anaconda**
O projeto utiliza **Python** como linguagem principal, gerenciado através do **Anaconda** para garantir reprodutibilidade e isolamento de dependências.

#### **Instalação do Ambiente:**
```bash
# Criar ambiente conda
conda create -n a python=3.9
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

## 📚 **Referências Bibliográficas**

AGARWAL, S. et al. Detecting face synthesis using convolutional neural networks and image quality assessment. **IEEE Transactions on Information Forensics and Security**, v. 15, p. 3044-3055, 2020.

AFCHAR, D. et al. MesoNet: a Compact Facial Video Forgery Detection Network. In: **IEEE International Workshop on Information Forensics and Security (WIFS)**. Hong Kong: IEEE, 2018. p. 1-7. DOI: [10.1109/WIFS.2018.8630761](https://doi.org/10.1109/WIFS.2018.8630761).

AMERINI, I. et al. Deepfake-o-meter: An open platform for deepfake detection. In: **Proceedings of the 29th ACM International Conference on Multimedia**. Virtual Event: ACM, 2021. p. 103-112. DOI: [10.1145/3474085.3475667](https://doi.org/10.1145/3474085.3475667).

ANDERSON, R. J. **Security Engineering: A Guide to Building Dependable Distributed Systems**. 3. ed. Hoboken: John Wiley & Sons, 2020.

ANTUNES, P. et al. Leveraging ordinal patterns for improved deepfake detection. **Neural Computing and Applications**, v. 34, n. 18, p. 15479-15493, 2022. DOI: [10.1007/s00521-022-07043-5](https://doi.org/10.1007/s00521-022-07043-5).

BANDT, C.; POMPE, B. Permutation entropy: a natural complexity measure for time series. **Physical Review Letters**, v. 88, n. 17, p. 174102, 2002. DOI: [10.1103/PhysRevLett.88.174102](https://doi.org/10.1103/PhysRevLett.88.174102).

BONETTINI, N. et al. Video face manipulation detection through ensemble of CNNs. In: **International Conference on Pattern Recognition (ICPR)**. Milan: IEEE, 2020. p. 5012-5019. DOI: [10.1109/ICPR48806.2021.9412711](https://doi.org/10.1109/ICPR48806.2021.9412711).

BROWN, T. et al. Language models are few-shot learners. In: **Advances in Neural Information Processing Systems**, v. 33, p. 1877-1901, 2020. Disponível em: [https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf](https://proceedings.neurips.cc/paper/2020/file/1457c0d6bfcb4967418bfb8ac142f64a-Paper.pdf). Acesso em: 26 ago. 2025.

CALDELLI, R.; BECARELLI, R.; AMERINI, I. Image origin classification based on social network provenance. **IEEE Transactions on Information Forensics and Security**, v. 12, n. 6, p. 1299-1308, 2017. DOI: [10.1109/TIFS.2017.2656842](https://doi.org/10.1109/TIFS.2017.2656842).

CHEN, S. et al. The eyes tell all: detecting fake face images via the eyes. **IEEE Access**, v. 8, p. 149915-149924, 2020. DOI: [10.1109/ACCESS.2020.3016867](https://doi.org/10.1109/ACCESS.2020.3016867).

CHOLLET, F. Xception: Deep learning with depthwise separable convolutions. In: **Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition**. Honolulu: IEEE, 2017. p. 1251-1258. DOI: [10.1109/CVPR.2017.195](https://doi.org/10.1109/CVPR.2017.195).

DOLHANSKY, B. et al. The DeepFake Detection Challenge (DFDC) Dataset and Benchmark. **arXiv preprint** arXiv:2006.07397, 2020. Disponível em: [https://arxiv.org/abs/2006.07397](https://arxiv.org/abs/2006.07397). Acesso em: 26 ago. 2025.

DOSOVITSKIY, A. et al. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. In: **International Conference on Learning Representations (ICLR)**. Vienna: OpenReview, 2021. Disponível em: [https://openreview.net/forum?id=YicbFdNTTy](https://openreview.net/forum?id=YicbFdNTTy). Acesso em: 26 ago. 2025.

DURALL, R. et al. Watch your up-convolution: CNN based generative deep neural networks are failing to reproduce spectral distributions. In: **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition**. Seattle: IEEE, 2020. p. 7890-7899. DOI: [10.1109/CVPR42600.2020.00791](https://doi.org/10.1109/CVPR42600.2020.00791).

FRANK, J.; EISENHOFER, T.; SCHÖNHERR, L. Leveraging frequency analysis for deep fake image recognition. In: **International Conference on Machine Learning**. PMLR, 2020. p. 3247-3258. Disponível em: [http://proceedings.mlr.press/v119/frank20a.html](http://proceedings.mlr.press/v119/frank20a.html). Acesso em: 26 ago. 2025.

GOODFELLOW, I. et al. Generative Adversarial Nets. In: **Advances in Neural Information Processing Systems**, v. 27, p. 2672-2680, 2014. Disponível em: [https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf](https://proceedings.neurips.cc/paper/2014/file/5ca3e9b122f61f8f06494c97b1afccf3-Paper.pdf). Acesso em: 26 ago. 2025.

GUARNERA, L. et al. Deepfake video detection through optical flow based CNN. In: **Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops**. Seoul: IEEE, 2019. p. 1205-1207. DOI: [10.1109/ICCVW.2019.00152](https://doi.org/10.1109/ICCVW.2019.00152).

HE, K. et al. Deep residual learning for image recognition. In: **Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition**. Las Vegas: IEEE, 2016. p. 770-778. DOI: [10.1109/CVPR.2016.90](https://doi.org/10.1109/CVPR.2016.90).

HEUSEL, M. et al. GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In: **Advances in Neural Information Processing Systems**, v. 30, 2017. Disponível em: [https://proceedings.neurips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/8a1d694707eb0fefe65871369074926d-Paper.pdf). Acesso em: 26 ago. 2025.

JIANG, L. et al. Celeb-DF: A large-scale challenging dataset for deepfake forensics. In: **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition**. Seattle: IEEE, 2020. p. 3207-3216. DOI: [10.1109/CVPR42600.2020.00327](https://doi.org/10.1109/CVPR42600.2020.00327).

KARRAS, T. et al. Analyzing and improving the image quality of StyleGAN. In: **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition**. Seattle: IEEE, 2020. p. 8110-8119. DOI: [10.1109/CVPR42600.2020.00813](https://doi.org/10.1109/CVPR42600.2020.00813).

LI, L. et al. Face X-ray for more general face forgery detection. In: **Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition**. Seattle: IEEE, 2020. p. 5001-5010. DOI: [10.1109/CVPR42600.2020.00505](https://doi.org/10.1109/CVPR42600.2020.00505).

LI, Y. et al. In ictu oculi: Exposing AI generated fake face videos by detecting eye blinking. In: **IEEE International Workshop on Information Forensics and Security (WIFS)**. Hong Kong: IEEE, 2018. p. 1-7. DOI: [10.1109/WIFS.2018.8630787](https://doi.org/10.1109/WIFS.2018.8630787).

LOPEZ-PAZ, D.; OQUAB, M. Revisiting classifier two-sample tests. In: **International Conference on Learning Representations**. Toulon: OpenReview, 2017. Disponível em: [https://openreview.net/forum?id=SJkXfE5xx](https://openreview.net/forum?id=SJkXfE5xx). Acesso em: 26 ago. 2025.

## 📬 **Contato**

📩 **E-mail:** [fl@ic.ufal.br](mailto:fl@ic.ufal.br)  
🔗 **LinkedIn:** [linkedin.com/in/fabio-linhares](https://www.linkedin.com/in/fabio-linhares)  
🐙 **GitHub:** [github.com/fabio-linhares](https://github.com/fabio-linhares)  
🌐 **Site do Projeto:** [fabiolinhares.com.br/ufal/orientacao/preprojeto](https://www.fabiolinhares.com.br/ufal/orientacao/preprojeto/preprojeto.html)

---

**Trabalho de Mestrado - Programa de Pós-Graduação em Informática**  
**Universidade Federal de Alagoas (UFAL)**  
**Orientador:** Prof.ª Dr.ª Fabiane da Silva Queiroz  
**Ano:** 2025
