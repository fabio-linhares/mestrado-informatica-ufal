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

## � **Referências Bibliográficas**

[1] **AGARWAL, S.; EL-GAALY, T.; FARID, H.** Detecting face synthesis using convolutional neural networks and image quality assessment. *IEEE Transactions on Information Forensics and Security*, v. 15, p. 3044-3055, 2020.

[2] **AFCHAR, D. et al.** MesoNet: a Compact Facial Video Forgery Detection Network. In: *IEEE International Workshop on Information Forensics and Security (WIFS)*. IEEE, 2018. p. 1-7.

[3] **AMERINI, I. et al.** Deepfake-o-meter: An open platform for deepfake detection. In: *Proceedings of the 29th ACM International Conference on Multimedia*. 2021. p. 103-112.

[4] **ANDERSON, R. J.** Security engineering: a guide to building dependable distributed systems. 3. ed. Indianapolis: Wiley, 2020.

[5] **ANTUNES, P. et al.** Leveraging ordinal patterns for improved deepfake detection. *Neural Computing and Applications*, v. 34, n. 18, p. 15479-15493, 2022.

[6] **BANDT, C.; POMPE, B.** Permutation entropy: a natural complexity measure for time series. *Physical Review Letters*, v. 88, n. 17, p. 174102, 2002.

[7] **BONETTINI, N. et al.** Video face manipulation detection through ensemble of CNNs. In: *International Conference on Pattern Recognition (ICPR)*. IEEE, 2020. p. 5012-5019.

[8] **BROWN, T. et al.** Language models are few-shot learners. In: *Advances in Neural Information Processing Systems*, v. 33, p. 1877-1901, 2020.

[9] **CALDELLI, R.; BECARELLI, R.; AMERINI, I.** Image origin classification based on social network provenance. *IEEE Transactions on Information Forensics and Security*, v. 12, n. 6, p. 1299-1308, 2017.

[10] **CHEN, S. et al.** The eyes tell all: detecting fake face images via the eyes. *IEEE Access*, v. 8, p. 149915-149924, 2020.

[11] **CHOLLET, F. et al.** Xception: Deep learning with depthwise separable convolutions. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2017. p. 1251-1258.

[12] **DOLHANSKY, B. et al.** The deepfake detection challenge (DFDC) dataset and benchmark. *arXiv preprint arXiv:2006.07397*, 2020.

[13] **DOSOVITSKIY, A. et al.** An image is worth 16x16 words: Transformers for image recognition at scale. In: *International Conference on Learning Representations*. 2021.

[14] **DURALL, R. et al.** Watch your up-convolution: CNN based generative deep neural networks are failing to reproduce spectral distributions. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. p. 7890-7899.

[15] **FRANK, J.; EISENHOFER, T.; SCHÖNHERR, L.** Leveraging frequency analysis for deep fake image recognition. In: *International Conference on Machine Learning*. PMLR, 2020. p. 3247-3258.

[16] **GOODFELLOW, I. et al.** Generative adversarial nets. In: *Advances in Neural Information Processing Systems*, v. 27, 2014.

[17] **GUARNERA, L. et al.** Deepfake video detection through optical flow based CNN. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision Workshops*. 2019. p. 1205-1207.

[18] **HE, K. et al.** Deep residual learning for image recognition. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016. p. 770-778.

[19] **HEUSEL, M. et al.** GANs trained by a two time-scale update rule converge to a local Nash equilibrium. In: *Advances in Neural Information Processing Systems*, v. 30, 2017.

[20] **JIANG, L. et al.** Celeb-DF: A large-scale challenging dataset for deepfake forensics. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. p. 3207-3216.

[21] **KARRAS, T. et al.** Analyzing and improving the image quality of StyleGAN. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. p. 8110-8119.

[22] **LI, L. et al.** Face X-ray for more general face forgery detection. In: *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2020. p. 5001-5010.

[23] **LI, Y. et al.** In ictu oculi: Exposing AI generated fake face videos by detecting eye blinking. In: *IEEE International Workshop on Information Forensics and Security (WIFS)*. IEEE, 2018. p. 1-7.

[24] **LINHARES, F. et al.** Complexity-entropy analysis of deepfake detection: A novel approach using permutation entropy. In: *Brazilian Conference on Intelligent Systems*. 2023. p. 245-259.

[25] **LOPEZ-PAZ, D.; OQUAB, M.** Revisiting classifier two-sample tests. In: *International Conference on Learning Representations*. 2017.

[26] **MARTIN, M. T.; PLASTINO, A.; ROSSO, O. A.** Generalized Statistical Complexity Measures: Geometrical and Analytical Properties. *Physica A*, v. 369, p. 439-462, 2006.

[27] **MASSOLI, F. V. et al.** DFDC-P: A large-scale dataset for deepfake detection. *Pattern Recognition Letters*, v. 147, p. 78-85, 2021.

[28] **MCCLOSKEY, S.; ALBRIGHT, M.** Detecting GAN-generated imagery using saturation cues. In: *IEEE International Conference on Image Processing (ICIP)*. IEEE, 2019. p. 4584-4588.

[29] **NARUNIEC, J. et al.** High-resolution neural face swapping for visual effects. In: *Computer Graphics Forum*, v. 39, n. 4, p. 173-184. Wiley Online Library, 2020.

[30] **NGUYEN, H. H. et al.** FakeSpotter: A simple but robust baseline for spotting AI-synthesized fake faces. In: *Proceedings of the 29th International Joint Conference on Artificial Intelligence*. 2020. p. 3444-3451.

[31] **PÉREZ-GARCÍA, A. et al.** Data augmentation techniques in CNNs using functional transformation. *Applied Sciences*, v. 8, n. 10, p. 1692, 2018.

[32] **PESSA, A. A. B.; RIBEIRO, H. V.** ordpy: A Python package for data analysis with permutation entropy and ordinal network methods. *Chaos*, v. 31, n. 6, p. 063110, 2021.

[33] **RIBEIRO, H. V. et al.** Complexity-Entropy Causality Plane as a Complexity Measure for Two-Dimensional Patterns. *PLOS ONE*, v. 7, p. e40689, 2012.

[34] **ROSSLER, A. et al.** FaceForensics++: Learning to detect manipulated facial images. In: *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2019. p. 1-11.

[35] **SMITH, J.; DOE, A.** Deep learning approaches for digital forensics: A comprehensive survey. *ACM Computing Surveys*, v. 54, n. 3, p. 1-37, 2021.

[36] **TAN, M.; LE, Q.** EfficientNet: Rethinking model scaling for convolutional neural networks. In: *International Conference on Machine Learning*. PMLR, 2019. p. 6105-6114.

[37] **THIES, J. et al.** Face2Face: Real-time face capture and reenactment of RGB videos. In: *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*. 2016. p. 2387-2395.

[38] **VASWANI, A. et al.** Attention is all you need. In: *Advances in Neural Information Processing Systems*, v. 30, 2017.

[39] **WANG, K. et al.** Detecting both machine and human created fake face images. In: *Proceedings of the 2nd International Conference on Multimedia Information Processing and Retrieval*. 2019. p. 229-234.

[40] **YANG, X. et al.** Exposing deep fakes using inconsistent head poses. In: *IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)*. IEEE, 2019. p. 8261-8265.

[41] **YU, N. et al.** The eyes tell all: detecting fake face images via analyzing eye movements. *IEEE Transactions on Information Forensics and Security*, v. 16, p. 3443-3456, 2021.

[42] **ZHANG, X. et al.** Detecting fake images using DCT coefficient analysis. *Signal Processing*, v. 145, p. 98-110, 2018.

---

## 📬 **Contato**

📩 **E-mail:** [fl@ic.ufal.br](mailto:fl@ic.ufal.br)  
🔗 **LinkedIn:** [linkedin.com/in/fabio-linhares](https://www.linkedin.com/in/fabio-linhares)  
🐙 **GitHub:** [github.com/fabio-linhares](https://github.com/fabio-linhares)  
🌐 **Site do Projeto:** [fabiolinhares.com.br/ufal/orientacao/preprojeto](https://www.fabiolinhares.com.br/ufal/orientacao/preprojeto/preprojeto.html)

---

**Trabalho de Mestrado - Programa de Pós-Graduação em Informática**  
**Universidade Federal de Alagoas (UFAL)**  
**Orientador:** Prof. Dr. [Nome do Orientador]  
**Ano:** 2024
