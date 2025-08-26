<div align="center">
  <img src="docs/img/logo.png" alt="Logo da UFAL" width="200"/>
</div>

<div align="center">

### Universidade Federal de Alagoas (UFAL)
### Instituto de Computação  
#### Programa de Pós-Graduação em Informática  

**Projeto de Mestrado**  

</div>

Repositório para armazenar estudos, projetos e materiais relacionados ao Mestrado em Informática na Universidade Federal de Alagoas (UFAL). Inclui códigos-fonte, documentos, apresentações e outros recursos desenvolvidos durante o curso.




## 📌 **Título**  
**Detecção Avançada de Mídias Sintéticas em Vídeos mediante Análise de Complexidade-Entropia**  

👨‍🎓 **Aluno:** Fábio Linhares  
👩‍🏫 **Orientadora:** Prof.ª Dr.ª Fabiane da Silva Queiroz  
🔬 **Linha de Pesquisa:** Computação Visual e Inteligente  
🎯 **Tema de Pesquisa:** Visão Computacional: Análise, Caracterização e Classificação de Padrões Dinâmicos e Estruturais em Mídias Sintéticas

---

[➡️ Relatório executivo da RSL](docs/picoc/qa/RELATORIO_EXECUTIVO.md)

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

Pesquisas voltadas à detecção desses produtos sintéticos concentradas, em grande parte, em abordagens baseadas em **Deep Learning (DL)**, como Redes Neurais Convolucionais (CNNs) e Vision Transformers (ViTs) têm demonstrado resultados promissores. No entanto, muitos desses métodos focam na análise de artefatos espaciais e na detecção de anomalias em quadros individuais.

A **natureza temporal dos vídeos**, onde a evolução dos padrões e correlações ao longo do tempo é crucial, nos parece menos explorada. Produtos de IA em vídeo frequentemente carregam **traços dinâmicos atípicos**, exibem **inconsistências temporais sutis**, como falhas em padrões de piscar, movimentos de cabeça não naturais, ou transições abruptas entre expressões faciais, que podem não ser evidentes em um único quadro, mas se tornam detectáveis ao analisar a série temporal de características extraídas.

### **Fundamentação Teórica**

É neste ponto que as ferramentas da **Teoria da Informação** e da **Análise de Sistemas Dinâmicos Complexos** se mostram particularmente adequadas. A **entropia de Shannon** quantifica a incerteza de um sistema, enquanto a **complexidade estatística** mede o grau de estrutura e padrões, complementando a entropia.

O **Plano Complexidade-Entropia (CECP)**, e sua extensão **Multivariada (MvCECP)**, provaram ser eficazes na distinção de sistemas com dinâmicas variadas — periódicas, caóticas e estocásticas — ao mapear as características de suas séries temporais em um espaço bidimensional.

A **entropia de permutação** (Bandt e Pompe) é uma medida robusta e computacionalmente eficiente para extrair padrões ordinais de séries temporais. O parâmetro **embedding delay (τ)**, por sua vez, permite investigar as séries temporais em diferentes escalas de tempo, revelando dinâmicas ocultas ou anômalas.

### **Potencial de Detecção**

Acreditamos que a aplicação dessas ferramentas aos produtos de IA permitirá capturar as **"digitais" dinâmicas da manipulação** de forma mais precisa. Por exemplo, a suavidade excessiva de certas áreas manipuladas ou a ausência de padrões ordinais esperados em movimentos faciais podem ser detectadas como desvios em medidas de complexidade-entropia. Além disso, a Teoria da Estimação Estatística, particularmente o princípio da máxima entropia de Jaynes, fornecerá a base formal para inferir as distribuições de probabilidade que melhor representam os dados, garantindo que as inferências sobre a natureza das mídias sintéticas sejam as menos preconceituosas e mais objetivas possíveis.

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

### **4️⃣ Datasets Experimentais**


A seguir estão os datasets utilizados neste trabalho. Para cada um fornecemos uma breve descrição, origem/identificador e observações relevantes para reprodutibilidade e conformidade.


#### AI Generated Images - High Quality  
- **Descrição breve:** Conjunto de imagens de alta qualidade geradas por modelos de síntese de imagens (GANs / diffusion models). Focado em rostos e cenas realistas produzidas por IA; útil para treinar e avaliar classificadores que discriminam imagens sintéticas de imagens reais.  
- **Origem / identificador:** Kaggle — `shahzaibshazoo/detect-ai-generated-faces-high-quality-dataset`  
- **Conteúdo típico:** Imagens geradas por IA em alta resolução, agrupadas por fonte/gerador quando disponível; metadados podem incluir rótulos de origem do gerador.  
- **Uso neste projeto:** Base para avaliação da capacidade do classificador em identificar artefatos de imagens sintetizadas e medir robustez frente a alta qualidade visual.  


---

#### Deepfake and Real Images  
- **Descrição breve:** Conjunto misto contendo imagens reais e imagens deepfake (manipuladas ou sintetizadas) — indicado para tarefas de classificação binária (real vs. fake) e experimentos de generalização.  
- **Origem / identificador:** Kaggle — `manjuts98/deepfake-and-real-images`  
- **Conteúdo típico:** Pares ou coleções de imagens reais e suas respectivas manipulações/deepfakes; pode conter subpastas por classe (real / fake) e metadados sobre método de síntese/forgery.  
- **Uso neste projeto:** Treino e validação de modelos para detecção de deepfakes com foco em desempenho entre diferentes fontes/fornecedores de manipulação.  

---

#### **🎯 Bases de Dados Utilizadas (até agora)**

##### Web of Science - Coleção Principal (Clarivate Analytics / Thomson Reuters)
- Provedor: Clarivate Analytics
- Resumo: Base multidisciplinar que indexa somente os periódicos mais citados em suas respectivas áreas. É também um índice de citações, informando, para cada artigo, os documentos por ele citados e os documentos que o citaram. Possui hoje mais de 9.000 periódicos indexados. É composta por:
  - Science Citation Index Expanded (SCI-EXPANDED): 1945 até o presente
  - Social Sciences Citation Index: 1956 até o presente
  - Arts and Humanities Citation Index: 1975 até o presente
  - A partir de 2012 o conteúdo foi ampliado com a inclusão do Conference Proceedings Citation Index - Science (CPCI-S) e Conference Proceedings Citation Index - Social Science & Humanities (CPCI-SSH).

---

##### IEEE Xplore Digital Library
- Provedor: Institute of Electrical and Electronic Engineers Incorporated (IEEE)
- Resumo: O Institute of Electrical and Electronic Engineers Incorporated (IEEE) é uma organização dedicada ao avanço da inovação e da excelência tecnológica para o benefício da humanidade que foi projetada para atender profissionais envolvidos em todos os aspectos dos campos elétrico, eletrônico, e de computação e demais áreas afins da ciência e tecnologia que fundamentam a civilização moderna. Criado em 1884, nos E.U.A., o IEEE congrega mais de 410.000 associados, entre engenheiros, cientistas, pesquisadores e outros profissionais, em cerca de 160 países. É composto por um Conselho de Diretores e por um Comitê Executivo que compreende 10 Regiões, 39 Sociedades Técnicas, 7 Conselhos Técnicos e aproximadamente 1850 Comitês Societários e 342 Seções. A coleção IEEE Xplore Digital Library inclui texto completo desde 1988. Oferece publicações de periódicos, normas técnicas e revistas em engenharia elétrica, computação, biotecnologia, telecomunicações, energia e dezenas de outras tecnologias. Além disso, o IEEE fornece acesso a mais de 6 milhões de documentos, incluindo artigos de pesquisa, normas técnicas, anais de congressos, tabelas e gráficos, conjunto de dados, artigos de transações internacionais, e-books, publicações de conferências, patentes e periódicos. Possui ainda a coleção IEEE Access, que é um periódico multidisciplinar, somente online, de acesso totalmente aberto (acesso gold), apresentando continuamente os resultados de pesquisa original ou desenvolvimento em todos os campos de interesse do IEEE. Apoiado por Taxa de Processamento de Artigo – APC, seus artigos são revisados por pares, a submissão para publicação é de 4 a 6 semanas e os artigos ficam disponíveis gratuitamente para todos os leitores. Possui fator de impacto próprio, pontos de influência do artigo e CiteScore estimados. Além de normas e proceedings, atualmente o Portal de Periódicos assina 239 períodicos para leitura e financia a publicação em 217 títulos disponibilizados pela IEEE, de modo que 163 instituições têm sido beneficiadas. O conteúdo atende as seguintes grandes áreas da tabela CAPES: Ciências Exatas e da Terra e Engenharias.

---

##### SCOPUS (Elsevier)
- Provedor: Reed Elsevier
- Resumo: Scopus is a comprehensive scientific, medical, technical and social science database containing all relevant literature.

---

##### ScienceDirect (Elsevier)
- Provedor: Elsevier
- Resumo: A ScienceDirect contém artigos de mais de 3.800 diários e mais de 37.000 títulos de livros, muitas de suas publicações são aprimoradas com elementos interativos fornecidos por autores, como áudio, vídeo, gráficos, tabelas e imagens. Os artigos também possuem links incorporados para conjuntos de dados externos, como Scopus®, PANGEA® e Reaxys®. Combinando esses extras de conteúdo com o texto de cada artigo e se obterá uma compreensão completa do panorama da informação antes de avançar seu trabalho. Estão disponíveis publicações cobrindo as áreas de Ciências Biológicas, Ciências da Saúde, Ciências Agrárias, Ciências Exatas e da Terra, Engenharias, Ciências Sociais Aplicadas, Ciências Humanas e Letras e Artes.

---

## 📚 **Base Teórica Fundamental**

#### **📄 Complexity-entropy causality plane as a complexity measure for two-dimensional patterns**
- **Autores:** Ribeiro, H. V.; Zunino, L.; Lenzi, E. K.; Santoro, P. A.; Mendes, R. S.
- **Ano:** 2012
- **Contribuição:** Fundamentação teórica do Plano CH para análise bidimensional de padrões
- **Aplicação:** Base matemática para extração de features F_CH

#### **📄 Distinguishing noise from chaos**
- **Autores:** Schreiber, T.; Schmitz, A.
- **Ano:** 1996
- **Contribuição:** Metodologia para separação de dinâmicas determinísticas e estocásticas
- **Aplicação:** Validação da Hipótese de Separação (H1)

#### **📄 Information Theory and Statistical Mechanics**
- **Autor:** Jaynes, E. T.
- **Ano:** 1957
- **Contribuição:** Princípio da máxima entropia para inferência estatística
- **Aplicação:** Inferência estatística objetiva sobre mídias sintéticas

#### **📄 How to conduct a systematic literature review: A narrative guide**
- **Autores:** Mengist, W.; Soromessa, T.; Legese, G.
- **Ano:** 2020
- **Contribuição:** Metodologia PICOC para revisão sistemática
- **Aplicação:** Estruturação da pesquisa bibliográfica

## 🔍 **PICOC: Implementação e Resultados**

Para estruturar sistematicamente a revisão da literatura, utilizaremos o protocolo **PICOC (Population, Intervention, Comparison, Outcomes, Context)**, que fornece um framework robusto para a formulação de questões de pesquisa e busca bibliográfica:

### **🎯 Population/Problem (Problema)**
- **Imagens e vídeos digitais** gerados por algoritmos de inteligência artificial
- **Mídias sintéticas** criadas ou alteradas por GANs, modelos de difusão e outras técnicas generativas
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
#### **🎯 Bases de Dados Utilizada (até agora)**
- **Web of Science:** Coleção Principal (1945-presente) - 9.000+ periódicos indexados
- **IEEE Xplore:** Biblioteca Digital completa (1988-presente) - 6M+ documentos
- **Scopus (Elsevier):** Base multidisciplinar abrangente
- **ScienceDirect:** 3.800+ periódicos e 37.000+ títulos de livros

---



#### **📄 Artigos Selecionados (Primeira avaliação em andamento)**

**Surveys e Reviews Fundamentais**

* Ahmed S.R. (2022): "Analysis Survey on Deepfake detection and Recognition with Convolutional Neural Networks" — *Hora 2022 (4th Int. Congress on Human Computer Interaction Optimization and Robotic Applications Proceedings)*
* Chennamma H.R. (2023): "A comprehensive survey on image authentication for tamper detection with localization" — *Multimedia Tools and Applications*
* Kadha V. (2025): "Unravelling Digital Forgeries: A Systematic Survey on Image Manipulation Detection and Localization" — *ACM Computing Surveys*
* Khan A.A. (2025): "A survey on multimedia-enabled deepfake detection: state-of-the-art tools and techniques, emerging trends, current challenges & limitations, and future directions" — *Discover Computing*
* Li C. (2025): "Survey on Technologies of Video Deepfake Detection" — *Lecture Notes on Data Engineering and Communications Technologies*
* Liang R. (2020): "A Survey of Audiovisual Deepfake Detection Techniques" — *Journal of Cyber Security*
* Nguyen T.T. (2022): "Deep learning for deepfakes creation and detection: A survey" — *Computer Vision and Image Understanding*
* Rana M.S. (2022): "Deepfake Detection: A Systematic Literature Review" — *IEEE Access*

**Métodos de Análise Temporal / Landmark & Motion**

* Liao X. (2023): "FAMM: Facial Muscle Motions for Detecting Compressed Deepfake Videos Over Social Networks" — *IEEE Transactions on Circuits and Systems for Video Technology*
* Sornavalli G. (2024): "DeepFake Detection by Prediction of Mismatch Between Audio and Video Lip Movement" — *ADICS 2024 (International Conference on Advances in Data Engineering and Intelligent Computing Systems)*
* Sharma H. (2021): "Video interframe forgery detection: Classification, technique & new dataset" — *Journal of Computer Security*
* Zhang Y. (2025): "Exploring coordinated motion patterns of facial landmarks for deepfake video detection" — *Applied Soft Computing*
* Zhu C. (2024): "Deepfake detection via inter-frame inconsistency recomposition and enhancement" — *Pattern Recognition*

**Abordagens de Análise de Frequência / Espacial–Frequencial**

* Frank J.; Eisenhofer T.; Schönherr L. (2020): "Leveraging Frequency Analysis for Deep Fake Image Recognition" — *ICML / PMLR* (referência clássica útil)
* Qiusong L. (2025): "Joint spatial-frequency deepfake detection network based on dual-domain attention-enhanced deformable convolution" — *Applied Intelligence*
* Shi Z. (2025): "Customized Transformer Adapter With Frequency Masking for Deepfake Detection" — *IEEE Transactions on Information Forensics and Security*

**Métodos Baseados em Teoria da Informação & Entropia**

* Sheng Z. (2025): "SUMI-IFL: An Information-Theoretic Framework for Image Forgery Localization with Sufficiency and Minimality Constraints" — *Proceedings of the AAAI Conference on Artificial Intelligence*
* Sudarsan M. (2025): "LEAD-AI: Lightweight Entropy Analysis for Distinguishing AI-Generated Images from Genuine Photographs" — *Proceedings of SPIE (The International Society for Optical Engineering)*
* Sun K. (2022): "An Information Theoretic Approach for Attention-Driven Face Forgery Detection" — *Lecture Notes in Computer Science*

**Arquiteturas Transformer / Vision Transformers / Adapters**

* Atamna M. (2025): "WaveConViT: Wavelet-Based Convolutional Vision Transformer for Cross-Manipulation Deepfake Video Detection" — *Lecture Notes in Computer Science*
* D. Zhang (2025): "DPL: Cross-Quality DeepFake Detection via Dual Progressive Learning" — *Lecture Notes in Computer Science*
* Li S. (2024): "UnionFormer: Unified-Learning Transformer with Multi-View Representation for Image Manipulation Detection and Localization" — *Proceedings of CVPR (IEEE Computer Society Conference on Computer Vision and Pattern Recognition)*

**Robustez, Generalização e Continual Learning**

* Bai N. (2025): "Towards generalizable face forgery detection via mitigating spurious correlation" — *Neural Networks*
* Sun K. (2025): "Continual Face Forgery Detection via Historical Distribution Preserving" — *International Journal of Computer Vision*
* Xu K. (2024): "RLGC: Reconstruction Learning Fusing Gradient and Content Features for Efficient Deepfake Detection" — *IEEE Transactions on Consumer Electronics*

**Localização de Forgeries / Detecção Forense e Otimização**

* Chen J. (2023): "Identification of image global processing operator chain based on feature decoupling" — *Information Sciences*
* Iseed S.Y. (2023): "Forensic approach for distinguishing between source and destination regions in copy-move forgery" — *Multimedia Tools and Applications*
* Joshi D. (2025): "Optimized detection and localization of copy-rotate-move forgeries using biogeography-based optimization algorithm" — *Journal of Forensic Sciences*
* Peng C. (2025): "Within 3DMM Space: Exploring Inherent 3D Artifact for Video Forgery Detection" — *IEEE Transactions on Information Forensics and Security*

**Modelos Leves, Codec Seguro & Consumer Electronics**

* Huang C.H. (2025): "A Secure Learned Image Codec for Authenticity Verification via Self-Destructive Compression" — *Big Data and Cognitive Computing*
* Jin Z. (2025): "Protecting Consumer Electronics Human-Computer Interactive Verification Security via Anomaly-Aware Reconstruction-Guided Forgery Localization" — *IEEE Transactions on Consumer Electronics*
* Sudarsan M. (2025): "LEAD-AI: Lightweight Entropy Analysis for Distinguishing AI-Generated Images from Genuine Photographs" — *Proceedings of SPIE*

**Multimodalidade (Áudio–Vídeo / Multimodal)**

* Das A.K. (2023): "A Multi-stage Multi-modal Classification Model for DeepFakes Combining Deep Learned and Computer Vision Oriented Features" — *Lecture Notes in Computer Science*
* Liu B. (2022): "Detecting Generated Images by Real Images" — *Lecture Notes in Computer Science*
* Sornavalli G. (2024): "DeepFake Detection by Prediction of Mismatch Between Audio and Video Lip Movement" — *ADICS 2024*

**Detecção baseada em Difusão / Latent Diffusion & Flags**

* Ricker J. (2024): "AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error" — *Proc. IEEE/CVPR (Conference on Computer Vision and Pattern Recognition)*
* Sun K. (2024): "DiffusionFake: Enhancing Generalization in Deepfake Detection via Guided Stable Diffusion" — *Advances in Neural Information Processing Systems (NeurIPS / NeurIPS Proceedings)*

**Trabalhos Empíricos, Heurísticos e Otimizações Algorítmicas**

* Meena K.B. (2021): "A deep learning based method for image splicing detection" — *Journal of Physics: Conference Series*
* Tripathi E. (2024): "An efficient digital image forgery detection using Pelican search optimisation-based DCNN" — *Journal of Experimental and Theoretical Artificial Intelligence*
* Tripathi E.; (outros trabalhos forenses/heurísticos relacionados) — (veja seção Forense & Otimização)

**Estudos Forenses & Processamento de Imagem (2023–2021)**

* Kadha V. (2023): "Forensic analysis of manipulation chains: A deep residual network for detecting JPEG-manipulation-JPEG" — *Forensic Science International: Digital Investigation*
* Chen J. (2023): "Identification of image global processing operator chain based on feature decoupling" — *Information Sciences*
* Hassan A. (2021): "Texture based Image Splicing Forgery Recognition using a Passive Approach" — *International Journal of Integrated Engineering*

**Conferências, Coleções e Misc**

* ICSPIS (2022): *2022 5th International Conference on Signal Processing and Information Security (ICSPIS 2022)* — (coleção de trabalhos relevantes)
* Blondé P. (2021): "In Medio Stat Virtus: intermediate levels of mind wandering improve episodic memory encoding in a virtual environment" — *Psychological Research*
* Wang Z. (2019): "Image forgery detection algorithm based on U-shaped detection network" — *Tongxin Xuebao Journal on Communications*

---


## ❓ **Questões de Pesquisa (QP)**

### **🔍 Questão Principal:**

**"Como a análise de complexidade-entropia pode aprimorar a detecção de mídias sintéticas em vídeos, superando as limitações de generalização dos métodos atuais baseados em deep learning?"**

### **📋 Questões Secundárias (QS)**

**QS1:** Quais são as assinaturas estatísticas distintivas de vídeos sintéticos no espaço complexidade-entropia comparadas às de vídeos autênticos?

**QS2:** Como a fusão de features de complexidade-entropia com representações de Vision Transformers impacta na capacidade de generalização cross-dataset?

**QS3:** Qual é a robustez das features baseadas em entropia de permutação contra degradações comuns (compressão, ruído) em vídeos?

**QS4:** Como os parâmetros de embedding (dx, dy) influenciam na separabilidade entre classes no Plano CH?

**QS5:** Qual é o trade-off entre interpretabilidade e performance dos detectores híbridos propostos comparados aos métodos estado-da-arte?

**QS6:** Como as características temporais dos vídeos deepfake se manifestam através da análise de séries temporais de complexidade-entropia?

### **📊 Critérios de Avaliação da Literatura (QA)**

#### **🔬 Rigor Metodológico**

**Q1:** O estudo reporta métricas de avaliação claras e apropriadas para a tarefa (ex: Acurácia, AUC-ROC, EER)?
- **Análise:** Avalia o **Rigor** e a **Qualidade do Relato**. Garante que os estudos utilizam métricas consolidadas, permitindo comparação justa e quantitativa.

**Q2:** O estudo utiliza datasets públicos e bem conhecidos para validação (ex: FaceForensics++, Celeb-DF)?
- **Análise:** Avalia a **Credibilidade** e o **Rigor**. O uso de datasets públicos é crucial para reprodutibilidade e validação em cenários reconhecidos pela comunidade científica.

**Q3:** O método proposto é comparado com pelo menos um outro método de detecção já existente (baseline)?
- **Análise:** Mede a **Relevância** e o **Rigor** do estudo. Sem comparação com baseline, é impossível aferir se a contribuição é de fato um avanço.

#### **🎯 Robustez e Aplicabilidade**

**Q4:** O estudo avalia a robustez do detector contra perturbações comuns (ex: compressão, ruído, variações de iluminação)?
- **Análise:** Diretamente ligada ao "Outcome" do PICOC (aumento da robustez). Avalia a **Credibilidade** e **Relevância** para aplicações no mundo real.

**Q5:** A metodologia proposta é descrita com detalhes suficientes para permitir a sua replicação?
- **Análise:** Avalia a **Reprodutibilidade**. Se um artigo não descreve claramente a metodologia, sua contribuição científica é limitada.

#### **📈 Credibilidade Científica**

**Q6:** Os autores discutem as limitações do estudo e as ameaças à validade dos resultados?
- **Análise:** O reconhecimento de limitações demonstra **maturidade acadêmica** e aumenta a credibilidade, indicando compreensão profunda do método.

**Q7:** Os objetivos da pesquisa, as contribuições e as questões de pesquisa do estudo estão claramente definidos?
- **Análise:** Garante que o artigo tem **foco claro** e contribuição bem definida, evitando trabalhos com escopo vago ou objetivos pouco claros.

#### **🎯 Avaliação Holística**
Estas 7 questões criam uma avaliação completa que analisa:
- **"O quê"** (resultados e métricas)
- **"Como"** (metodologia e relato)
- **"Por quê"** (relevância e limitações)

### **📊 Resultados Preliminares da Avaliação QA**

📋 **Arquivo Completo de Resultados:** [RESULTADOS_QA.md](docs/picoc/qa/RESULTADOS_QA.md)  
🔍 **Índice de Navegação:** [INDICE_ARTIGOS.md](docs/picoc/qa/INDICE_ARTIGOS.md)

> **💡 Navegação:** Clique nos links da tabela abaixo para acessar diretamente os PDFs dos artigos ou suas avaliações detalhadas. Cada artigo foi avaliado usando 7 critérios rigorosos de qualidade acadêmica.

#### **🏆 Artigos Aprovados (10/10 - 100%)**

| # | Artigo | Pontuação | PDF | Avaliação |
|---|--------|-----------|-----|-----------|
| 1 | **Customized Transformer Adapter With Frequency Masking** | 6.5/8.0 | [📄 PDF](docs/picoc/aprovados/8/Customized_Transformer_Adapter_With_Frequency_Masking_for_Deepfake_Detection.pdf) | [📊 QA](docs/picoc/qa/Customized%20Transformer%20Adapter%20With%20Frequency%20Masking%20for%20Deepfake%20Detection) |
| 2 | **Joint spatial-frequency deepfake detection network** | 6.5/8.0 | [📄 PDF](docs/picoc/aprovados/2/s10489-025-06761-2.pdf) | [📊 QA](docs/picoc/qa/Joint%20spatial-frequency%20deepfake%20detection%20network%20based%20on%20dual-domain%20attention-enhanced%20deformable%20convolution) |
| 3 | **Detecting face tampering in videos using deepfake forensics** | 6.5/8.0 | [📄 PDF](docs/picoc/aprovados/11/Detecting%20face%20tampering%20in%20videos%20using%20deepfake%20forensics.pdf) | [📊 QA](docs/picoc/qa/Detecting%20face%20tampering%20in%20videos%20using%20deepfake%20forensics) |
| 4 | **Unravelling Digital Forgeries: Systematic Survey** | 6.5/8.0 | [📄 PDF](docs/picoc/aprovados/3/Unravelling%20Digital%20Forgeries:%20A%20Systematic%20Survey%20on%20Image%20Manipulation%20Detection%20and%20Localization.pdf) | [📊 QA](docs/picoc/qa/Unravelling%20Digital%20Forgeries%20A%20Systematic%20Survey%20on%20Image%20Manipulation%20Detection%20and%20Localization) |
| 5 | **DPL: Cross-quality DeepFake Detection** | 6.0/8.0 | [📄 PDF](docs/picoc/aprovados/1/s10791-025-09550-0.pdf) | [📊 QA](docs/picoc/qa/DPL%20Cross-quality%20DeepFake%20Detection%20via%20Dual%20Progressive%20Learning) |
| 6 | **SUMI-IFL: Information-Theoretic Framework** | 6.0/8.0 | [📄 PDF](docs/picoc/aprovados/5/32054-Article%20Text-36122-1-2-20250410.pdf) | [📊 QA](docs/picoc/qa/SUMI-IFL%20An%20Information-Theoretic%20Framework%20for%20Image%20Forgery%20Localization%20with%20Sufficiency%20and%20Minimality%20Constraints) |
| 7 | **LEAD-AI: lightweight entropy analysis** | 5.0/8.0 | [📄 PDF](docs/picoc/aprovados/9/LEAD-AI_%20lightweight%20entropy%20analysis%20for%20distinguishing%20AI-generated%20images%20from%20genuine%20photographs.pdf) | [📊 QA](docs/picoc/qa/LEAD-AI%20lightweight%20entropy%20analysis%20for%20distinguishing%20AI-generated%20images%20from%20genuine%20photographs) |
| 8 | **Exploring coordinated motion patterns** | 4.5/8.0 | [📄 PDF](docs/picoc/aprovados/6/Exploring%20coordinated%20motion%20patterns%20of%20facial%20landmarks%20for%20deepfake%20video%20detection%20-%20ScienceDirect.pdf) | [📊 QA](docs/picoc/qa/Exploring%20coordinated%20motion%20patterns%20of%20facial%20landmarks%20for%20deepfake%20video%20detection) |
| 9 | **Markov Observation Models and Deepfakes** | 4.5/8.0 | [📄 PDF](docs/picoc/aprovados/4/mathematics-13-02128-v2.pdf) | [📊 QA](docs/picoc/qa/Markov%20Observation%20Models%20and%20Deepfakes) |
| 10 | **A survey on multimedia-enabled deepfake detection** | 4.5/8.0 | 📚 Survey | [📊 QA](docs/picoc/qa/A%20survey%20on%20multimedia-enabled%20deepfake%20detection%20state-of-the-art%20tools%20and%20techniques,%20emerging%20trends,%20current%20challenges%20&%20limitations,%20and%20future%20directions) |
| 11 | **Continual Face Forgery Detection via Historical Distribution Preserving** | 5.0/8.0 | [📄 PDF](docs/picoc/aprovados/7/Continual%20Face%20Forgery%20Detection%20via%20Historical%20Distribution%20Preserving.pdf) | [📊 QA](docs/picoc/qa/Continual%20Face%20Forgery%20Detection%20via%20Historical%20Distribution%20Preserving) |

#### **📈 Estatísticas Gerais**
- **Taxa de Aprovação:** 100% (10/10 artigos)
- **Pontuação Média:** 5.8/8.0
- **Melhor Desempenho:** Q1 e Q7 (100% de conformidade)
- **Área de Melhoria:** Q6 - Discussão de limitações (40% de conformidade)

---


## 🔧 **Ambiente de Desenvolvimento**

### **🐍 Ambiente Anaconda**
O projeto utiliza **Python** como linguagem principal, gerenciado através do **Anaconda** para garantir reprodutibilidade e isolamento de dependências.

#### **Instalação do Ambiente:**
```bash
# Criar ambiente conda para detecção de mídias sintéticas
conda create -n ia-product python=3.9 -y

# Ativar o ambiente
conda activate ia-product

# Instalar dependências principais
conda install numpy pandas matplotlib scikit-learn jupyter -y

# Instalar PyTorch (versão compatível com CUDA se disponível)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Instalar bibliotecas de visão computacional e ML
pip install opencv-python pillow transformers timm

# Instalar ordpy para análise de entropia de permutação
pip install ordpy

# Instalar bibliotecas adicionais para o projeto
pip install xgboost lightgbm seaborn plotly

# Verificar instalação
python -c "import torch, ordpy, cv2; print('Ambiente configurado com sucesso!')"
```


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
