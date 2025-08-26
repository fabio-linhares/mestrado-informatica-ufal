# Relatório Executivo da RSL — Detecção de Mídias Sintéticas

## Contexto geral

- Foco da RSL: detecção de mídias sintéticas (deepfakes) com ênfase em métodos de Teoria da Informação (complexidade‑entropia, robustez, interpretabilidade) comparados a abordagens de Aprendizado Profundo (CNN/ViT/Transformers), no contexto de segurança digital e forense multimídia.
- Panorama atual: conjunto de estudos recentes (2024–2025) cobrindo vídeos e imagens, incluindo propostas com atenção em frequência, landmarks faciais, aprendizado contínuo, formulações teóricas e duas revisões amplas do estado‑da‑arte.

---

## Artigos incluídos e em análise

### 1) Joint spatial-frequency deepfake detection network based on dual-domain attention-enhanced deformable convolution (Applied Intelligence, 2025)
- Resumo: propõe uma rede para detecção de deepfakes que integra informações espacial e de frequência, com convolução deformável aprimorada por atenção em dois domínios. Reporta métricas como ACC/Precision/Recall/AUC e compara contra vários SOTA (e.g., Xception, F3‑Net). Avalia também compressão (FF++ c23) e generalização cross‑dataset.
- Palavras‑chave: Deepfake; DeformableConv; Attention; Frequency; Dual‑domain; Robustez.
- Relevância PICOC: forte em Comparison/Outcome (robustez e generalização), alinhado ao eixo “frequência + visão profunda”.
- Links: [PDF](../aprovados/2/s10489-025-06761-2.pdf) · [QA](../qa/Joint%20spatial-frequency%20deepfake%20detection%20network%20based%20on%20dual-domain%20attention-enhanced%20deformable%20convolution)

### 2) Exploring coordinated motion patterns of facial landmarks for deepfake video detection (Applied Soft Computing, 2025)
- Resumo: introduz um método baseado em padrões coordenados de movimento de landmarks faciais (LTDRM), integrável a backbones de vídeo (CNN, Mamba, Transformer). Avaliado em FF++, Celeb‑DF e DFDC, com AUC e melhorias sobre os modelos originais.
- Palavras‑chave: Deepfake detection; Facial landmark; Motion patterns; Vídeo; AUC; Benchmarks.
- Relevância PICOC: endereça Comparison com vários backbones e Outcome (acurácia e generalização entre datasets).
- Links: [PDF](../aprovados/6/Exploring%20coordinated%20motion%20patterns%20of%20facial%20landmarks%20for%20deepfake%20video%20detection%20-%20ScienceDirect.pdf) · [QA](../qa/Exploring%20coordinated%20motion%20patterns%20of%20facial%20landmarks%20for%20deepfake%20video%20detection)

### 3) SUMI-IFL: An Information-Theoretic Framework for Image Forgery Localization with Sufficiency and Minimality Constraints (AAAI, 2025)
- Resumo: framework de base teórica para localização de falsificações em imagens com restrições de suficiência e minimalidade (informação). Valida em múltiplos datasets (e.g., DEFACTO‑12, CASIAv2, NIST16), compara com 7 métodos SOTA e inclui avaliação de robustez (JPEG, desfoque).
- Palavras‑chave: Teoria da Informação; Localização de falsificação; Suficiência/Minimalidade; Robustez; AUC; F1.
- Relevância PICOC: centro da RSL (Intervention de Teoria da Informação), cobrindo Outcome (robustez, interpretabilidade) e Comparison (SOTA).
- Links: [PDF](../aprovados/5/32054-Article%20Text-36122-1-2-20250410.pdf) · [QA](../qa/SUMI-IFL%20An%20Information-Theoretic%20Framework%20for%20Image%20Forgery%20Localization%20with%20Sufficiency%20and%20Minimality%20Constraints)

### 4) Markov Observation Models and Deepfakes (Mathematics, 2025)
- Resumo: estudo matemático com modelos de observação de Markov (MOM) e cadeias Markov por trechos (PMC), com detalhamento de derivações e EM. A validação usa sequências sintéticas (cara/coroa) em vez de datasets públicos de deepfake; ainda assim, apresenta replicabilidade rigorosa.
- Palavras‑chave: HMM; MOM; PMC; EM; Modelagem probabilística; Teoria; Replicabilidade.
- Relevância PICOC: contribui no eixo teórico/interpretável (Intervention – modelagem estatística), com ressalva em Q2 (datasets).
- Links: [Bib](../aprovados/4/Markov%20Observation%20Models%20and%20Deepfakes.bib) · [QA](../qa/Markov%20Observation%20Models%20and%20Deepfakes)

### 5) Unravelling Digital Forgeries: A Systematic Survey on Image Manipulation Detection and Localization (ACM Computing Surveys, 2025)
- Resumo: survey abrangente sobre manipulação de imagens e localização, cobrindo datasets, métricas, processos de pós‑processamento (JPEG, reamostragem), limitações e tendências. Útil como mapa da área e fonte de comparação metódica.
- Palavras‑chave: Image manipulation; Forensics; Datasets; Métricas; Robustez; Survey.
- Relevância PICOC: consolida Comparison (múltiplos métodos), Context (forense) e Outcome (discussão de robustez/limitações).
- Links: [PDF](../aprovados/3/Unravelling%20Digital%20Forgeries:%20A%20Systematic%20Survey%20on%20Image%20Manipulation%20Detection%20and%20Localization.pdf) · [QA](../qa/Unravelling%20Digital%20Forgeries%20A%20Systematic%20Survey%20on%20Image%20Manipulation%20Detection%20and%20Localization)

### 6) Continual Face Forgery Detection via Historical Distribution Preserving (IJCV, 2025)
- Resumo: abordagem de aprendizado contínuo para detecção de falsificações faciais preservando a distribuição histórica, visando reduzir esquecimento catastrófico e manter desempenho sob mudança de distribuição. Foco em cenários dinâmicos de geração de deepfakes.
- Palavras‑chave: Continual learning; Face forgery; Distribution shift; Preservação de distribuição; Generalização temporal.
- Relevância PICOC: forte em Outcome (generalização e robustez ao longo do tempo); amplia o escopo da RSL para evolução contínua dos geradores.
- Links: [PDF](../aprovados/7/Continual%20Face%20Forgery%20Detection%20via%20Historical%20Distribution%20Preserving.pdf) · [QA](../qa/Continual%20Face%20Forgery%20Detection%20via%20Historical%20Distribution%20Preserving)

### 7) Customized Transformer Adapter With Frequency Masking for Deepfake Detection (IEEE TIFS, 2025)
- Resumo: propõe um adapter para Transformers com máscara de frequência, visando melhorar discriminação de artefatos e generalização; repositório de código mencionado. Alinha análise espectral e arquitetura ViT/Transformers.
- Palavras‑chave: Deepfakes; Frequency‑domain analysis; Feature masking; Transformers; Adapter; Generalização.
- Relevância PICOC: Intervention híbrida (domínio de frequência em deep learning) e Outcome (generalização/explicabilidade parcial via frequência).
- Links: [PDF](../aprovados/8/Customized_Transformer_Adapter_With_Frequency_Masking_for_Deepfake_Detection.pdf) · [QA](../qa/Customized%20Transformer%20Adapter%20With%20Frequency%20Masking%20for%20Deepfake%20Detection)

### 8) A survey on multimedia-enabled deepfake detection… (Discover Computing, 2025)
- Resumo: revisão de ferramentas e técnicas de detecção multimídia de deepfakes, com tendências, desafios e lacunas; útil para consolidar terminologia, taxonomias e linhas futuras.
- Palavras‑chave: Deepfakes; Multimedia Systems; ML/DL; Forensics; Vídeo.
- Relevância PICOC: apoia Context/Comparison; referência transversal para framing da RSL.
- Links: [PDF](../aprovados/1/s10791-025-09550-0.pdf) · [QA](../qa/A%20survey%20on%20multimedia-enabled%20deepfake%20detection%20state-of-the-art%20tools%20and%20techniques,%20emerging%20trends,%20current%20challenges%20&%20limitations,%20and%20future%20directions)

### 9) Detecting face tampering in videos using deepfake forensics (Multimedia Tools and Applications, 2025)
- Resumo sintético (com base em metadados): estudo focado em detecção forense de adulteração facial em vídeo, possivelmente combinando pistas espácio‑temporais/forenses clássicas com avaliação quantitativa. Texto completo está disponível no repositório.
- Palavras‑chave sugeridas: Deepfake forensics; Face tampering; Vídeo; Pistas espácio‑temporais; Avaliação quantitativa.
- Relevância PICOC: reforça Comparison em vídeo; potencial para Outcome (robustez), a verificar na leitura integral.
- Links: [PDF](../aprovados/11/Detecting%20face%20tampering%20in%20videos%20using%20deepfake%20forensics.pdf)

### 10) DPL: Cross-quality DeepFake Detection via Dual Progressive Learning (arXiv, 2024)
- Resumo sintético (com base no título): método para detecção cross‑quality com dupla aprendizagem progressiva, mirando robustez a diferentes qualidades de compressão/ruído.
- Palavras‑chave sugeridas: Cross‑quality; Progressive learning; Robustez; Generalização; Deepfake detection.
- Relevância PICOC: foca Outcome (robustez a qualidade variada), alinhado ao objetivo de generalização.
- Links: [arXiv](../aprovados/10/DPL%3A%20Cross-quality%20DeepFake%20Detection%20via%20Dual%20Progressive%20Learning.bib) · [QA](../qa/DPL%20Cross-quality%20DeepFake%20Detection%20via%20Dual%20Progressive%20Learning)

### 11) LEAD-AI: lightweight entropy analysis for distinguishing AI-generated images (2025)
- Resumo sintético: abordagem leve de análise de entropia para distinguir imagens IA vs. reais; provável conexão com medidas de Teoria da Informação/complexidade.
- Palavras‑chave sugeridas: Entropia; Teoria da Informação; Imagens geradas por IA; Leve/eficiente; Detecção binária.
- Relevância PICOC: diretamente no eixo Intervention da RSL (complexidade/entropia).
- Links: [PDF](../aprovados/9/LEAD-AI_%20lightweight%20entropy%20analysis%20for%20distinguishing%20AI-generated%20images%20from%20genuine%20photographs.pdf) · [QA](../qa/LEAD-AI%20lightweight%20entropy%20analysis%20for%20distinguishing%20AI-generated%20images%20from%20genuine%20photographs)

---

## Próximos passos sugeridos
- Consolidar JSONs de QA restantes e padronizar evidências (Q1–Q7) com citações.
- Tabela unificada de robustez (cross‑dataset, compressão forte, ruído) e generalização.
- Marcação explícita dos trabalhos por eixo: Teoria da Informação vs. Deep Learning vs. Híbridos.
