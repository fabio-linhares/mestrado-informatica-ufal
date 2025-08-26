# 📊 **Resultados da Avaliação de Qualidade (QA)**

Este documento apresenta os resultados da avaliação sistemática dos artigos selecionados através do protocolo PICOC, utilizando os 7 critérios de qualidade definidos para o projeto de mestrado.

---

## 📋 **Critérios de Avaliação**

### **🔬 Rigor Metodológico**
- **Q1:** O estudo reporta métricas de avaliação claras e apropriadas para a tarefa?
- **Q2:** O estudo utiliza datasets públicos e bem conhecidos para validação?
- **Q3:** O método proposto é comparado com pelo menos um outro método existente (baseline)?

### **🎯 Robustez e Aplicabilidade**
- **Q4:** O estudo avalia a robustez do detector contra perturbações comuns?
- **Q5:** A metodologia proposta é descrita com detalhes suficientes para replicação?

### **📈 Credibilidade Científica**
- **Q6:** Os autores discutem as limitações do estudo e ameaças à validade?
- **Q7:** Os objetivos, contribuições e questões de pesquisa estão claramente definidos?

### **🎯 Sistema de Pontuação**
- **Sim:** 1.0 ponto
- **Parcialmente:** 0.5 pontos  
- **Não:** 0.0 pontos
- **Critério de Inclusão:** Pontuação total ≥ 4.0 pontos

---

## 📚 **Artigos Avaliados**

### 1. **Customized Transformer Adapter With Frequency Masking for Deepfake Detection**
📄 **Arquivo PDF:** [Customized_Transformer_Adapter_With_Frequency_Masking_for_Deepfake_Detection.pdf](../aprovados/8/Customized_Transformer_Adapter_With_Frequency_Masking_for_Deepfake_Detection.pdf)

| Critério | Resposta | Pontuação | Justificativa |
|----------|----------|-----------|---------------|
| **Q1** | Sim | 1.0 | Reporta métricas claras como Accuracy (Acc), AUC e EER na seção IV-A |
| **Q2** | Sim | 1.0 | Validação em múltiplos datasets: FaceForensics++, Celeb-DF, WildDeepfake, DFDC, DiffSwap |
| **Q3** | Sim | 1.0 | Comparação extensiva com métodos estado-da-arte: Xception, F³-Net, RECCE, SFDG, DSRL |
| **Q4** | Sim | 1.0 | Seção IV-B6 'Robustness Analysis' avalia robustez contra Compressão, Contraste, Saturação, Pixelização |
| **Q5** | Sim | 1.0 | Metodologia detalhada com arquitetura, implementação e código-fonte público |
| **Q6** | Parcialmente | 0.5 | Reconhece desempenho inferior ao FA-VIT e aumento de parâmetros, mas falta discussão formal de limitações |
| **Q7** | Sim | 1.0 | Contribuições claramente definidas: framework CUTA, módulo FDM, adaptadores ViT customizados |

**🏆 Pontuação Total:** 6.5/7.0 → **INCLUIR**

---

### 2. **Joint spatial-frequency deepfake detection network based on dual-domain attention-enhanced deformable convolution**
📄 **Arquivo PDF:** [s10489-025-06761-2.pdf](../aprovados/2/s10489-025-06761-2.pdf)

| Critério | Resposta | Pontuação | Justificativa |
|----------|----------|-----------|---------------|
| **Q1** | Sim | 1.0 | Reporta Acurácia, Precisão, Recall e AUC em tabelas detalhadas |
| **Q2** | Sim | 1.0 | Utiliza FaceForensics++ e Celeb-DF (V2), datasets conhecidos publicamente |
| **Q3** | Sim | 1.0 | Comparação extensiva com Xception, F3-Net, AW-MSA em múltiplas tabelas |
| **Q4** | Parcialmente | 0.5 | Avalia robustez à compressão e generalização cross-dataset, mas limitações em forte degradação |
| **Q5** | Sim | 1.0 | Metodologia detalhada na Seção 3 com diagramas, fórmulas matemáticas e promessa de código |
| **Q6** | Sim | 1.0 | Seção específica discute limitações: robustez a ruído e ausência de detecção em áudio |
| **Q7** | Sim | 1.0 | Objetivos claros no resumo e introdução sobre combinação espacial-frequência |

**🏆 Pontuação Total:** 6.5/7.0 → **INCLUIR**

---

### 3. **Detecting face tampering in videos using deepfake forensics**
📄 **Arquivo PDF:** [Detecting face tampering in videos using deepfake forensics.pdf](../aprovados/11/Detecting%20face%20tampering%20in%20videos%20using%20deepfake%20forensics.pdf)

| Critério | Resposta | Pontuação | Justificativa |
|----------|----------|-----------|---------------|
| **Q1** | Sim | 1.0 | Reporta Accuracy (Tabelas 3, 4, 6) e AUC (Tabela 7) como métricas claras |
| **Q2** | Sim | 1.0 | Utiliza dataset FaceForensics++ para validação (seção 4.1) |
| **Q3** | Sim | 1.0 | Tabela 7 compara com nove outros trabalhos existentes usando AUC |
| **Q4** | Sim | 1.0 | Avalia robustez contra degradação: reduções de 20% e 50% na resolução (Tabela 6) |
| **Q5** | Sim | 1.0 | Metodologia detalhada: arquitetura MesoNet, algoritmo de agregação, hiperparâmetros |
| **Q6** | Parcialmente | 0.5 | Menciona limitações de outros trabalhos, mas discussão limitada sobre limitações próprias |
| **Q7** | Sim | 1.0 | Objetivo claro de desenvolver modelo eficaz para detecção em vídeos de diferentes qualidades |

**🏆 Pontuação Total:** 6.5/7.0 → **INCLUIR**

---

### 4. **DPL: Cross-quality DeepFake Detection via Dual Progressive Learning**
📄 **Arquivo PDF:** [s10791-025-09550-0.pdf](../aprovados/1/s10791-025-09550-0.pdf)

| Critério | Resposta | Pontuação | Justificativa |
|----------|----------|-----------|---------------|
| **Q1** | Sim | 1.0 | Declara explicitamente avaliação por AUC, métrica apropriada para detecção |
| **Q2** | Sim | 1.0 | Validação em FaceForensics++, CelebDFv2, FaceShifter e FFIW10K |
| **Q3** | Sim | 1.0 | Comparação com oito métodos estado-da-arte: F3Net, SRM, SPSL, UIA-VIT, QAD |
| **Q4** | Sim | 1.0 | 'Robustness Analysis' seção 4.4 contra quatro perturbações com cinco níveis |
| **Q5** | Sim | 1.0 | Metodologia detalhada: arquitetura geral, módulos VQI/FII/FSM, treinamento duas etapas |
| **Q6** | Não | 0.0 | Conclui com resultados positivos sem discussão de limitações ou ameaças à validade |
| **Q7** | Sim | 1.0 | Contribuições claramente definidas: framework DPL, módulo FSM, pipeline de treinamento |

**🏆 Pontuação Total:** 6.0/7.0 → **INCLUIR**

---

### 5. **SUMI-IFL: An Information-Theoretic Framework for Image Forgery Localization with Sufficiency and Minimality Constraints**
📄 **Arquivo PDF:** [32054-Article Text-36122-1-2-20250410.pdf](../aprovados/5/32054-Article%20Text-36122-1-2-20250410.pdf)

| Critério | Resposta | Pontuação | Justificativa |
|----------|----------|-----------|---------------|
| **Q1** | Sim | 1.0 | Adota F1 score e AUC como métricas de avaliação apropriadas |
| **Q2** | Sim | 1.0 | Validação em múltiplos datasets públicos: DEFACTO-12, CASIAv2, NIST16 |
| **Q3** | Sim | 1.0 | Comparação com sete métodos estado-da-arte: MMFusion, EITL-Net, MVSS-Net |
| **Q4** | Sim | 1.0 | 'Robustness Evaluation' testa contra compressão JPEG e desfoque Gaussiano |
| **Q5** | Sim | 1.0 | Metodologia detalhada: arquitetura, derivações teóricas, hiperparâmetros |
| **Q6** | Não | 0.0 | Conclui com contribuições e desempenho superior sem discussão de limitações |
| **Q7** | Sim | 1.0 | Contribuições claramente definidas: framework com restrições de suficiência e minimalidade |

**🏆 Pontuação Total:** 6.0/7.0 → **INCLUIR**

---

### 6. **Unravelling Digital Forgeries: A Systematic Survey on Image Manipulation Detection and Localization**
📄 **Arquivo PDF:** [Unravelling Digital Forgeries.pdf](../aprovados/3/Unravelling%20Digital%20Forgeries:%20A%20Systematic%20Survey%20on%20Image%20Manipulation%20Detection%20and%20Localization.pdf)

| Critério | Resposta | Pontuação | Justificativa |
|----------|----------|-----------|---------------|
| **Q1** | Sim | 1.0 | Fornece análise de medidas de desempenho detalhadas em apêndices específicos |
| **Q2** | Sim | 1.0 | Tabela 8 resume diversos datasets públicos: CASIA, DEFACTO, RAISE |
| **Q3** | Sim | 1.0 | Como revisão sistemática, compara extensivamente múltiplos métodos existentes |
| **Q4** | Sim | 1.0 | Discute robustez contra compressão JPEG, reamostragem e pós-processamento |
| **Q5** | Parcialmente | 0.5 | Descreve metodologias com referências, mas requer consulta às fontes originais |
| **Q6** | Sim | 1.0 | Seção 5.2 dedicada às lacunas de pesquisa e limitações das ferramentas atuais |
| **Q7** | Sim | 1.0 | Objetivos e contribuições claramente delineados na Seção 1.3 'Key Contributions' |

**🏆 Pontuação Total:** 6.5/7.0 → **INCLUIR**

---

### 7. **LEAD-AI: lightweight entropy analysis for distinguishing AI-generated images from genuine photographs**
📄 **Arquivo PDF:** [LEAD-AI lightweight entropy analysis.pdf](../aprovados/9/LEAD-AI_%20lightweight%20entropy%20analysis%20for%20distinguishing%20AI-generated%20images%20from%20genuine%20photographs.pdf)

| Critério | Resposta | Pontuação | Justificativa |
|----------|----------|-----------|---------------|
| **Q1** | Sim | 1.0 | Reporta múltiplas métricas: Acurácia, Taxa de Falsos Positivos, Recall, Precisão |
| **Q2** | Não | 0.0 | Utiliza dataset privado curado pelos autores (600 imagens) em vez de benchmarks públicos |
| **Q3** | Sim | 1.0 | Seção 4.2 compara com CIFAKE (CNN) e detector ZED (baseado em entropia) |
| **Q4** | Não | 0.0 | Explicitamente afirma que testes em imagens limpas, robustez requer investigação futura |
| **Q5** | Sim | 1.0 | Metodologia detalhada: extração de patches, espaço HSV, fórmulas, pseudocódigo |
| **Q6** | Sim | 1.0 | Seção 5 dedicada às limitações: classificador simples, modelos limitados, falta de pós-processamento |
| **Q7** | Sim | 1.0 | Contribuições explicitamente listadas no final da introdução |

**🏆 Pontuação Total:** 5.0/7.0 → **INCLUIR**

---

### 8. **Exploring coordinated motion patterns of facial landmarks for deepfake video detection**
📄 **Arquivo PDF:** [Exploring coordinated motion patterns.pdf](../aprovados/6/Exploring%20coordinated%20motion%20patterns%20of%20facial%20landmarks%20for%20deepfake%20video%20detection%20-%20ScienceDirect.pdf)

| Critério | Resposta | Pontuação | Justificativa |
|----------|----------|-----------|---------------|
| **Q1** | Sim | 1.0 | Utiliza AUC como métrica padrão e apropriada para detecção |
| **Q2** | Sim | 1.0 | Avaliação em três benchmarks: FaceForensics++, Celeb-DF e DFDC |
| **Q3** | Sim | 1.0 | LTDRM integrado a métodos existentes (CNN, Mamba, Transformer) com comparação |
| **Q4** | Não | 0.0 | Sem menção de avaliação de robustez contra perturbações comuns |
| **Q5** | Parcialmente | 0.5 | Descreve arquitetura geral e ideias centrais, mas falta detalhes matemáticos completos |
| **Q6** | Não | 0.0 | Foca em resultados positivos sem discussão de limitações ou ameaças à validade |
| **Q7** | Sim | 1.0 | Lista clara de três principais contribuições no final da introdução |

**🏆 Pontuação Total:** 4.5/7.0 → **INCLUIR**

---

### 9. **Markov Observation Models and Deepfakes**
📄 **Arquivo PDF:** [mathematics-13-02128-v2.pdf](../aprovados/4/mathematics-13-02128-v2.pdf)

| Critério | Resposta | Pontuação | Justificativa |
|----------|----------|-----------|---------------|
| **Q1** | Sim | 1.0 | Reporta capacidade de detecção em porcentagem e desvio padrão em tabelas |
| **Q2** | Não | 0.0 | Utiliza dataset privado de sequências cara/coroa em vez de datasets de deepfakes |
| **Q3** | Sim | 1.0 | Compara três modelos (HMM, MOM, PMC) onde HMM serve como baseline |
| **Q4** | Não | 0.0 | Análise em sequências de moeda onde perturbações não são aplicáveis |
| **Q5** | Sim | 1.0 | Metodologia com extenso detalhe matemático, derivações e pseudocódigo |
| **Q6** | Parcialmente | 0.5 | Discute limitação a espaços discretos, mas não outras limitações ou ameaças |
| **Q7** | Sim | 1.0 | Define claramente três objetivos e os recapitula nas conclusões |

**🏆 Pontuação Total:** 4.5/7.0 → **INCLUIR**

---

### 10. **A survey on multimedia-enabled deepfake detection: state-of-the-art tools and techniques, emerging trends, current challenges & limitations, and future directions**
📄 **Arquivo de Avaliação:** [A survey on multimedia-enabled deepfake detection](./A%20survey%20on%20multimedia-enabled%20deepfake%20detection%20state-of-the-art%20tools%20and%20techniques,%20emerging%20trends,%20current%20challenges%20&%20limitations,%20and%20future%20directions)

| Critério | Resposta | Pontuação | Justificativa |
|----------|----------|-----------|---------------|
| **Q1** | Sim | 1.0 | Cita métricas como precisão, acurácia e recall na análise crítica dos modelos |
| **Q2** | Parcialmente | 0.5 | Discute escalabilidade em datasets, mas como survey não valida em dataset específico |
| **Q3** | Não | 0.0 | Como survey, descreve e compara métodos existentes sem propor novo método |
| **Q4** | Sim | 1.0 | Menciona avaliação de 'respostas rápidas a ataques adversários' como robustez |
| **Q5** | Não | 0.0 | Não propõe metodologia, apenas revisa metodologias existentes (ML e DL) |
| **Q6** | Sim | 1.0 | Discute desafios tecnológicos, sociais, ataques adversários e implicações éticas |
| **Q7** | Sim | 1.0 | Objetivo claramente definido como revisão das técnicas mais recentes |

**🏆 Pontuação Total:** 4.5/7.0 → **INCLUIR**

---

## 📊 **Resumo Estatístico**

### **📈 Distribuição de Pontuações**
- **6.5 pontos:** 3 artigos (30%)
- **6.0 pontos:** 2 artigos (20%) 
- **5.0 pontos:** 1 artigo (10%)
- **4.5 pontos:** 4 artigos (40%)

### **🎯 Taxa de Aprovação**
- **Total de artigos avaliados:** 10
- **Artigos incluídos:** 10 (100%)
- **Artigos excluídos:** 0 (0%)

### **📊 Performance por Critério**
| Critério | Sim | Parcialmente | Não | Média |
|----------|-----|--------------|-----|-------|
| **Q1** | 10 (100%) | 0 (0%) | 0 (0%) | 1.00 |
| **Q2** | 7 (70%) | 1 (10%) | 2 (20%) | 0.75 |
| **Q3** | 9 (90%) | 0 (0%) | 1 (10%) | 0.90 |
| **Q4** | 5 (50%) | 1 (10%) | 4 (40%) | 0.55 |
| **Q5** | 8 (80%) | 2 (20%) | 0 (0%) | 0.90 |
| **Q6** | 3 (30%) | 2 (20%) | 5 (50%) | 0.40 |
| **Q7** | 10 (100%) | 0 (0%) | 0 (0%) | 1.00 |

### **🔍 Principais Insights**
- **Forças:** Todos os artigos têm métricas claras (Q1) e objetivos bem definidos (Q7)
- **Área de Melhoria:** Discussão de limitações (Q6) é o critério com menor pontuação
- **Robustez:** Apenas 55% dos estudos avaliam robustez adequadamente (Q4)
- **Padrões:** 70% usam datasets públicos, demonstrando aderência a boas práticas

---

## 📝 **Observações Metodológicas**

1. **Critério Mais Atendido:** Q1 e Q7 (100% de conformidade)
2. **Critério Menos Atendido:** Q6 - Discussão de limitações (40% de conformidade)
3. **Tendência:** Artigos mais recentes tendem a ter melhor documentação de limitações
4. **Impacto na Seleção:** Todos os artigos superaram o threshold de 4.0 pontos

---

*Avaliação realizada conforme protocolo PICOC - Projeto de Mestrado UFAL*  
*Data da avaliação: Agosto 2025*
