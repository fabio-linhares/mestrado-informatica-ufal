# Sobre
Repositório para armazenar estudos, projetos e materiais relacionados ao Mestrado em Informática na Universidade Federal de Alagoas (UFAL). Inclui códigos-fonte, documentos, apresentações e outros recursos desenvolvidos durante o curso.

---  

# 🎓 **Projeto de Mestrado**  
## Universidade Federal de Alagoas (UFAL) - Instituto de Computação  
### Programa de Pós-Graduação em Informática  

## 📌 **Título**  
**Detecção de Vídeos Criados ou Alterados por Inteligência Artificial através de Técnicas Avançadas de Visão Computacional e Aprendizado de Máquina**  

👨‍🎓 **Aluno:** Fábio Sant’Anna Linhares  
👩‍🏫 **Orientadora:** Prof.ª Dr.ª Fabiane da Silva Queiroz  
🔬 **Linha de Pesquisa:** Computação Visual e Inteligente  

---

## 📝 **Resumo**  

A crescente disseminação de **mídias sintéticas**, criadas ou alteradas por Inteligência Artificial (IA), representa um grande desafio para a **segurança da informação** e a **veracidade dos conteúdos online**. A capacidade dessas tecnologias de gerar vídeos altamente realistas e convincentes torna difícil distinguir conteúdos autênticos de falsificações, levando a implicações sociais, políticas e jurídicas.  

Este projeto busca aprimorar as técnicas de **detecção de vídeos manipulados** por IA, combinando **visão computacional** e **aprendizado de máquina**. A metodologia baseia-se na abordagem proposta por **Rafique et al. (2023)**, que utiliza **Análise de Nível de Erro (ELA) e Redes Neurais Convolucionais (CNNs)** para identificar alterações digitais. No entanto, nosso trabalho avança essa abordagem ao integrar técnicas de **análise de textura** e regras especializadas de **detecção forense**.  

O objetivo é desenvolver uma estrutura de detecção capaz de superar a **precisão de 89,5%** dos métodos existentes, contribuindo significativamente para a mitigação dos desafios impostos pelas mídias sintéticas.  

---

## 🎯 **Objetivos do Projeto**  

O projeto visa desenvolver uma estrutura eficiente para **detecção e classificação de mídias sintéticas em vídeos**, utilizando técnicas de visão computacional e aprendizado de máquina.  

### 🔹 **Objetivos Específicos**  
✔ **Identificar e classificar técnicas de detecção** de mídias sintéticas.  
✔ **Compilar e combinar múltiplos métodos** para aprimorar a detecção.  
✔ **Utilizar CNNs pré-treinadas** (GoogLeNet, ResNet18, SqueezeNet) para extrair padrões visuais de vídeos manipulados.  
✔ **Implementar Análise de Nível de Erro (ELA)** para identificar regiões manipuladas digitalmente.  
✔ **Testar diversos algoritmos de classificação** (SVM, Random Forest, XGBoost, MLP, entre outros) para avaliar a eficácia das técnicas.  
✔ **Aprimorar a precisão de detecção**, superando os 89,5% alcançados pelos métodos anteriores.  

---

## 📚 **Justificativa**  

O avanço das tecnologias de geração de vídeos sintéticos levou à criação de deepfakes **quase indistinguíveis da realidade**. Segundo **Rodrigues et al. (2024)**, esses conteúdos têm sido usados para manipulação política, disseminação de desinformação e crimes cibernéticos.  

Estudos como o de **Vahdati et al. (2024)** e **Xu et al. (2024)** demonstram que detectores atuais são **menos eficazes para vídeos do que para imagens**. Modelos tradicionais de CNNs são eficientes na detecção de deepfakes em imagens, mas falham na análise de padrões temporais presentes em vídeos.  

Além disso, segundo **Pei et al. (2024)**, os **modelos de difusão** emergiram como uma tecnologia revolucionária para geração de deepfakes, exigindo o aprimoramento dos métodos de detecção. Este projeto propõe uma abordagem inovadora que combina **análise de padrões texturais, ELA e redes neurais profundas**, aumentando a robustez do processo de detecção.  

---

## 🛠 **Metodologia**  

A pesquisa será baseada em um conjunto de **etapas experimentais**, conforme descrito abaixo:  

### **1️⃣ Seleção de Dados**  
- Uso de bases públicas, como **FaceForensics++ (FF++)**, contendo vídeos reais e manipulados.  
- Aplicação de pré-processamento para padronizar a qualidade dos vídeos.  

### **2️⃣ Extração de Características**  
- Uso de **CNNs pré-treinadas (GoogLeNet, ResNet18, SqueezeNet)** para extrair características de alto nível.  
- Implementação da **Análise de Nível de Erro (ELA)** para identificar padrões anômalos.  
- Aplicação de **análise de textura** para capturar artefatos gerados por IA.  

### **3️⃣ Treinamento de Modelos**  
- Treinamento de modelos de classificação como **SVM, Random Forest, XGBoost, LightGBM, MLP**.  
- Otimização dos hiperparâmetros para melhorar a precisão e reduzir falsos positivos.  

### **4️⃣ Validação e Avaliação**  
- Testes com conjuntos de dados independentes para validar a eficácia do modelo.  
- Uso de métricas como **precisão, recall e F1-score** para avaliação do desempenho.  

---

## 📈 **Resultados Esperados**  

Espera-se que o modelo desenvolvido atinja **uma precisão superior a 89,5%**, fornecendo uma abordagem mais eficaz para a **detecção de deepfakes em vídeos**. Os principais benefícios incluem:  

✔ **Aprimoramento da detecção** de mídias sintéticas com novas técnicas.  
✔ **Criação de um framework híbrido** combinando diferentes métodos de análise.  
✔ **Contribuição para a segurança da informação**, reduzindo a disseminação de vídeos falsificados.  
✔ **Publicação dos resultados** em conferências e periódicos científicos.  

---

## 🔬 **Referências**  

📄 **Rafique, R. et al. (2023).** *Deep Fake Detection and Classification Using Error-Level Analysis and Deep Learning.* Scientific Reports. Disponível em: [https://doi.org/10.1038/s41598-023-34629-3](https://doi.org/10.1038/s41598-023-34629-3)  

📄 **Rodrigues, G. S. et al. (2024).** *Uma Abordagem a DeepFake via Algoritmos de Aprendizagem Profunda.* Anais do ENCOMPIF. Disponível em: [https://sol.sbc.org.br/index.php/encompif/article/view/25238](https://sol.sbc.org.br/index.php/encompif/article/view/25238)  

📄 **Pei, G. et al. (2024).** *Deepfake Generation and Detection: A Benchmark and Survey.* arXiv. Disponível em: [https://arxiv.org/pdf/2403.17881](https://arxiv.org/pdf/2403.17881)  

📄 **Vahdati, D. S. et al. (2024).** *Beyond Deepfake Images: Detecting AI-Generated Videos.* CVPR 2024 Workshops. Disponível em: [https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Vahdati](https://openaccess.thecvf.com/content/CVPR2024W/WMF/papers/Vahdati)  

📄 **Xu, S. et al. (2024).** *VASA-1: Lifelike Audio-Driven Talking Faces Generated in Real Time.* arXiv. Disponível em: [https://arxiv.org/pdf/2404.10667](https://arxiv.org/pdf/2404.10667)  

---

## 📬 **Contato**  
📩 **E-mail:** fl@ic.ufal.br  
🔗 **LinkedIn:** [linkedin.com/in/fabio-linhares](https://www.linkedin.com/in/fabio-linhares)  
🐙 **GitHub:** [github.com/fabio-linhares](https://github.com/fabio-linhares)  

