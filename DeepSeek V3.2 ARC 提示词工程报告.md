

# **激发抽象推理能力：基于DeepSeek V3.2模型针对ARC基准的提示词工程策略技术报告**

## **引言**

在人工智能（AI）迈向通用人工智能（AGI）的征途中，如何衡量和提升机器的“流体智力”（Fluid Intelligence）已成为核心议题。流体智力，即在面对全新问题时，不依赖先验知识，仅通过少量示例快速学习、抽象规则并解决问题的能力，是人类认知能力的关键组成部分 1。为了量化这一能力，抽象与推理语料库（Abstraction and Reasoning Corpus, ARC）应运而生。它并非传统的基准测试，而是一个对当前AI范式，尤其是依赖大规模数据统计学习的大语言模型（LLM）的根本性挑战 3。

本次考核聚焦于利用提示词工程（Prompt Engineering）技术，驾驭DeepSeek V3.2这一前沿大语言模型，以解决ARC任务。这一挑战的核心矛盾在于：ARC任务的设计初衷是抵制模式匹配和记忆，而LLM的根本优势恰恰在于从海量数据中学习和复现模式 5。因此，成功的关键不在于模型本身蕴含的知识，而在于提示词工程师能否构建一个精密的“认知脚手架”（Cognitive Scaffold）。这个脚手架必须能引导一个本质上进行序列预测的模型，去执行一个它并未被明确训练过的、高度抽象的推理过程。

本报告旨在为参与者提供一份全面的技术和策略指南。报告将从三个维度展开：首先，深入剖析ARC任务的结构、认知基础及其对当前AI系统构成严峻挑战的根本原因；其次，详细解读本次考核指定工具——DeepSeek V3.2模型的架构特性，特别是其“非思考模式”(deepseek-chat)的关键约束；最后，系统性地调研并整理一系列针对抽象推理的提示词工程方法论。通过整合这三方面内容，本报告旨在揭示如何通过提示词设计，在模型与任务之间架起一座桥梁，从而有效激发其潜在的抽象推理能力。

---

## **第一部分：抽象与推理语料库（ARC）——通用智能的试金石**

本部分旨在为ARC基准提供一个基础性的、深入的理解，不仅关注其结构，更探讨其背后的哲学和认知科学原理，这些原理使其成为衡量AI推理能力的独特且困难的标尺。

### **1.1 ARC任务的解剖学分析**

ARC任务的核心在于通过极少的示例进行归纳学习，其结构简洁而要求严苛 1。

* **核心结构**：每个ARC任务由两部分组成：一个包含少量（通常为2-3个）“训练”样本（train）的集合，以及一个“测试”样本（test）1。每个样本都是一个输入-输出对（input-output pair）。数据本身以二维网格（grid）的形式呈现，网格由0到9的整数填充，代表10种不同的颜色，其尺寸在1x1到30x30之间可变 4。  
* **任务目标**：解题者的根本目标是从训练样本对中推理出隐藏的、通用的变换逻辑或规则，并将此规则应用于测试样本的输入网格，以生成正确的输出网格 1。这不仅要求预测网格的内容，还要求正确预测输出网格的尺寸。  
* **成功标准：精确匹配（Exact Match）**：ARC的评估标准极为严格。一个解答只有在预测的输出网格与标准答案在维度和每一个单元格的颜色上都完全一致时，才被判定为正确 1。这种“零容忍”的评分机制杜绝了任何近似解的可能性，对推理的精确性提出了极高的要求.5

### **1.2 ARC的认知基础：探测量“核心知识”**

ARC的设计理念超越了传统的机器学习基准，旨在直接测量智能体最本质的能力。

* **衡量流体智力**：ARC被明确设计为衡量“流体智力”的工具，即适应和解决新奇问题的能力，而非依赖于已习得的技能或知识（即“晶体智力”）1。这种能力是人类进行抽象思维和逻辑推理的基础。  
* **最小化先验知识**：任务的设计原则是，解题仅需依赖一套人类与生俱来或在幼儿时期普遍掌握的认知先验，即“核心知识”（Core Knowledge）6。这些核心知识包括：  
  * **物体恒存性（Objectness）**：将相邻的同色像素识别为连贯的“物体”。  
  * **目标导向性（Goal-directedness）**：理解变换是有目的的。  
  * **基本数感（Number Sense）**：进行简单的计数。  
  * **基础几何与拓扑概念**：理解对称、连接、包含、旋转等关系。  
* **独立于语言和文化**：ARC任务的一个关键设计原则是完全剥离对自然语言理解能力和特定文化背景知识的依赖 6。这使得ARC成为一个比绝大多数基于文本的基准更为普适的推理能力测试。也正因如此，作为语言大师的LLM在面对这种非语言的、纯粹的逻辑谜题时会遇到根本性的困难。

### **1.3 推理鸿沟：为何ARC至今仍是AI的巨大挑战**

ARC任务揭示了当前主流AI技术与人类智能之间存在的深刻鸿沟。

* **人类与AI的悬殊表现**：人类，即使未经特殊训练，也能轻松解决超过80%的ARC任务。相比之下，即便是最顶尖的通用大语言模型（如GPT-4）在未经特定调整的情况下，其准确率也通常低于20%。而专门为ARC设计的竞赛级系统，直到近期才突破50%的准确率大关 5。这一巨大的性能差距明确地指出了当前AI范式在抽象推理能力上的短板。  
* **统计模式匹配的失效**：LLM的核心能力是在海量数据中学习复杂的统计相关性。然而，ARC的每个任务都是独一无二的，其设计旨在抵抗记忆和简单的模式匹配 6。一个在互联网规模语料上训练的模型，几乎不可能在其训练数据中见过与特定ARC任务底层逻辑相匹配的文本描述。  
* **泛化能力的断层**：模型的核心失败模式在于无法从极少量的样本（Few-shot）中泛化出抽象的、底层的规则 11。LLM往往会过拟合训练样本网格的表面特征（例如，“把所有红色像素变成蓝色”），而无法推断出更深层次的、具有组合性的逻辑（例如，“将最大物体的颜色变为与最小物体的颜色一致”）11。  
* **求解范式的演进**：回顾ARC竞赛的历史，可以看到求解范式的清晰演进。早期的主流方法是基于领域特定语言（DSL）的暴力程序搜索 9。近年来，随着LLM的发展，最成功的策略转变为利用LLM来*指导*程序合成（一种归纳法，Induction），或者采用针对每个任务进行即时微调的测试时训练（Test-Time Training, TTT）技术 12。

这种演进本身揭示了一个关键问题：当前LLM更擅长*生成能解决问题的代码*（归纳），而不是*直接生成问题的解决方案*（转导，Transduction）。本次考核的规则——要求模型直接输出最终网格——恰恰强制参与者采用转导的方式，这正是LLM的薄弱环节。因此，提示词的核心作用，就是要引导一个转导模型去模拟归纳式的推理过程。这不仅是学习一个规则，更是学习一种*学习规则的方法*，即元学习（Meta-learning）。提示词必须隐式地教会模型如何面对全新的、小样本的学习任务，引导它启动抽象规则发现的“思维模式”。

#### **表1：ARC基准测试的性能对比**

为了直观地展示ARC任务的挑战性，下表汇总了不同实体在该基准上的性能表现。

| 系统/实体 | 在ARC上的报告准确率 (%) | 方法论/备注 |
| :---- | :---- | :---- |
| 人类基线 (平均) | \~$85\\%$ | 无需特殊训练，依赖直觉和核心知识 5。 |
| GPT-4 (示例) | \~$18\\%$ | 通用大语言模型，采用少样本提示 5。 |
| Claude (示例) | \~$15\\%$ | 通用大语言模型，在抽象和逻辑链上存在困难 5。 |
| Gemini (示例) | \~$20\\%$ | 通用大语言模型，在复杂抽象和一致性上表现不佳 5。 |
| ARC Prize 2024 获胜者 | $53.5\\%$ | 结合了LLM引导的程序合成与测试时训练的专门化系统 9。 |

---

## **第二部分：指定的工具：DeepSeek V3.2大语言模型**

本部分将从ARC的问题域转向本次考核指定的工具。报告将对DeepSeek V3.2模型进行深入的技术剖析，重点关注与本次任务直接相关的特性和限制。

### **2.1 架构基础与模型规模**

DeepSeek V3.2建立在一个强大且高效的架构之上，其规模和训练数据为其提供了广阔的能力基础。

* **核心架构：混合专家（MoE）**：DeepSeek-V3系列模型采用了混合专家（Mixture-of-Experts, MoE）架构 16。与传统的密集型（Dense）模型在处理每个输入时激活所有参数不同，MoE模型仅激活参数总量的一个子集。这种设计在保持巨大模型容量的同时，显著提高了推理效率 17。  
* **参数规模**：该模型总参数量高达6710亿（671B），但在处理每个token时，仅激活其中的370亿（37B）参数 16。这种规模为其提供了存储海量模式和知识的能力。  
* **训练数据**：模型在一个包含14.8万亿（trillion）token的庞大、高质量语料库上进行了预训练 16。这使得模型对自然语言、代码和各类结构化数据都有着深刻的理解。  
* **关键技术创新**：该架构集成了多项先进技术，如多头隐注意力（Multi-head Latent Attention, MLA）和在V3.2-Exp版本中首次应用的DeepSeek稀疏注意力（DeepSeek Sparse Attention, DSA）16。这些技术主要为提升长上下文处理效率而设计，虽然ARC任务的上下文不长，但这些底层的效率优化使得运行如此庞大的模型在实际应用中成为可能。

### **2.2 关键约束：“非思考”的deepseek-chat模式**

理解并遵守考核对模型使用模式的规定，是成功的先决条件。

* **两种截然不同的模式**：DeepSeek API提供了两种主要的模型调用模式：deepseek-reasoner和deepseek-chat 21。  
* **deepseek-reasoner（禁用的工具）**：此模式被明确设计用于处理复杂推理任务。它在生成最终答案之前，会先在内部生成一个思维链（Chain of Thought, CoT），并通过API响应中的reasoning\_content字段将这个思考过程暴露给用户，最终答案则位于content字段 21。**根据考核规则，此模式被明确禁止使用** 1。  
* **deepseek-chat（指定的工具）**：此模式对应模型的“非思考模式”（non-thinking mode）21。它直接对用户查询作出响应，不会在独立的字段中暴露任何中间推理步骤。这是一个标准的聊天补全（Chat Completion）接口，与OpenAI的API格式兼容，这也是考核要求的使用方式 1。  
* **约束的深层含义**：这是整个考核中最核心的一条规则。它意味着，激发、构建并捕捉模型推理过程的全部重担，都必须在提示词和模型生成的单一文本输出中完成。提示词工程师必须设法**强制一个“非思考”模式的模型，去“展示其思考过程”**。由于模型内部的CoT功能被禁用，提示词本身必须承担起外部“推理模块”的职责，通过精巧的结构设计，引导模型按部就班地执行一个deepseek-reasoner模式本可以自动完成的思考流程。

### **2.3 可控性与随机性：API参数分析**

考核对API参数的特定设置，为我们选择何种策略提供了重要线索。

* **可用参数**：deepseek-chat模式支持标准的API参数，如temperature, top\_p, max\_tokens等 22。这一点与deepseek-reasoner模式形成鲜明对比，后者不支持temperature和top\_p等控制采样随机性的参数 23。  
* **temperature=1.0的强制规定**：考核统一规定temperature参数值必须设为1.0 1。temperature控制模型输出的随机性，值越高，输出越具多样性和“创造性”；值越低，输出越确定和集中 22。1.0是一个相对较高的值。  
* **高temperature的策略启示**：对于像ARC这样要求精确、确定性输出的逻辑任务，设置高temperature似乎是反直觉的，因为它会增加输出的随机性，导致结果难以复现。然而，这一看似矛盾的设定，恰恰是某些高级提示词策略（如“自洽性”，Self-Consistency）得以有效实施的关键。该策略依赖于模型在多次运行时生成多样化的推理路径。因此，考核方强制设定temperature=1.0，很可能是在强烈暗示：单一的、确定性的解法并非最优，一个鲁棒的系统需要通过多次采样并聚合结果来对抗随机性，从而提升最终答案的可靠性。

#### **表2：DeepSeek V3.2 API模式功能对比**

下表清晰地对比了两种API模式的关键区别，突出了本次考核的核心限制。

| 特性 | deepseek-chat (非思考模式) | deepseek-reasoner (思考模式) |
| :---- | :---- | :---- |
| **主要用途** | 通用对话、直接问答 | 复杂推理、分步问题求解 |
| **输出结构** | 单一content字段，包含最终答案 | reasoning\_content (思维链) 和 content (最终答案) 两个字段 |
| **暴露推理步骤** | 否（需在提示词中引导） | 是（通过reasoning\_content字段） |
| **支持的采样参数** | 支持 temperature, top\_p 等 | 不支持 temperature, top\_p 等 |
| **考核许可** | **允许且唯一指定** | **禁止** |

---

## **第三部分：提示词工程的艺术与科学：面向抽象推理**

本部分将综合前两部分的分析——ARC的挑战与DeepSeek模型的特性——系统性地探讨如何运用提示词工程技术，来应对本次考核的独特难题。

### **3.1 基础策略：为ARC构建结构化的上下文学习**

在引导模型进行推理之前，首要任务是将非语言的、二维的网格问题，转化为模型能够理解的、一维的文本格式。

* **网格的文本化表示**：将二维视觉信息序列化为一维文本是第一步，也是至关重要的一步。常见的表示方法包括：  
  * **嵌套列表**：例如，\[, \]。这种格式结构清晰，与编程语言兼容，但可能削弱模型对角线或跨行空间关系的感知。  
  * **ASCII艺术**：用字符直接绘制网格。这种方式在视觉上更直观，可能更好地保留了空间邻近性，但会消耗更多token。  
  * 逐行描述：例如，“Row 1: 0, 1\. Row 2: 2, 0.”。这种方式较为冗长，但对模型的指令遵循能力要求较低。  
    选择何种表示法并非无足轻重，它直接影响模型“看见”和理解空间模式的能力。鉴于LLM在空间推理上存在固有弱点，这一选择本身就是提示词工程的关键环节 10。  
* **零样本提示（Zero-Shot Prompting）**：即只提供任务指令和测试输入，不提供任何已完成的示例 28。考虑到ARC任务的高度复杂性，单纯的零样本提示几乎不可能成功，但它是构建更复杂提示的基础。  
* **少样本提示（Few-Shot Prompting）**：这是ARC任务设置的核心 1。提示词中必须包含考核数据提供的2-3个“训练”样本对，作为上下文中的示例，然后才给出“测试”输入 28。这些示例的组织方式至关重要。采用清晰的定界符（如XML标签 \<example\>, \<input\>, \<output\>）来结构化地分隔每个训练样本的输入和输出，以及最后的测试输入，是提升模型理解能力的关键最佳实践 30。

### **3.2 在“非思考”模型中诱导隐式推理**

由于deepseek-reasoner被禁用，我们必须通过提示词本身的设计，来诱导deepseek-chat模型进行并展示其推理过程。

* **零样本思维链（Zero-Shot CoT）**：这是最简单的推理激发技术。在提示的末尾附加一句“魔法咒语”，如“让我们一步一步地思考”（Let's think step by step），可以有效促使模型在给出最终答案前，先生成一段分析和推理过程 32。这相当于在单一的输出文本块内，模拟了CoT的行为。  
* **结构化推理提示**：一种更稳定、更可控的方法是将提示词设计成一个多阶段的指令模板 35。这种提示结构将复杂的推理任务分解为一系列更小的、更易于管理的子任务，其逻辑如同科学研究方法：  
  1. **观察（Observation）**：清晰地呈现所有训练样本。“这里有几个输入-输出网格的示例，它们共同演示了一个隐藏的变换规则。”  
  2. **假设（Hypothesis）**：明确要求模型归纳规则。“你的第一个任务是仔细分析这些示例，然后用自然语言清晰地描述出这个变换规则是什么。”  
  3. **实验（Experimentation）**：指令模型应用其假设。“你的第二个任务是，将你上面总结出的规则，一步一步地应用到下面的测试输入网格上。请详细展示你的应用过程。”  
  4. 结论（Conclusion）：要求模型给出最终格式化的输出。“最后，将经过变换得到的最终输出网格，以嵌套列表的格式呈现。”  
     这种结构化提示强迫模型将其内部的思考过程外部化，使其推理路径变得透明和可调试。  
* **元提示与角色扮演**：为模型设定一个专家角色，例如“你是一位精通解决抽象视觉推理谜题的专家”，可以有效地启动模型处理此类任务所需的认知框架 37。元提示（Meta-Prompting）则更进一步，引导模型反思问题本身的结构，例如，“要解决这类问题，通常需要注意哪些类型的特征（如对称性、物体数量、颜色变化等）？” 28。

### **3.3 提升鲁棒性与探索能力的高级框架**

为了应对ARC任务的多样性和temperature=1.0带来的随机性，需要采用更高级的策略。

* **自洽性（Self-Consistency）**：该技术是应对temperature=1.0这一强制设置的最直接、最有效的策略 28。其核心思想是，对同一个ARC任务，使用相同的提示词调用模型多次（例如5-10次）。由于高temperature的存在，模型每次可能会生成略有不同的推理路径和最终答案。最后，通过对所有生成的答案进行“投票”，选择出现次数最多的那个作为最终提交的答案。这种集成方法能够显著平滑随机性带来的噪声，大幅提升在复杂推理任务上的准确率和稳定性。  
* **程序辅助/代码化思维**：尽管模型无法在当前环境中执行代码，但可以利用其强大的代码生成和理解能力，引导它像程序员一样思考 38。提示词可以要求模型将变换规则描述为一系列简单的、确定性的伪代码或算法步骤。例如：“1. 遍历输入网格，找到所有颜色为红色的独立对象。2. 计算每个红色对象的像素数量。3. 找到像素数量最多的红色对象。4. 在输出网格的相同位置，将该对象的所有像素颜色变为蓝色。” 这种方式利用了模型在结构化、逻辑化表达上的优势，来规范其解决非代码问题的推理过程。  
* **概念化思维树（Tree of Thoughts, ToT）**：虽然无法在单次API调用中实现一个完整的ToT框架，但其核心思想——探索多种可能性并进行评估剪枝——可以在提示词中进行模拟 38。可以这样设计提示：“请提出三种不同的、可能解释训练样本的变换规则。对于每一种规则，请评估它与所有训练样本的匹配程度。最后，选择你认为最 plausible 的规则，并将其应用于测试输入。” 这种方法鼓励模型进行更广泛的假设空间搜索，避免其过早地锁定在第一个想到的、但可能是错误的思路上。

---

## **第四部分：综合与应考策略建议**

本部分将前述的理论分析转化为具体、可操作的策略，为参与者在实现construct\_prompt和parse\_output函数时提供指导。

### **4.1 ARC任务的提示词策略手册**

针对不同类型的ARC任务，可以采用不同的提示词架构。

* **基线提示（The Baseline Prompt）**：这是一个稳健的起点，融合了多项最佳实践。它应包括：使用XML标签清晰界定的少样本示例、明确的角色设定（如“抽象推理专家”），以及一个零样本CoT指令（“让我们一步一步地思考”）。  
* **“规则优先”提示（The "Rule-First" Prompt）**：适用于那些逻辑看似复杂、具有多步组合性的任务。此提示将采用3.2节中描述的“科学方法”结构化模板，强制模型在应用规则之前，必须先用自然语言清晰地陈述它所归纳出的规则。  
* **“程序化”提示（The "Programmatic" Prompt）**：适用于涉及物体操作（移动、旋转、缩放）、计数或几何变换的任务。此提示将引导模型以伪代码或算法步骤的形式进行思考，如3.3节所述。  
* **自洽性包装器（The Self-Consistency Wrapper）**：这并非一种提示词，而是一种元策略。它意味着在主运行逻辑中，将上述任何一种提示策略包装在一个循环里，执行k次API调用，然后对返回的k个结果进行多数投票，以确定最终答案。这是对抗temperature=1.0随机性的主要武器。

#### **表3：针对不同ARC任务类型的提示词策略决策矩阵**

下表提供了一个快速参考指南，帮助根据任务特点选择最合适的提示策略。

| ARC任务原型 | 基线少样本提示 | “规则优先”CoT提示 | “程序化”提示 | 自洽性包装器 |
| :---- | :---- | :---- | :---- | :---- |
| **物体移动/缩放/旋转** | 可行 | 推荐 | **强烈推荐** | 总是推荐 |
| **颜色填充/映射** | **推荐** | 可行 | 可行 | 总是推荐 |
| **模式重复/分形** | 可行 | **推荐** | 可行 | 总是推荐 |
| **对称/翻转** | 可行 | 可行 | **推荐** | 总是推荐 |
| **基于计数的逻辑** | 不推荐 | **推荐** | **强烈推荐** | 总是推荐 |

### **4.2 construct\_prompt函数的架构设计**

该函数是提示词工程的核心实现。

* **动态生成**：函数不应使用单一的静态字符串作为提示。它应该根据传入的字典d动态地构建整个提示内容。  
* **网格序列化模块**：建议创建一个独立的辅助函数，如grid\_to\_text(grid)，负责将网格的列表形式转换为选定的文本表示（如嵌套列表字符串或ASCII艺术）。这样做便于对不同的表示法进行快速实验和切换。  
* **示例格式化**：函数需要遍历d\['train'\]列表，将每一个训练样本的输入和输出网格，通过序列化模块转换后，包装在清晰的定界符（如\<train\_input\>...\</train\_input\>）内，并拼接到主提示字符串中。  
* **指令头部**：提示的开头应包含一个明确的系统级指令（{"role": "system",...}），用于设定模型的角色、定义任务的总体目标以及规定最终输出的格式。  
* **构建最终测试案例**：提示的结尾部分应包含格式化后的测试输入d\['test'\]\['input'\]，并附上最终的行动指令，例如“现在，请应用你推断的规则，生成对应的输出网格。”

### **4.3 parse\_output函数的健壮性工程**

该函数负责从模型自由生成的文本中，精确地提取出结构化的答案。

* **处理可变性**：由于temperature=1.0和CoT风格的提示，模型的输出将是解释性文本和最终网格的混合体。解析器绝不能假设输出只有网格本身。  
* **使用正则表达式进行鲁棒提取**：Python的re模块是实现该功能的最佳工具。需要设计一个健壮的正则表达式，用于在长文本中定位并提取出Python风格的嵌套列表字符串，例如 \\\[\\s\*\\\[\\s\*\\d+\\s\*(?:,\\s\*\\d+\\s\*)\*\\s\*\\\]\\s\*(?:,\\s\*\\\[\\s\*\\d+\\s\*(?:,\\s\*\\d+\\s\*)\*\\s\*\\\]\\s\*)\*\\s\*\\\]。  
* **错误处理**：函数必须能优雅地处理各种异常情况。如果在输出文本中没有找到有效的网格，应返回一个空列表或约定的错误标识。对于格式错误的网格（例如，各行长度不一），应尝试进行修复或直接判定为解析失败。  
* **类型转换**：从文本中提取出的数字是字符串形式，解析器必须将它们全部转换为整数，以满足函数list\[list\[int\]\]的返回类型要求。

## **结论**

本次基于DeepSeek V3.2模型解决ARC任务的考核，本质上是对提示词工程师综合能力的深度检验。其核心挑战源于一个根本性的矛盾：用一个为语言模式识别而生的工具，去解决一个为抵抗模式识别而设计的抽象推理问题。

本报告的分析表明，通往成功的路径并非依赖于模型的“灵光一现”，而是建立在一套系统性的工程方法之上。成功的策略必须认识到并主动去弥补ARC任务对抽象泛化能力的要求与LLM对统计模式的依赖之间的鸿沟。

最终，制胜策略可归结为两大支柱：

1. **认知脚手架的构建**：通过精心设计的、结构化的提示词，为“非思考”的deepseek-chat模型强加一个外部的、显式的推理框架。该框架应模仿科学的探究过程——观察、假设、实验、结论——迫使模型将其推理过程透明化，从而提高逻辑的严谨性和结果的准确性。  
2. **随机性的管理**：通过采用自洽性（Self-Consistency）等多样本集成方法，主动利用并最终克服考核中temperature=1.0这一强制性随机因素带来的挑战。这承认了单次推理的不可靠性，并通过群体智慧的方式来逼近确定性的正确答案。

综上所述，本次考核将提示词工程从一门“玄学”推向了一门严谨的、为人工智能心智搭建认知辅助工具的系统性学科。最终的胜出者，将是那些能够最深刻地理解模型局限性，并能最创造性地通过提示词来拓展其能力边界的工程师。

#### **Works cited**

1. 提示词工程考试说明（2025年秋）.pdf  
2. Generalized Planning for the Abstraction and Reasoning Corpus, accessed October 23, 2025, [https://ojs.aaai.org/index.php/AAAI/article/view/29996/31747](https://ojs.aaai.org/index.php/AAAI/article/view/29996/31747)  
3. The Abstraction and Reasoning Corpus (ARC) \- Kaggle, accessed October 23, 2025, [https://www.kaggle.com/datasets/tunguz/the-abstraction-and-reasoning-corpus-arc](https://www.kaggle.com/datasets/tunguz/the-abstraction-and-reasoning-corpus-arc)  
4. fchollet/ARC-AGI: The Abstraction and Reasoning Corpus \- GitHub, accessed October 23, 2025, [https://github.com/fchollet/ARC-AGI](https://github.com/fchollet/ARC-AGI)  
5. The ARC Benchmark: Evaluating LLMs' Reasoning Abilities, accessed October 23, 2025, [https://graphlogic.ai/blog/utilities/the-arc-benchmark-evaluating-llms-reasoning-abilities/](https://graphlogic.ai/blog/utilities/the-arc-benchmark-evaluating-llms-reasoning-abilities/)  
6. ARC-AGI-2: A New Challenge for Frontier AI Reasoning Systems \- arXiv, accessed October 23, 2025, [https://arxiv.org/html/2505.11831v1](https://arxiv.org/html/2505.11831v1)  
7. About ARC – Lab42, accessed October 23, 2025, [https://lab42.global/arc/](https://lab42.global/arc/)  
8. ARC-AGI Benchmark: Fluid AI Evaluation \- Emergent Mind, accessed October 23, 2025, [https://www.emergentmind.com/topics/arc-agi-benchmark](https://www.emergentmind.com/topics/arc-agi-benchmark)  
9. ARC Prize 2024: Technical Report \- arXiv, accessed October 23, 2025, [https://arxiv.org/html/2412.04604v1](https://arxiv.org/html/2412.04604v1)  
10. Why ARC-AGI is not Proof that Models are incapable of Reasoning : r/OpenAI \- Reddit, accessed October 23, 2025, [https://www.reddit.com/r/OpenAI/comments/1g8a1pw/why\_arcagi\_is\_not\_proof\_that\_models\_are\_incapable/](https://www.reddit.com/r/OpenAI/comments/1g8a1pw/why_arcagi_is_not_proof_that_models_are_incapable/)  
11. Line Goes Up? Inherent Limitations of Benchmarks for Evaluating Large Language Models, accessed October 23, 2025, [https://arxiv.org/html/2502.14318v1](https://arxiv.org/html/2502.14318v1)  
12. Product of Experts with LLMs: Boosting Performance on ARC Is a Matter of Perspective, accessed October 23, 2025, [https://arxiv.org/html/2505.07859v2](https://arxiv.org/html/2505.07859v2)  
13. Easy Problems That LLMs Get Wrong \- arXiv, accessed October 23, 2025, [https://arxiv.org/html/2405.19616v1](https://arxiv.org/html/2405.19616v1)  
14. Daily Papers \- Hugging Face, accessed October 23, 2025, [https://huggingface.co/papers?q=Abstract%20Reasoning%20Corpus](https://huggingface.co/papers?q=Abstract+Reasoning+Corpus)  
15. Boosting Performance on ARC is a Matter of Perspective \- arXiv, accessed October 23, 2025, [https://arxiv.org/html/2505.07859v1](https://arxiv.org/html/2505.07859v1)  
16. DeepSeek-V3 Technical Report \- arXiv, accessed October 23, 2025, [https://arxiv.org/pdf/2412.19437](https://arxiv.org/pdf/2412.19437)  
17. deepseek-ai/DeepSeek-V3 \- GitHub, accessed October 23, 2025, [https://github.com/deepseek-ai/DeepSeek-V3](https://github.com/deepseek-ai/DeepSeek-V3)  
18. DeepSeek-V3 Technical Report \- ResearchGate, accessed October 23, 2025, [https://www.researchgate.net/publication/387512415\_DeepSeek-V3\_Technical\_Report](https://www.researchgate.net/publication/387512415_DeepSeek-V3_Technical_Report)  
19. Introducing DeepSeek-V3.2-Exp, accessed October 23, 2025, [https://api-docs.deepseek.com/news/news250929](https://api-docs.deepseek.com/news/news250929)  
20. DeepSeek-v3.2-Exp \- GitHub, accessed October 23, 2025, [https://github.com/deepseek-ai/DeepSeek-V3.2-Exp](https://github.com/deepseek-ai/DeepSeek-V3.2-Exp)  
21. Change Log | DeepSeek API Docs, accessed October 23, 2025, [https://api-docs.deepseek.com/updates](https://api-docs.deepseek.com/updates)  
22. Create Chat Completion \- DeepSeek API Docs, accessed October 23, 2025, [https://api-docs.deepseek.com/api/create-chat-completion](https://api-docs.deepseek.com/api/create-chat-completion)  
23. Reasoning Model (deepseek-reasoner), accessed October 23, 2025, [https://api-docs.deepseek.com/guides/reasoning\_model](https://api-docs.deepseek.com/guides/reasoning_model)  
24. DeepSeek-V3.2-Exp: A Guide With Demo Project \- DataCamp, accessed October 23, 2025, [https://www.datacamp.com/tutorial/deepseek-v-3-2-exp](https://www.datacamp.com/tutorial/deepseek-v-3-2-exp)  
25. deepseek-ai/DeepSeek-V3.1 \- Hugging Face, accessed October 23, 2025, [https://huggingface.co/deepseek-ai/DeepSeek-V3.1](https://huggingface.co/deepseek-ai/DeepSeek-V3.1)  
26. DeepSeek API Docs: Your First API Call, accessed October 23, 2025, [https://api-docs.deepseek.com/](https://api-docs.deepseek.com/)  
27. Exploring Model Parameters in DeepSeek | CodeSignal Learn, accessed October 23, 2025, [https://codesignal.com/learn/courses/creating-a-personal-tutor-with-deepseek-in-python/lessons/exploring-model-parameters-in-deepseek-1](https://codesignal.com/learn/courses/creating-a-personal-tutor-with-deepseek-in-python/lessons/exploring-model-parameters-in-deepseek-1)  
28. Prompt Engineering Techniques | IBM, accessed October 23, 2025, [https://www.ibm.com/think/topics/prompt-engineering-techniques](https://www.ibm.com/think/topics/prompt-engineering-techniques)  
29. The Few Shot Prompting Guide \- PromptHub, accessed October 23, 2025, [https://www.prompthub.us/blog/the-few-shot-prompting-guide](https://www.prompthub.us/blog/the-few-shot-prompting-guide)  
30. Include few-shot examples | Generative AI on Vertex AI \- Google Cloud, accessed October 23, 2025, [https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/few-shot-examples](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/prompts/few-shot-examples)  
31. Zero-Shot, One-Shot, and Few-Shot Prompting, accessed October 23, 2025, [https://learnprompting.org/docs/basics/few\_shot](https://learnprompting.org/docs/basics/few_shot)  
32. Advanced Prompt Engineering Techniques \- Mercity AI, accessed October 23, 2025, [https://www.mercity.ai/blog-post/advanced-prompt-engineering-techniques](https://www.mercity.ai/blog-post/advanced-prompt-engineering-techniques)  
33. Chain of Thought Prompting Guide \- Medium, accessed October 23, 2025, [https://medium.com/@dan\_43009/chain-of-thought-prompting-guide-3fdfd1972e03](https://medium.com/@dan_43009/chain-of-thought-prompting-guide-3fdfd1972e03)  
34. Chain-of-Thought (CoT) Prompting \- Prompt Engineering Guide, accessed October 23, 2025, [https://www.promptingguide.ai/techniques/cot](https://www.promptingguide.ai/techniques/cot)  
35. I reverse-engineered ChatGPT's "reasoning" and found the 1 prompt pattern that makes it 10x smarter : r/PromptEngineering \- Reddit, accessed October 23, 2025, [https://www.reddit.com/r/PromptEngineering/comments/1mjhdk8/i\_reverseengineered\_chatgpts\_reasoning\_and\_found/](https://www.reddit.com/r/PromptEngineering/comments/1mjhdk8/i_reverseengineered_chatgpts_reasoning_and_found/)  
36. Foundations of Prompt Engineering: From Structured Problems to Efficient Prompts | by Luis Alberto Cruz | Medium, accessed October 23, 2025, [https://medium.com/@luicruz/foundations-of-prompt-engineering-from-structured-problems-to-efficient-prompts-32e8e7430418](https://medium.com/@luicruz/foundations-of-prompt-engineering-from-structured-problems-to-efficient-prompts-32e8e7430418)  
37. What is Prompt Engineering? A Detailed Guide For 2025 \- DataCamp, accessed October 23, 2025, [https://www.datacamp.com/blog/what-is-prompt-engineering-the-future-of-ai-communication](https://www.datacamp.com/blog/what-is-prompt-engineering-the-future-of-ai-communication)  
38. Prompting Techniques | Prompt Engineering Guide, accessed October 23, 2025, [https://www.promptingguide.ai/techniques](https://www.promptingguide.ai/techniques)