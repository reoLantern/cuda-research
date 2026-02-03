# (2024-07-03) Control Flow Management in Modern GPUs

<table><tbody><tr><td><p><strong>作者:</strong> Mojtaba Abaie Shoushtary; Jordi Tubella Murgadas; Antonio Gonzalez;</p></td></tr><tr><td><p><strong>期刊:</strong></p></td></tr><tr><td><p><strong>期刊分区:</strong></p></td></tr><tr><td><p><strong>本地链接: </strong><a href="zotero://open-pdf/0_VF6MTBN4" rel="noopener noreferrer nofollow">2024_Control Flow Management in Modern GPUs_Mojtaba Abaie Shoushtary_arXiv.pdf</a></p></td></tr><tr><td><p><strong>DOI: </strong><a href="https://doi.org/10.48550/arXiv.2407.02944" rel="noopener noreferrer nofollow">10.48550/arXiv.2407.02944</a></p></td></tr><tr><td><p><strong>摘要: </strong>在 GPU 中，control flow management mechanism 决定了在任意时间点一个 warp 中哪些线程处于活动状态。<br>该机制会监控 warp 内的 scalar threads 的 control flow，以优化线程调度，并在执行资源利用上发挥关键作用。<br>该 mechanism 可以通过软件经由 instructions 进行控制或辅助。<br><br>然而，GPU vendors 并不公开其 compiler、ISA 或硬件实现的细节。<br>这种不透明性使研究者难以理解 control flow management mechanism 的具体工作方式、实现路径，及其如何被软件辅助——尤其当它对研究结果具有显著影响时。<br><br>这种不透明性同样为 GPU 的性能建模带来问题：研究者只能依赖来自 real hardware 的 control flow traces，无法对该 mechanism 的功能进行建模或修改，从而难以探究“改变机制本身”所带来的影响。<br><br>本文通过对多种 benchmarks 的实验数据进行分析，给出了 Turing native ISA 中 control flow instructions 的一个合理语义定义（plausible semantic）。<br>在此基础上，我们提出了一种低开销（low-cost）的高效 control flow management 机制，命名为 **Hanoi**。<br><br>Hanoi 在保证正确性的同时，生成的 control flow 与 real hardware 非常接近。<br>我们的评估显示：real hardware 的 control flow trace 与 Hanoi 机制之间的差异平均仅为 1.03%。<br>进一步地，将采用 Hanoi 的 GPUs 与实际硬件原生的 control flow management 在 Instructions Per Cycle (IPC) 上进行对比，平均差异仅为 0.19%。</p></td></tr><tr><td><p><strong>笔记日期: </strong>2025/9/8 16:30:41</p></td></tr></tbody></table>

## 📜 研究核心

> Tips: 做了什么，解决了什么问题，创新点与不足？

### ⚙️ 内容

### 💡 创新点

### 🧩 不足

## 🤔 Introduction

GPU（Graphics Processing Units）采用 Single Instruction Multiple Threads (SIMT) [23] 架构，在 Single Instruction Multiple Data (SIMD) 处理单元上同时执行多条线程。  
在该架构中，SIMD lanes 对来自不同线程的不同操作数执行相同的运算。  
线程调度（thread scheduling）对 SIMD 利用率以及此类架构的整体性能影响显著，因为只有被调度的线程才能占用 SIMD lanes。

GPU 采用两类机制来调度线程执行：  
a) 将线程分组为称为 warps 的集合 [23], [33]，并在每个 cycle 选择一个 warp 执行 [36]；  
b) 在任意时间点确定一个 warp 中哪些线程处于 active 状态。

我们将后者称为 control flow management，主要因为它决定每个 warp 的 control flow，并且高度受各个线程自身 control flow 的影响。  
control flow management 机制监控 warp 内线程的 control flow，并将执行同一条 instruction 的线程进行协同调度（co-schedule）。  
该机制可通过 Instruction Set Architecture (ISA) 中的 instructions 由软件进行控制或辅助，以实现最优且高效的线程调度。

为现代 GPU 设计高效的 control flow management 机制既关键又具有挑战性，根本原因在于现代 GPU 支持丰富的 control-flow instructions。  
例如，NVIDIA Turing 架构 [29] 在其原生 ISA（称为 SASS [28]）中包含 20 条 control-flow instructions。  
这些 control-flow instructions 可能使线程发生 divergence，走向不同的执行路径；由于线程执行的 instructions 不同，它们无法被共同调度。  
线程 divergence 会降低 SIMD 利用率与性能；因此，control flow management 机制会利用各个线程在运行时的 control flow 信息以及软件提供的信息，**将线程“重新聚合”（reunite）以获得更高的效率**。

研究者已基于公开信息并借助开源工具提出了众多软硬件 control flow 管理机制 [8], [9], [11][14], [23], [25], [27], [35], [39]。  
他们广泛使用 LLVM [20] 进行 compiler 实现，并使用 GPGPU-Sim 3.x [3] 作为性能模型进行评估。  
GPGPU-Sim 3.x 与 LLVM 都采用文档完备的 Parallel Thread Execution (PTX) ISA [30] 作为软硬件之间的接口。  
多年来，研究者一直以此策略来应对 NVIDIA 等领先 GPU 厂商在 compiler、ISA 与硬件方面的不透明性。

然而，这一路径存在一个主要问题：它依赖于 PTX 作为 ISA（因为其文档完善），从而隐含地要求对运行该 ISA 的微架构进行建模与优化。  
但 PTX 并非 GPU 实际运行的 ISA，因此这些模型可能与真实硬件存在显著偏差。  
在 control flow management 上这种偏差尤为明显：NVIDIA GPU 实际运行的是原生 ISA——SASS，而 PTX 到 SASS 的翻译并不像早期 NVIDIA GPU 世代 [23] 那样近乎一一对应。  
仅就 control-flow instructions 而言，Turing 中的 SASS 有 20 条，而 PTX 只有 5 条。  
此外，PTX 代码在生成最终 SASS 之前会经历静态优化（static optimizations），这可能改变最终的 control flow。

当基于 PTX 设计 control flow management 机制时，往往会忽略一些约束——这些约束由 SASS 中需要支持而 PTX 中不存在的 instructions 所施加。  
进一步地，为研究深度学习（deep learning）[2], [24], [34] 或图分析（graph analytics）[26] 等现代工作负载的最先进实现，我们别无选择只能依赖 SASS ISA，因为这些应用使用了高度调优的库（如 cuDNN [5]、cuBLAS [32] 等）。  
这些库由 NVIDIA 提供，优化程度极高，但其源代码不可得；因此只能在真实硬件上运行时通过 profiling 或收集 SASS traces 来进行研究。

研究者也已认识到仅模拟 PTX 的局限，因而开发了诸如 Accel-Sim [19] 这样的 trace-driven 模拟器。  
Accel-Sim 可模拟在现代 GPU 架构（如 Volta [31] 与 Turing）上运行的 SASS instruction traces。  
**与多数 trace-driven 模拟器类似，它只对硬件组件的 performance 进行建模，而不覆盖其 functionality**；根本原因在于 instructions 与硬件组件的功能并未公开，发掘底层硬件机制需要额外投入且并非总是可行。  
因此，诸如 Accel-Sim 的模拟器只能依赖真实硬件为 control flow management 机制所生成的 control flow traces。

然而，当研究者开发会改变硬件组件 functionality 的新机制时，单纯模拟 performance 或依赖硬件 traces 是不够的。  
这在新的 control flow management 机制上尤为突出：新的方案会改变 control flow，其连带效应会影响 issue schedulers、dependence checking 等其他组件。  
因此，必须对 control-flow management 机制的 functionality 进行建模，才能研究其影响并评估替代设计。

要对 control-flow management 机制进行功能级建模，首先需要了解 ISA 中 control-flow instructions 的语义（semantics），其次需要微架构（microarchitecture）细节。  
但这些细节并未公开。  
即便是对现代 GPU（如 Volta [18]、Turing [17]）的架构与 ISA 进行剖析的相关工作，也尚未涉及 control flow management 机制。

在本工作中，我们基于对多种应用的 binary 与 traces 的研究所收集的实验数据，给出了 Turing ISA 中 control-flow instructions 的语义定义。  
这一方法使我们得以设计一种新的 control flow management 机制 **Hanoi**，其支持上述 control-flow instructions。  
我们证明，Hanoi 生成的 control flow 对所有基准测试（benchmarks）均产生正确输出，并与真实硬件高度一致：  
将真实硬件与 Hanoi 的 control flow traces 进行比较，平均差异仅为 1.03%，对应的性能变化仅为 0.19%。

据我们所知，**Hanoi** 是首个针对 Turing ISA、并覆盖多样且知名 benchmarks 中出现的所有 control-flow instructions 的 control flow management 机制。  
Hanoi 在硬件成本上十分轻量（lightweight），且与实际硬件机制高度相似。  
其他已提出的方案 [11][14], [25], [35] 要么面向 PTX ISA 设计，要么其成本/收益不足以支撑在真实产品中采用 [8]。  
此外，这是首次尝试为 Turing ISA 中在常见 benchmarks 里出现的 control-flow instructions 提出合理（plausible）的语义描述；文献 [8] 中提及的 Turing 少数 control-flow instructions 的语义并不完整，20 条中仅定义了 3 条，无法覆盖我们在 benchmarks 中遇到的全部场景。

**总结与贡献如下：**  
• 我们定义了 Turing ISA 中 control-flow instructions 的语义。  
• 我们设计了新型的 Turing control flow management 机制 **Hanoi**。  
• 我们将 **Hanoi** 与 Turing 实际硬件中的 control flow management 机制进行了对比，证明二者极为相似：control flow trace 平均仅相差 1.03%，从而导致 IPC（Instructions Per Cycle）仅有 0.19% 的轻微变化。

## 👩🏻‍💻 2 PRE-VOLTA CONTROL FLOW MANAGEMENT

领先的 GPU 厂商（如 NVIDIA）通过其已文档化的 execution model 向程序员披露了部分 control flow management 机制。  
该 execution model 为使用 CUDA 或 PTX 等编程语言编写正确且无 deadlock 的程序提供了必要的假设。  
同时，它也为研究者推断一个 plausible 的 control flow management 实现提供了有价值的线索，尽管具体的实现细节从未被公开。

为便于说明，本文假设一个 warp 仅包含四个线程。  
Figure 1 展示了基于 execution model 推导出的、适用于 pre-Volta GPUs 的一种可能的 divergent threads control flow management。  
同一 warp 中的所有线程以 lockstep 方式开始执行 Figure 1a 所示的源代码。  
当线程执行到第 2 行时发生 branch divergence：warp 的前一半线程走 taken 路径，其余线程走 not-taken 路径。  
在这种情形下，NVIDIA 提供的 execution model 规范 [10], [23], [33] 描述的执行如 Figure 1b：对 taken 与 not-taken 两条路径进行串行执行，并在第 9 行进行后续的 reconvergence。  
第 9 行是称为 Immediate Post Dominator (IPDom) [6] 的点，也即程序中两条分歧路径保证再次汇合的**最近位置**。  
尽管 execution model 未说明分歧路径的优先级，在该示例中假定 taken 路径的优先级高于 not-taken。

一种可实现上述行为的机制是使用称为 **SIMT-Stack** 的结构 [14], [39]。  
该机制利用 SIMT-Stack 跟踪“哪些线程位于哪些路径”，并在 IPDom 处强制执行 reconvergence。  
每个栈条目（stack entry）存储“下一条 instruction 的 PC”以及“执行该 instruction 的线程的 active mask”。  
active mask 中对应位为 1 的线程执行下一条 instruction，而其余线程保持 idle。  
Figure 1c 展示了执行该示例代码时 stack 的更新方式。

初始时，stack 只有一个条目，表示所有线程共同执行指令 A（①）。  
当发生 branch divergence 时，栈顶条目被弹出，并向栈中压入三个新条目（②）：  
首先压入包含 IPDom 的 PC 与完整 active mask 的条目，因为所有线程在 reconvergence 后都必须从 IPDom 处继续；  
随后再分别压入两个条目：一个对应 taken 路径的线程，另一个对应 not-taken 路径的线程。  
当 taken 路径结束，其条目被弹出，执行转向 not-taken 路径（③）。  
当 not-taken 路径也结束后，其条目被弹出，此时栈顶指针指向含有 IPDom PC 与 full active mask 的条目（④）。  
这便实现了在 IPDom 处的 reconvergence。

需要指出的是，SIMT-Stack 虽由 execution model 推导而来，但该 model 并未揭示 control flow management 机制的更深层细节，例如该 stack 是否完全由硬件实现，或是否由软件管理。  
此外，model 也未涉及 SIMT-Stack 的诸多细微之处，而这些细节可能显著影响性能，例如路径选择策略的多样性——优先 taken、优先 not-taken，抑或在硬件或软件中采用 heuristic 进行选择。  
研究者通过 micro-benchmarking 揭示了此类细节（如路径选择策略）[38]。  
他们也广泛采用一种“完全由硬件实现、并由软件提供少量辅助以标识每个分支的 IPDom 位置”的 SIMT stack 方案 [1], [3], [14]，并对该机制如何集成进 GPU 的核心微架构作出了一些假设。

Figure 2 展示了研究社区对该机制集成进 GPU SIMT cores 的一个常见假设：为每个 warp 配置专用单元，称为 **Control Flow Management Units (CFUs)** [1]。  
每个 CFU 包含一个 stack 与必要的 control logic。  
CFU 为下一条 instruction 提供 PC 与 active mask。  
fetch unit 使用该 PC（①）从 memory 请求 instructions；当 instructions 被 fetch 并 decode 后，它们被放入 Instruction Buffer (I-Buffer) 的若干 slots 中，每个 warp 在 I-Buffer 中拥有专属的 slots。

在这种 SIMT core 架构中，一个 warp 在被选中时按程序顺序（program order）发射（issue）instructions。  
issue logic 会选择其“下一条 instruction 不存在 data/structural hazard”的特定 warp。  
为处理 data hazards，体系为每个 warp 配置私有的 scoreboard 进行 dependence checking。  
当一条 instruction 被 issue 时，active mask 表示哪些线程正在执行该 instruction；CFU 提供该 mask（②），后续 pipeline stages 的各个组件据此进行操作（例如屏蔽对寄存器或 memory 的更新）。  
一条 control-flow instruction 可能改变控制流或 CFU 的内部状态，因此在执行这些 instructions 之后需要更新 CFU（③）。

利用 SIMT-Stack 进行 control flow management 大幅简化了 SIMT core 架构。  
其简化的核心在于：基于 stack 的实现不支持不同路径的交错执行（interleaved execution），从而无需像部分文献所提议的那样为每条路径单独配置 I-Buffer slots 或 scoreboards [12]。  
然而，SIMT-Stack 也对线程调度施加了约束，在某些场景下可能导致 deadlock，这一点必须由程序员加以考虑。

## 📌 3 SIMT-INDUCED DEADLOCKS IN PRE-VOLTA

在 **pre-Volta** 的 execution model 中，在线程发生 divergence 时，线程调度会受到三项由 control flow management 机制实现所带来的约束。  
程序员必须谨慎考虑这些约束，以确保程序正确性并避免 deadlock。约束如下：

1. divergent paths 串行（serially）执行，逐条路径依次进行。
2. 每条路径内的线程以 lockstep 方式执行。
3. 在 **immediate postdominator (IPDom)** 点强制执行 reconvergence。

在相关文献中，由 SIMT 实现约束引发的 deadlocks 被称为 **SIMT-induced deadlocks** [4], [11], [15], [22], [38]。  
在 **pre-Volta GPUs** 中，control flow management 机制是大多数 SIMT-induced deadlocks 的主要原因。

例如，Figure 3 所示的一个 CUDA 中的 **spinlock** 实现，会因为 control flow management 机制而产生 deadlock [11]。  
在该示例中，当某个线程获得 lock 后，它会退出循环（loop），执行 **critical section**，并释放 lock 以便其他线程进入。  
竞争获取 lock 的线程分化为两条路径：  
— 路径 a：退出循环并进入 critical section；  
— 路径 b：返回循环起点继续自旋。

获得 lock 的线程走路径 a，而同一组中其余线程走路径 b，并在循环中等待直到 lock 被释放。  
由于约束 1 与约束 3，该场景在 pre-Volta GPUs 上会导致 deadlock：

• 若调度优先级给到路径 b（满足约束 1 的“先执行其一”），则发生 deadlock：路径 b 无限期等待路径 a 在第 4 行释放 lock，而这一步永远不会发生。  
• 反之，若优先路径 a，deadlock 仍然存在，原因是约束 3 要求在 IPDom 点（第 3 行，紧随 loop exit 之后）完成 reconvergence，随后才能执行 lock release（第 4 行）。然而，等待 lock 的线程被困在循环中，无法到达 IPDom；而已获得 lock 的线程又被 block 在 reconvergence 点等待其他线程，因而永远无法执行到释放 lock 的位置。

如果没有上述 control flow management 约束，此 deadlock 本可避免。  
为解决这一问题，NVIDIA 在 **post-Volta** 的 execution model 规范中移除了这些约束。  
这表明 post-Volta GPUs 采用了实质性不同的 control flow management 机制——但其具体细节仍未公开。

## 🙋‍♀️ 4 POST-VOLTA CONTROL FLOW MANAGEMENT

在 **pre-Volta** 的 execution model 中，程序员直接暴露于底层的 control flow management 约束之下，不得不人为介入以避免 **SIMT-induced deadlocks**，过程繁琐。  
随着 **Volta** 的引入，这种人工介入的需求被显著降低，从而简化了编程。

**post-Volta** 的 execution model 移除了 **pre-Volta** 的三项约束，仅引入一项新的约束。  
因此，程序员现在需要确保这唯一的约束不会在线程的执行之间造成循环依赖（cyclic dependencies）。

在 **post-Volta** execution model 中：  
— divergent paths 不必串行（serialized）执行；  
— 一个 warp 内的线程不必以 **lockstep** 方式执行；  
— 在 **IPDom** 点的 reconvergence 不再被严格强制。

取而代之的是，一种名为 **independent thread scheduling** 的新调度机制 [10], [31], [33]：在任意一个 cycle，凡是 **PC** 相同的 warp 内任意线程都可以被一起调度。  
这种调度方式允许不同路径的 **interleaved execution**，并允许 warp 内线程在任意 instruction 处发生 divergence 或 convergence，而不仅局限于 control-flow instructions 或 IPDom 点。  
因此，相较于 IPDom，reconvergence 可能更早或更晚出现，甚至在某些分支上被完全忽略。  
该模型唯一的约束是：在为此目的新增的指令（如 **syncwarp**）处强制执行 **intra-warp synchronization** [10], [30], [33]。于是，程序员必须在需要时对 warp 内线程进行同步以维持正确性。

**示例与直观理解。**  
Figure 4 展示了与 Figure 1 相似的示例源代码；主要差异是在第 10 行加入了 `syncwarp()` 以同步整个 warp。  
Figure 4b 给出了该代码在 post-Volta GPU 上的一种可能执行：  
分歧路径上的线程以 **interleaved** 方式执行，并忽略了在 IPDom（第 9 行）处的 reconvergence；  
但在执行 `syncwarp` 之后，warp 中的所有线程被同步。  
该示例说明了程序员如何显式地实施 **intra-warp synchronization**。

**抽象简化与信息缺失的权衡。**  
这一 execution model 虽然简化了编程，却几乎遮蔽了底层 control flow management 机制的细节。  
例如，Turing 中的 control flow management 显然是 **hardware/software design**，因为原生 ISA 中存在大量可用于辅助的 control-flow instructions；然而，这些指令的语义及机制如何利用它们并未公开。  
同时，具体的调度策略也未披露。尽管理论上 post-Volta GPU 可对线程进行独立调度，但一种更具成本效益的策略也许是：在每条路径内把所有线程合并调度，并“偶尔在路径间切换”以获取效率——这只是众多可行策略中的一个。

由于 **post-Volta** execution model 向程序员隐藏了大量细节，研究者在推断底层 control flow management 时面临困难，原因在于存在过多的合理设计选择。  
本质上，任何不会导致 **pre-Volta** 中 **SIMT-induced deadlocks** 且严格遵守所有 **intra-warp synchronizations** 的硬件/软件 control flow management 机制，都可被视为 **plausible** 的设计。

在本工作中，我们通过研究 **post-Volta** GPU 在其原生 ISA（**SASS**）上的 binary 与 traces，力图揭示其底层的 control flow management 机制。

## 🔬 5 TURING CONTROL FLOW INSTRUCTIONS

我们对一款 Turing GPU 的 native ISA 进行了分析，以理解其底层的 control flow management 机制。  
Turing ISA 支持 predication，并包含 20 条 control-flow instructions（见 Table I）。  
NVIDIA 的文档虽有简要提及这些 instructions，但对其 functionality 与 semantics 语焉不详。  
我们通过研究多种 benchmarks 的 binary 与 traces 来破译这些 instructions；其中仅有在图表中以绿色高亮的 instructions 出现在我们的基准集中。  
本节阐述我们的发现，并解释我们为每条 instruction 定义语义的依据与理由。

### A. Predicated Control Flow Instructions

我们的研究表明，Turing 中的 control-flow instructions 最多可由 2 个 predicate registers 进行守护（guard），并且在使用前可以对它们取反（negate）。  
这些 predicate registers 对于一个包含 32 条线程的 warp 来说是 32-bit 寄存器（每线程 1 bit）。  
第一个 predicate register 以 @ 前缀出现在 instruction 之前；第二个 predicate register 始终作为 第一个 operand。  
若要对某个 predicate 取反，需要在其寄存器名之前加 !。  
当一条 instruction 同时带有两个 predicates 时，系统会先对二者进行 boolean AND，再执行 predication。

示例：`@P0 INST !P1, R0` 表示仅当 P0 为真 且 P1 为假 时，该 instruction 才会执行。

### B. EXIT Instruction

EXIT 指令用于终止线程的执行。  
该 instruction 最多可带 一个 predicate，且 无 operands。  
被 predicate 屏蔽（masked）的线程会从 下一条 instruction 继续执行，而其他线程则被 终止。

### C. BRA Instruction

BRA 指令用于 跳转（jump） 到目标地址，可 条件 或 无条件 执行。  
除 predicates 外，其唯一的 operand 为 target address。  
若未带任何 predicates，则为 无条件跳转；否则，BRA 最多可由两个 predicates 守护，从而形成 conditional branch。

示例：`@!P0 BRA P1, target` 表示仅当 P0 为假 且 P1 为真 的线程跳转到 target。

### D. CALL 与 RET Instructions

NVIDIA 在进行函数调用时，使用 registers 而非 stack-based 机制来存储 return address。  
在该方案中，compiler 会在执行 CALL 之前，将 return address 写入 registers（通常通过 MOV 指令完成）。  
在被调用函数内部，RET 使用相同的 registers 返回到 caller。  
CALL/RET 指令带有 modifiers，可对其行为做轻微调整。

### E. BMOV、BSSY、BSYNC 与 BREAK Instructions

我们的研究显示，compiler 会在程序流中插入 BMOV、BSSY、BSYNC、BREAK 等 instructions，以辅助 control flow management 机制在分支之后实现 thread reconvergence。  
这些 instructions 对 CUDA 或 PTX 程序员 不可见，仅由 compiler 自动加入。

BSYNC 用于在 reconvergence points 处将 warp 的线程在分支后重新汇合。  
相较于强制在 IPDom 点 reconverge，在 BSYNC 处 reconverge 可以发生在 早于 或 晚于 IPDom 的位置：  
\- 更早 的 reconvergence 可提升部分 unstructured control flow 程序的性能（详见 Section VI-B）；  
\- 更晚 的 reconvergence 则是为避免 pre-Volta GPUs 中出现的 SIMT-induced deadlocks 所必需（详见 Section VI-C）。  
NVIDIA 的 compiler 会分析源代码，并在合适位置插入 BSYNC，该位置既可能是 IPDom，也可能是其他位置。  
BSYNC 只有一个 operand：一个特殊用途的寄存器 Bx。  
Bx 中存储的数值对于理解 BSYNC 至关重要。

遗憾的是，NVIDIA 的二进制插桩工具 **NVBit [37] 无法像读取通用寄存器那样捕获 Bx 的值**。  
然而，BMOV 可以在 Bx 与 Rx 之间传递数值；因此，通过读取 Rx 的值，我们间接获得了 Bx 的内容。  
我们发现 Bx 寄存器中存放的是一个 mask。进一步研究表明，该 mask 指示 BSYNC 需要 reconverge 的 warp 线程集合。  
我们将该 mask 称为 reconvergence mask。  
例如，若 B0 中的 reconvergence mask 为 1100，则 `BSYNC B0` 仅会 reconverge 线程 2 与 3。

读取 Bx 也帮助我们把握 BSSY 的语义：  
BSSY 用于 初始化 某个 Bx 并通过一个 instruction PC 指定 reconvergence point（**该 PC 总是 指向某条 BSYNC**）。  
Bx 在 BSSY 执行时 被初始化为当时 warp 的 active mask。  
This mask represents the reconvergence mask since a branch that causes divergence is always preceded by a BSSY instruction. 因此 reconvergence mask 所指示的所有线程，在分歧发生前都已执行过 BSSY。  
例如，若某 warp 的 线程 2 与 3 在分支前执行了 `BSSY B0, 1000`，则必须将 B0 初始化为 1100；  
当地址 1000 处的 `BSYNC B0` 执行时，它会读取 B0 并重聚 线程 2 与 3。

我们还发现，当 reconvergence 点 不在 IPDom 时，需要 BREAK 来避免 deadlocks。  
按定义，所有发生分歧的线程在程序结束前都必须经过 IPDom；但并无法保证它们一定会经过那些 非 IPDom 的 reconvergence 点。  
因此，某些分歧线程可能 永远到达不了 指定的 reconvergence 点；若在该点等待这些线程，就会产生 deadlock，除非使用 BREAK 将它们从 reconvergence mask 中 移除。  
Section VI-B 详细说明了当 reconvergence 点 早于 IPDom 时，如何用 BREAK 来避免 deadlocks。  
BREAK 带 一个或两个 predicates 以及一个 Bx 寄存器；predicates 用来判定需要从 Bx 的 reconvergence mask 中 移除 的线程集合。  
被移除的线程将 不再 与 mask 中的其他 active 线程 reunite。  
例如，`@P0 BREAK !P1, B0` 会将 P0 为真 且 P1 为假 的线程从 B0 的 reconvergence mask 中移除。  
此后，`BSYNC B0` 将 不会 等待被移除的线程，除非有其他 instructions 改变了 B0。

### F. WARPSYNC Instruction

WARPSYNC 用于在 warp 内 synchronize 线程，其作用与 BSYNC 类似。  
该 instruction 只有一个 operand：一个 mask，指示需要同步的 warp 线程集合；其含义与 BSYNC 的 reconvergence mask 相同，我们沿用同一术语。  
WARPSYNC 的 reconvergence mask 可以来自 Rx 寄存器或 immediate。  
例如，`WARPSYNC 1100` 同步 线程 2 与 3，其效果与当 R0 中包含 1100 时执行 `WARPSYNC R0` 完全一致。

### G. YIELD Instruction

我们基于若干关键观测来定义 YIELD 的语义：  
（1）warp 内的线程仅在 branches 处发生 divergence，并只在 BSYNC/WARPSYNC 处 reconverge；  
（2）对其他 instructions，warp 内线程以 lockstep 方式执行；  
（3）不同路径的 interleaved execution 仅在执行 YIELD 之后才会出现。

据此，我们推断：对于一条分支，其两条路径默认按 顺序（sequentially） 依次被调度，除非 执行了 YIELD；  
YIELD 触发 control flow 切换到 新路径。  
YIELD 不带 operands 来指示下一条路径应为何者；我们假设它切换到 sibling path，因为这既与我们实验中观测到的 hardware control flow trace 一致，又避免了高成本的微架构设计。  
若不存在 sibling path，则执行继续从 YIELD 的下一条 instruction 进行。

该定义放宽了 pre-Volta 在调度分歧线程时的 第一条约束（见 Section III）。  
pre-Volta GPUs 从不交错执行不同路径，而 post-Volta（如 Turing）则通过 YIELD 实现路径间的交错，从而能够化解部分由第一条约束导致的 SIMT-induced deadlocks。

我们设计了实验验证 NVIDIA 使用 YIELD 来解决此类 deadlocks：  
从 Figure 3 的源程序对应的 binary 中 移除 YIELD；该程序是 pre-Volta GPUs 上 SIMT-induced deadlocks 的经典示例（见 Section III）。  
在 Turing GPUs 上，原始程序可以 正常结束，但去掉 YIELD 后程序 永不结束。  
该实验验证了 Turing 依赖 YIELD 来避免某些 pre-Volta 上存在的 SIMT-induced deadlocks。

由于 YIELD 仅对 compiler 可见，NVIDIA 的 compiler algorithms 通过在 恰当位置插入 YIELD 对化解这些 deadlocks 至关重要（详见 Section VI-C）。

## 💭 6 Pratical Applications OF TURING CONTROL FLOW INSTRUCTIONS

本节说明 Turing 中的 control-flow management 机制如何在多种场景下使用 control-flow instructions，并给出三个实际案例：  
1）嵌套分支后的线程重聚（reconvergence）；2）早于 IPDom 点的重聚；3）Spinlock 实现。

### A. Reconvergence after Nested Branches

嵌套分支可能引发嵌套的线程 divergence，需要使用多个 Bx registers 存放 reconvergence masks。  
理论上，最多可能需要 31 个 Bx，因为一个拥有 32 条线程的 warp 最多可分化为 32 条独立路径。  
但 NVIDIA 实际使用较少的 Bx，并将其值 spill 到 Rx registers，以节省硬件资源；为此 BMOV 在 Bx 与 Rx 间搬移数值。研究表明，运行期许多 Rx registers 是 dead 的 [16]，适合作为在到达 reconvergence 点之前的 Bx 的临时存储位置。

在嵌套 divergence 场景中，BMOV、BSSY、BSYNC 的使用顺序至关重要：  
BSSY 负责初始化某个 Bx；BSYNC 使用该 Bx 来强制执行 reconvergence。  
当 Bx 完成初始化后，必须把其值搬到 Rx；在到达 reconvergence 点之前，再把该值写回 Bx。  
若随后需要把 Bx 分配给“嵌套分支产生的新的 reconvergence mask”，这种数据往返搬移就是必要的。

Figure 5 展示了含两层嵌套 divergence 的示例程序中 BSSY、BSYNC、BMOV 的用法：  
其 control flow graph 见 Figure 5a，而 Figure 5b 展示了一个示例 warp 在执行期间如何更新寄存器。  
每个 basic block 实际上可能包含更多指令，这里仅展示 control-flow instructions 及其在 basic blocks 内的正确顺序。  
通过多轮实验，我们观察到：执行 BMOV 的所有 active 线程 对 Rx registers 读/写的 数值相同；因此图中仅以一个 active 线程的 Rx 值作为代表。寄存器中的 dead value 以虚线表示。  
在该示例中，reconvergence 由 BSYNC 实施，且在分支后 taken path 被优先执行。我们还观察到：NVIDIA 倾向优先执行 大多数线程所走的路径，但这只是优化；NVIDIA 的 compiler 会生成与“运行期到底优先 taken 还是 not-taken”无关的 正确 binary。

在示例中，B0 被两个 BSYNC 使用：  
— E 中的 BSYNC 重聚 线程 2 与 3；  
— F 中的 BSYNC 重聚 所有线程。  
每条 BSYNC 都需要一条 BSSY 来以 reconvergence mask 初始化 B0。  
用于 F 的 BSYNC 需要 A 中的 BSSY 将 B0 初始化为 1111（1）；  
用于 E 的 BSYNC 需要 B 中的 BSSY 将 B0 初始化为 1100（3）。  
由于 B 在 A 之后执行，它会 覆盖 B0 的值。这个被覆盖的值（reconvergence mask）稍后在 F 处还要用于重聚所有线程。  
为避免在执行 B 时丢失该值，A 中的 BMOV 将 B0 复制到 R0（2）；该值一直保存在 R0，直到 F 中的 BSYNC 需要它。  
在此之前，F 中的 BMOV 会先从 R0 取回该值并 写回 B0（4，5）。  
该示例说明：在嵌套 divergence 场景里，BMOV 用 R0 作为 B0 的备份。借此，NVIDIA 无需增加 Bx 的数量来覆盖最坏的嵌套情况；当 Bx 不足 时，compiler 会在程序中插入 BMOV，把 Bx 的值 spill 到 Rx。

### B. Reconvergence Earlier than IPDom

使用 BSYNC 的优势在于：reconvergence 可发生在 早于 或 晚于 IPDom 的位置。  
对 unstructured control flow 的程序而言，更早 的 reconvergence 可能带来性能提升。  
但若在 早于 IPDom 的位置进行 reconvergence，不当 使用 control-flow instructions 可能导致 deadlock。原因是 “早期重聚点” 并非 IPDom，并**不保证 所有分歧线程 在程序结束前都必然经过该点**；如果在此处等待那些永远不会到达的线程，就会形成 deadlock。

Figure 6 给出一个可进行 early reconvergence 的示例：  
Figure 6a 是 control flow graph，Figure 6b 展示了单个 warp 在执行过程中寄存器的更新。  
为简明起见，我们只展示每个 basic block 内的 control-flow instructions；NVIDIA 的 compiler 会以图中所示 相同顺序 将这些指令插入基本块。  
假定：在分歧后 taken path 先于 not-taken path 执行，且 warp 内线程在 BSYNC 后实现 reconvergence。

此示例中，D 是 A 的 IPDom，因为它是 warp 中 所有线程 在结束前必然经过的 最早 位置。  
然而，线程 0 在 A 处发生分歧后 从未经过 B，而是执行 C、D、E 后结束。  
因此，B 不能是 A 的 IPDom；但 B 却是 线程 1、2、3 的 early reconvergence 点，因为它们都能在执行 D 之前先行在 B 汇合。

该示例包含两条 BSYNC，用于重聚在 A 之后分歧的线程：  
— 一条插在 B，  
— 另一条插在 D。  
它们分别从 B0 与 B1 读取 reconvergence masks，这两个寄存器均在分歧前由 A 的两条 BSSY 初始化为 1111（1）（因为 warp 的所有线程一起执行了这两条 BSSY）。

若 B0 的值 不改变，该程序会因为 deadlock 而 永不结束：  
B0=1111 且被 B 中的 BSYNC 使用，意味着 warp 的所有线程 都必须在 B 之后 reconverge；  
但 线程 0 永远不执行 B，于是其余线程在 B 处 无限等待 线程 0。

解决方法是在 C 中插入 BREAK：  
BREAK 将 C 中 P1 为假 的线程 从 B0 的 reconvergence mask 中移除。  
这些线程在 C 的分支后会分化到 D，且 不会再执行 B。  
在本例中，我们假设仅 线程 0 在执行 C 中的 BREAK 时 P1 为假；  
把线程 0 从 B0 中移除后，B0 变为 1110，即 B 处 BSYNC 的 reconvergence mask（2）。  
因此，只有线程 1、2、3 在 B 之后重聚（3），并 不再等待线程 0；  
线程 0 则在 D 之后再与它们会合（4），且此后程序不再发生新的分歧。  
该示例表明：在 C 中加入 BREAK 可在 B 处实施 early reconvergence，从而 无死锁 地正确结束程序；这也强调了 compiler 只需在 恰当位置 插入 control-flow instructions，便能协助机制实现 早于 IPDom 的重聚。

### C. Spinlock Implementation

Figure 3 所示的 CUDA spinlock 在 pre-Volta GPUs 上会因两条约束而产生著名的 SIMT-induced deadlock：  
1）在 IPDom 处 强制 reconvergence；2）分歧路径串行、逐条执行（详见 Section III）。  
Turing 借助 compiler 的帮助，通过 移除这些约束 来避免该 deadlock：  
当在 IPDom 处放置 BSYNC 会引发 deadlock 时，compiler 不会 把 BSYNC 放在 IPDom；  
当持续执行同一路径将导致 deadlock 时，compiler 会插入 YIELD，切换到 sibling path。

Figure 7 说明在 Figure 3 的 spinlock 实现中，BSYNC 与 YIELD 在避免 deadlock 上的关键作用：  
Figure 7a 给出加入 Turing 控制流指令后的 control-flow graph；  
Figure 7b 则展示了一个示例 warp 的可能执行过程（假设分歧后 taken path 先于 not-taken path 执行）。  
在 taken path 结束前，YIELD 介入并切换到 not-taken path。  
YIELD 是 必要 的：我们观察到，若从程序 binary 中移除 YIELD，其执行将 永不结束。

该程序在 D 中（紧随 loop 之后）包含 critical section。  
在 loop 内，线程竞争 acquire a lock 以获准进入 critical section；仅有一条线程能够成功，并在执行 C 时令 P0 为真。  
该线程从同一 warp 的其他线程中分化，退出 loop 执行 critical section；完成后 release the lock，允许下一个线程进入。  
在本例中，线程 3 获得 lock 并分化，但 不会立即 执行 critical section，因为 taken path 优先且必须先执行。  
于是 线程 0、1、2 跳回 loop 起始并执行 YIELD（1）。  
若 不存在 YIELD，这些线程将 永远无法获得 lock（因为线程 3 仍持有 lock），从而 被困在 loop 中。  
YIELD 解决了该问题：它将执行切换到 D（2）（sibling path），由 线程 3 在 D 中执行 critical section 并释放 lock。  
随后，线程 3 执行 E 中的 BSYNC，用于 reconverge 所有线程；此后，线程 3 阻塞等待 其余线程也到达该 BSYNC。  
在这一场景中，其余线程从 C（3）（YIELD 之后的下一条指令）继续执行；此时，由于线程 3 已释放 lock，它们中的某一个能够成功 acquire the lock 并进入 critical section。  
在示例里，线程 2 获得 lock 并执行 critical section（4）。

需要注意的是：若 E 中的 BSYNC 被放在 release the lock 之前，同样会引发 deadlock。  
原因是持有 lock 的线程会 无限等待 在 BSYNC 处与其他线程重聚，而其他线程又 困在循环 中（无人能获得 lock、退出 loop）。  
在此程序中，IPDom 就在分支之后、release the lock 之前。  
因此，在 检测到 deadlock 场景 后，NVIDIA 的 compiler 决定将 BSYNC 放在 晚于 IPDom 的位置——即 E 中、在 lock 已经被释放之后。

## ♨️ 7 HANOI MICROARCHITECTURE

Figure 8 展示了我们提出的 **Hanoi** 设计，用于 Turing 的 control flow management 机制。该设计包含两类 stacks：Warp Split（WS）（1）与 Reconvergence（REC）（2）；此外还包含若干 Bx registers（3）以及两种 masks：waiting（4）与 finished（5）。

WS stack 按“每条待执行路径（path）”设置一个 entry，每个 entry 含一个 PC 与一个 active mask。  
WS stack 中的各条路径按照 stack 顺序执行，栈顶 entry 对应“当前正在执行”的路径。  
active mask 指示 warp 中哪些线程跟随该路径，PC 指定这些线程接下来要执行的 instruction。  
在图例中，warp 的线程 2 将在某条路径上执行 instruction 20，而线程 1 将在另一条路径上执行 instruction 50。

REC stack 按“每个 reconvergence 点”设置一个 entry，每个 entry 含一个 PC 与一个 Bx register ID。  
reconvergence 按照 REC stack 中的顺序发生，栈顶 entry 对应当前正在执行的路径。  
其中 PC 指向线程在 reconvergence 之后必须执行的 instruction；Bx register ID 指向存放 reconvergence mask 的 Bx。  
reconvergence mask 指示 warp 中哪些线程必须在该 reconvergence 点重聚。仅当 Bx 的 valid（V）位为 1 时，Bx 中的 reconvergence mask 才是有效的。

图中包含两个 reconvergence 点：一个在 PC 100，另一个在 PC 500。  
REC stack 中“PC 100”对应的 entry 引用 B1，B1 的 mask 表示线程 2 与 3 必须在此处重聚。  
“PC 500”对应的 entry 引用 B0，B0 的 mask 表示线程 2 与 3 必须在此处与线程 1 会合。

waiting mask 指示哪些线程正在“当前 reconvergence 点”（即 REC 栈顶）等待；图中线程 3 已到达其在 PC 100 的 reconvergence 点。  
finished mask 追踪已执行 EXIT 并完成的线程；图中线程 0 已完成。

### A. Managing Bx Registers

BSSY instruction 用于初始化 Bx。它通过其 operand 中的 Bx register ID 指定目标 Bx。  
执行 BSSY 的 warp 内 active 线程集合会被写入该 Bx；这些线程**恰好**与 WS 栈顶 entry 的 active mask 所指示的线程一致。  
因此，当执行 BSSY 时，会将该 active mask **复制**到指定的 Bx，并将其 valid 位设为 1。

BREAK instruction 用于更新 Bx。它通过 ID 定位某个 Bx，并将特定线程从其中存放的 reconvergence mask 中移除。  
之所以可以这样更新，是因为 REC stack 中的 reconvergence 点从 Bx 读取其 reconvergence mask；换言之，这些 entry **间接**引用了装载其 mask 的 Bx。  
如果把 reconvergence mask 直接存进 REC 的 entries，那么对于“不在栈顶”的 entry，就无法从其 mask 中移除线程。

BMOV instruction 在 Bx 与 Rx registers 之间搬运数值。  
当 BMOV 将某个 Bx 的值搬到 Rx 时，会同时使该 Bx **失效**（invalidate）；当数值被写回该 Bx 时，再将其 valid 位重新设为 1。  
BMOV 的用途之一是让不同的 reconvergence 点**共享** Bx：REC stack 中的多个 entries 可以引用同一个 Bx。  
但注意：只有当某个 entry 位于 REC 栈顶时，它才会读取对应的 Bx。  
因此，只要能在“需要该值的 reconvergence 点成为 REC 栈顶并被程序执行到”之前把数值写回，就可以先用 BMOV 把 Bx 的值搬到 Rx 暂存。

EXIT instruction 用于完成线程的执行。  
一旦执行 EXIT，Hanoi 会从**所有** Bx 中移除已完成的线程，并将这些线程加入 finished mask。  
若某个 reconvergence mask 从 Rx 读取后要写回 Bx，则必须先从该 mask 中移除 finished mask 中的线程——这是保证 control flow 正确性的必要步骤，因为当 mask 暂存在 Rx 期间，可能已有部分线程结束。

当某个 reconvergence 发生后，对应的 reconvergence mask（位于某个 Bx 中）不再需要，因此该 Bx 会被**失效化**（invalidate）。

### B. Managing REC Stack

当发生 reconvergence 时，Hanoi 会从 REC stack 弹出（pop）栈顶 entry，读取其所指 Bx register 中的 reconvergence mask，并向 WS stack 压入（push）一个新 entry。  
该新 entry 带有来自 REC 栈顶 entry 的 PC 与 reconvergence mask；因此，处于该 reconvergence mask 中的所有线程将从“重聚点之后的下一条 instruction”继续执行。

Hanoi 使用 waiting mask 判定何时需要执行 reconvergence，这对于在执行期间正确遍历 control flow graph 至关重要。  
在调度 WS 栈顶 entry 之前，Hanoi 会检查 REC 栈顶 entry 的 reconvergence mask 是否“有效（valid）”且其位集合是否“完全包含在 waiting mask 中”。  
若条件满足，表示该 reconvergence mask 所指的所有线程均已到达重聚点，Hanoi 便在此时进行 reconvergence。  
若 REC 栈顶 entry 所引用的 Bx 标记为 invalid（例如已被覆盖用于其他重聚点），Hanoi 绝不 执行 reconvergence。

Hanoi 使用 REC stack 来处理 BSYNC 或 WARPSYNC 指令触发的 reconvergence。两者都在 warp 内重聚线程，但存在重要差异，需要分别处理：

- 对 BSYNC：其前必有一条配对的 BSSY，用于给出 reconvergence PC 以及保存 reconvergence mask 的 Bx ID。执行 BSSY 之后，Hanoi 将“指定的 reconvergence PC 与 Bx ID”作为一个 entry 压入 REC 栈。
- 对 WARPSYNC：其前 没有 类似 BSSY 的指令，且 不带 Bx 操作数。Hanoi 需 分配一个空闲 Bx，并用 WARPSYNC 的 reconvergence mask 初始化之。我们假设总有空闲 Bx；若不足，compiler 可借助 BMOV 将其 spill 到 Rx。  
    WARPSYNC 的“下一条 instruction 的 PC”成为该 WARPSYNC 在 REC 栈 entry 中的 reconvergence PC（因为重聚后线程从此处继续）。  
    “reconvergence PC + 分配的 Bx ID”共同构成压入 REC 栈的 entry。  
    但请注意：仅当 reconvergence mask 的“第一个到达 WARPSYNC 的线程子集”执行到 WARPSYNC 时，才进行此压栈；对后续到达的其他线程子集而言，该 entry 已在 REC 栈中，Hanoi 不会重复压栈。

采用独立的 REC stack 来跟踪重聚点，使 Hanoi 能高效处理 WARPSYNC：在仅用“单栈”的备选机制中，要么难以支持 WARPSYNC，要么效率极低。原因在于单栈设计通常需要在 divergence 发生之前 就明确“未来的重聚点”，以便及时更新栈结构并在该点重聚；而执行 WARPSYNC 时，我们只知道线程“曾在 WARPSYNC 之前的某处发生过 divergence”，但并不知道确切位置，因此无法（或代价巨大地）更新单栈以保证 WARPSYNC 时正确重聚。

### C. Managing WS Stack

Hanoi 会因 BSYNC、WARPSYNC、EXIT、BRA、YIELD 等指令对 WS stack 执行 push/pop；  
对其他大多数指令，仅需更新 WS 栈顶 entry 的 PC。  
例如 CALL/RET 会改变 PC 以实现函数跳转与返回；对多数非控制流指令，简单地“PC 自增到下一条 instruction”即可。

BRA 可能引发线程 divergence，从而在 WS 栈压入**两个** entries：一个对应 taken path，另一个对应 not-taken path。  
我们的观测显示：大多数线程所走的路径会先执行；因此 Hanoi 会先压入另一条路径，再压入“多数路径”，以使栈顶先执行“多数路径”。

当某条路径执行结束时，必须 pop 其 entry。  
一条路径被视为结束的条件是遇到 BSYNC、WARPSYNC 或 EXIT：

- 对 BSYNC/WARPSYNC：执行后，Hanoi 弹出该路径 entry，并将其 active 线程加入 waiting mask；
- 对 EXIT：仅当该路径中的所有线程都执行了 EXIT，才从 WS 栈 弹出该 entry。  
    某些线程可能被 predicate off 而未执行 EXIT；对这些线程，下一 cycle 将继续执行 EXIT 之后 的指令。  
    执行 EXIT 的线程会被加入 finished mask。

当执行 YIELD 且 存在 sibling path 时，Hanoi 切换到 sibling。切换到 非 sibling 会产生错误的 control flow。  
若存在 sibling，其 entry 位于 WS 栈顶下方紧邻位置；因此，执行 YIELD 时，Hanoi 只需 交换 WS 栈顶两个 entries 即可切换到 sibling。  
然而，某条路径的下方 entry 未必属于其 sibling。两条路径成为 siblings 的充要条件是：共享同一重聚点，该重聚点总是由 REC 栈顶 entry 表示。  
据此，Hanoi 用如下判定：取 WS 栈顶两条路径的 active mask 的并集，与 REC 栈顶 entry 的 reconvergence mask 进行比较；  
若该并集是该 reconvergence mask 的 子集，两条路径就是 siblings，否则不是。

示例（见 Figure 8）：线程 1 与 2 处于 WS 栈顶的两个 entries 中；然而它们 不是 siblings，因为 REC 栈顶 entry 的 reconvergence mask 并未包含 线程 1。  
因此，执行 YIELD 不会切换到其他路径，其效果等同于 NOP。

## 🕹️ 8 实验方法

我们使用 NVIDIA 提供的工具，为 大量且知名的 benchmarks（见 Table II） 生成了 native assembly source code (SASS)、control flow graph (CFG) 与 traces。  
具体而言：使用 cuobjdump [28] 生成 SASS 代码，使用 nvdisasm [28] 生成 CFG，并使用 NVBit [37] 生成 trace。  
对于部分 benchmarks，我们采用了不同的输入数据集；在本文中，这些数据集通过在基准名称前加上数字进行区分。

通过对上述 benchmarks 的深入分析，我们识别出一系列 patterns、scenarios 与 use cases，展示了在 Turing GPUs 中，control-flow instructions 是如何被 定义、使用与由 control-flow management 机制处理 的。  
基于这些观察，我们提出了一组 合理且直观的假设（hypotheses），用于刻画 control-flow instructions 的 semantics 及 Turing 中 control flow management 的 细化策略（detailed policies）。

随后，我们开发了一个 checker program 来验证这些假设。  
为此，我们编写了 CFG、SASS 与 trace 的 parsers；在完成解析后，将其载入 checker，由其检查“来自实际硬件的 traces 是否违反我们的任一假设”。  
例如，若我们的假设是“interleaving 只会在执行 YIELD 之后发生”，而 traces 显示 interleaving 出现在其他位置，则视为 violation。  
一旦出现 violation，就会生成 log file，提供导致 violation 的线索；我们据此 修正假设 并 重复验证流程。  
我们的目标是得到一组 **在所有 benchmarks 上从未被违反** 的假设，以此解释 control-flow instructions 的 semantics，以及 Turing control flow management 采用的 详细策略。

考虑到 NVIDIA 采用了 runtime heuristics，要完全发现并将所有 heuristics 纳入我们的假设集合 极具挑战性。  
尽管如此，我们仍揭示了其中的许多要点，使得我们的设计 Hanoi 与 Turing 的设计 非常接近。  
为了量化这种接近程度，我们基于假设实现 Hanoi 并生成其 control flow trace；  
随后将该 trace 与 Turing 的 trace 进行比较，并据差异对 Hanoi 进行微调以 最小化不一致。

为评估这些差异对性能的影响，我们将 Hanoi 的 traces 输入至知名的 trace-driven simulator——Accel-Sim [19]，并与以 Turing 的 traces 作为 baseline 的结果进行对比。  
在性能仿真中，我们采用了 NVIDIA RTX 2060 的配置（详见 Table III）。

## 🎶 9 评估

我们基于对 control-flow instructions 语义以及 Turing 中 control flow management 机制的 detailed policies 的一组假设来开展工作。  
在这些假设之上，我们设计了 Hanoi，并不断迭代假设与设计，使之与 Turing 的架构高度贴近，最终得到第 8 节所描述的设计。

为评估 Hanoi 与 Turing 的接近程度，我们比较了二者的 control flow traces。  
control flow trace 展示了从程序开始到结束，每个 warp 在每个 cycle 所执行的 instructions 序列。  
若某个特定 warp 在 Hanoi 与 Turing 中的 control flow trace 完全一致，则表示二者在所有时刻都以完全相同的方式将该 warp 的线程一起执行。

需要注意的是，序列上的显著差异并不必然意味着设计上的巨大不同。  
例如，仅仅在一次 branch divergence 之后把 taken 路径优先级设为高于 not-taken，就可能导致 control flow trace 出现显著差异。  
本质上，control flow trace 是一串 instructions；我们用 Levenshtein distance [21] 来比较两条序列——它度量把一条序列变为另一条所需的 insertions/deletions/updates 的最小次数。  
将该距离除以序列长度即可得到一个“百分比差异”指标。

我们共分析了 59 次程序执行：包括 41 个不同的 benchmarks，以及 18 次对部分基准的不同输入数据的执行。  
在这 59 次中，有 46 次的差异为 0%，表明 Hanoi 与 Turing 生成了 完全相同 的 control flow trace。  
其余执行的差异如 Figure 9 所示：除 BFSD 外，所有 benchmarks 的差异百分比均 低于 2.4%，平均差异仅 1%。  
BFSD 的差异较高（49.5%），原因在于 Turing 采用了一个 runtime heuristic 而 Hanoi 并未支持：Hanoi 在 所有 BSYNC 处强制 reconvergence，而 Turing 在个别情形会为性能考虑 忽略 这些重聚。

Figure 10 展示了这些差异对性能的影响。  
我们计算了 Hanoi 与 Turing 的 相对 IPC difference：平均仅 0.2%，可忽略不计。  
除 BFSD 外，其余 benchmarks 的性能影响均 低于 1.2%。  
而 BFSD 在 Hanoi 上表现为 83.3% 的性能提升，其原因是 Hanoi 在所有 BSYNC 处强制 reconvergence，使 SIMD unit utilization 提升了 31.9%，从而带来显著加速。

### A. Hardware Overhead

Hanoi 是一种轻量级方案，可无缝集成进 GPU pipeline（见 Figure 2）。  
它 不需要 增加 I-Buffer 大小，也 不引入 显著的 scoreboard 开销；仅需扩展 scoreboard 以跟踪少量 Bx registers 之间的 dependencies。

WS stack 在最坏情况下需要 最多 32 个 entries（对应一个 warp 的 32 条线程完全分歧为 32 条路径）；此时 REC stack 需要 31 个 entries 来完成重聚。  
管理 WS/REC stacks 的成本也很低，因为它们都按 stack 方式组织与操作。  
WS 的每个 entry 仅包含一个 PC 和一个 32-bit mask，这比 SIMT-Stack 更简单（后者在这些字段之外还需要 reconvergence PC）。  
SIMT-Stack 通过比较 reconvergence PC 与 PC 来判定何时 pop；而 Hanoi 在执行 WARPSYNC、BSYNC 或 EXIT 时直接 pop。  
REC 的每个 entry 也很经济，仅由一个 PC 与一个指向 Bx register 的 index 构成。  
由于 Bx registers 可在 REC entries 间共享，所需数量很少；例如，系统有 8 个 Bx registers 时，仅需 3 bits 作为索引。  
waiting/finished masks 也都只是 32-bit masks。

总体而言，若设计包含 8 个 Bx registers，Hanoi 的总存储需求约为 432 bytes，几乎可以忽略，且比 SIMT-Stack 节省约 43% 的存储。

## 🙌 10 相关工作

GPU 厂商（如 NVIDIA）从未完全公开其 control flow management 机制。  
不过，早期的公开发表 [23] 与文档 [33] 曾披露部分信息。研究者通过 microbenchmarking [38] 发掘出更多细节，由此 SIMT-Stack 设计 [14], [39] 被广泛接受为 baseline。基于该思路，研究者开发了执行 PTX [30] 的性能模拟器（如 GPGPU-Sim [1], [3]），这对早期 GPU 与传统工作负载已足够。

然而，使用闭源库（如 cuDNN [5]、cuBLAS [32]）的现代工作负载无法在 PTX 层面完成模拟。为此，研究者提出了支持 SASS [28] 的 trace-driven simulators（如 Accel-Sim [19]）。Accel-Sim 在 control flow 上依赖 traces，并未对 control flow 机制的 functionality 建模，因为对 Volta [31] 等现代 GPU 的相关细节仍然不足。Volta 引入了 Independent Thread Scheduling [10], [31]，大幅改变了 NVIDIA 的 execution model，暗示 control flow management 机制也发生了实质变动。尽管如此，公开资料仍付阙如；即便是对 Volta [18] 与 Turing [17] 进行解密的研究，也没有披露该机制的细节。

据我们所知，本工作首次描述了现代 NVIDIA GPU binaries 中出现的 control-flow instructions 的 semantics，并提出了一个支持这些指令的实现——Hanoi。为了支持这些指令，Hanoi 的微架构与文献中的其它替代方案存在显著差异。举例来说，Hanoi 在任一时刻只执行一条路径，并且在 YIELD 之后可切换到其 sibling。这种 software-controlled interleaving 仅需简单硬件，且能提供更可预测的 control flow，从而可用于优化 GPU 的其他组件。其他方案也支持 interleaving，但均采用 fine-grained interleaving 的运行时策略，硬件复杂度更高。

以若干代表性方案为例对比如下：  
— DWS [25] 需要一个 table 与 path scheduler 来替代 Hanoi 的 WS stack。与 Hanoi 不同，它会忽略一些 reconvergence points，并尝试在运行时对 PC 相同 的 warp splits 执行重聚；该方案对 path scheduling policy 敏感，易错失重聚时机。  
— DualPath [35] 通过扩展 SIMT-Stack entries，在每个 entry 中保存 两条 active paths；任意一个 cycle 都可从栈顶两条路径中选择发射。然而它 无法支持 BREAK 与 WARPSYNC：BREAK 需要修改的 reconvergence mask 可能 不在栈顶；而 WARPSYNC 缺乏类似 BSSY 的先导信息，无法知晓何时应把重聚点压栈。  
— Multi-Path (MP) [12] 允许调度任一路径，但需要 per-path 的 I-Buffer slots 与 scoreboards，成本巨大，而 Hanoi 避免了该成本；此外，它要求将 warp splits 与 reconvergence points 存放在 random access memory 与 content addressable memory 中，而非 Hanoi 的“两栈”结构。并且 MP 未给出支持 WARPSYNC、YIELD、BREAK、EXIT 的机制。  
— AWARE [11] 虽减少了 MP 的部分成本，但仍未提出上述未支持指令的机制。  
— Subwarp interleaving [8] 的设计与 Hanoi 差异很大，支持 fine-grained interleaving；但作者指出其显著成本 不适合商用产品。Hanoi 则不同：它基于真实硬件 traces 设计，轻量 且 成本可控。

从整体轻量性看，Hanoi 明显优于其他替代方案。它是 首个 将 reconvergence masks 存入 Bx registers 的设计：  
这些 Bx 可被修改以支持 BREAK，还能 transfer 到 Rx registers 并在多个重聚点之间 共享；transfer 由 compiler 管理，从而保持硬件简单。共享 Bx 还降低了为线程重聚所需 metadata 的存储成本。  
Hanoi 也是 首个 不需要在 每条路径 与 每个重聚点 存储 reconvergence PC 的设计（与 [11]–[14], [25], [27], [35], [39] 相比）；它也不需要像 MP 或 AWARE 那样为每个重聚点存储 pending masks。相反，Hanoi 借助 compiler assistance 与更简单的硬件机制来完成重聚。  
此外，可从 不同 warps 调度线程的方案 [13], [14], [27] 与仅在 单个 warp 内 调度的 Hanoi 完全不同。至于那些在 非 IPDom 点进行重聚的方案 [9], [12], [25]，它们往往对 path scheduling policies 敏感，或依赖 profiling [13]、甚至 oracle information [14]。Hanoi 则利用 BREAK 及其他 control flow instructions，在可能时 保证更早的重聚。我们并不清楚 compiler 如何插入这些指令，但该机制与文献中的 speculative reconvergence [7] 高度相似。另有研究提出在存在 deadlock 风险时 延迟重聚 [11]，但 Hanoi 的做法不同：它依赖 YIELD 来避免 deadlock。我们实验性地观察到现代 NVIDIA GPUs 的确如此运作，尽管我们并不掌握 NVIDIA compiler 判定或插入这些指令的具体算法。

## 📜 结论

本文揭示了 SASS ISA 中 control-flow instructions 的 semantics，以及 Turing 的 control-flow management 机制所采用的 detailed policies。为此，我们利用了常用 benchmarks 的 traces 与 binaries 所包含的信息。基于这些发现与经验证的假设，我们设计了 Turing 的 control flow management 机制实现 Hanoi。  
Hanoi 成本低、生成的 control flow trace 与 Turing 高度相似：两者的 trace 平均 差异 1%，在大量多样化 benchmarks 上带来的 相对 IPC 差异 小于 0.2%。