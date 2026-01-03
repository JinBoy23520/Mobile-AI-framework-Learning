# Mobile-AI-framework-Learning
本仓库为移动端 AI 框架学习与部署资料集合。README 作为总目录，按主题对仓内文档分类介绍并提供跳转，便于查阅。

**主要分类**
- **人脸跟踪与超分**: 人脸检测/跟踪与图像超分相关实现与说明。
- **部署与集成**: 各平台部署流程、集成架构与可行性评估。
- **Android 实现与优化**: 针对 Android 的实现细节与性能优化建议。
- **Windows 与 TV 部署**: 在 Windows 与 Android TV 上的构建与部署指南。
- **工具与脚本**: 构建脚本与辅助工具。

**框架介绍**
目前仓库主要研究了两个移动端 AI 框架：
- **[Cactus](https://github.com/cactus-compute/cactus)**: 一个跨平台、高能效的内核、运行时和AI推理引擎，专为移动设备设计。提供Cactus Graph（通用数值计算框架）和Cactus Engine（支持OpenAI兼容API的AI推理引擎），支持INT8量化，适用于多种模型如Gemma、Whisper等。
- **[NNDeploy](https://github.com/nndeploy/nndeploy)**: 一款简单易用和高性能的AI部署框架，基于可视化工作流和多端推理，支持桌面端、移动端、边缘计算设备等平台。集成13种主流推理框架，提供并行优化、内存优化和开箱即用的算法节点。

---

**文档索引（按框架分类）**

**Cactus NNDeploy 相关文档**
- [cactus_nndeploy_integration_architecture.md](cactus_nndeploy_integration_architecture.md) — Cactus NNDeploy 集成架构说明。
- [cactus_nndeploy_feasibility_summary.md](cactus_nndeploy_feasibility_summary.md) — 可行性评估报告。
- [cactus_nndeploy_android_implementation_guide.md](cactus_nndeploy_android_implementation_guide.md) — Android 平台实现指南。
- [人脸跟踪&&超分.md](人脸跟踪&&超分.md) — 人脸跟踪与超分辨率策略与实验记录（基于 Cactus NNDeploy）。
- [android_face_track_super_resolution.md](android_face_track_super_resolution.md) — Android 平台上人脸跟踪与超分实现要点（使用 Cactus NNDeploy）。
- [GFPGAN_Android_部署与优化建议.md](GFPGAN_Android_部署与优化建议.md) — GFPGAN 在 Android 上的部署与优化建议（集成 Cactus NNDeploy）。
- [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md) — 通用部署流程与注意事项汇总（适用于 Cactus NNDeploy）。

**NNDeploy 相关文档**
- [build_windows_detailed.md](build_windows_detailed.md) — Windows 构建与详细说明（可能涉及 NNDeploy 工具）。
- [deploy_windows_to_android_tv.md](deploy_windows_to_android_tv.md) — 将 Windows 构建部署到 Android TV 的流程说明。
- [deploy_android_tv.md](deploy_android_tv.md) — Android TV 部署指南。
- [deploy_android_tv_on_mac.md](deploy_android_tv_on_mac.md) — 在 macOS 上为 Android TV 构建与部署的说明。
- [build_win.py](build_win.py) — Windows 构建脚本（辅助 NNDeploy 部署）。

---

如需补充或重分类某些文档，请在仓库中直接编辑对应 Markdown 文件或在 issue 中提出。是否需要我为每个文件添加更详尽的说明或生成带锚点的目录？

