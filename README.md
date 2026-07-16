# 电商用户评论智能情感分析系统

基于京东 300 万条真实评论数据（JD Full），从零构建的多级情感分析流水线。不只是判断"好评差评"，还能精确定位"好在哪、差在哪"。

## 🎯 项目概览

| 能力 | 实现 |
|------|------|
| 整体情感分类 | 二分类（好/坏）· 三分类（好/中/坏） |
| 细粒度方面级分析 | 6 维度（商品质量/外观设计/使用体验/物流配送/价格/客服服务） |
| 模型压缩 | BERT(400MB) → BiLSTM(81MB)，精度保留 96% |
| Web 部署 | Flask API + Streamlit 前端 + ngrok 公网访问 |

## 📊 核心结果

| 模型 | 二分类 F1 | 三分类 F1 | 方面级 F1 |
|------|:--:|:--:|:--:|
| TF-IDF + 随机森林 | 0.761 | — | — |
| FastText | 0.799 | 0.596 | — |
| **BERT-base-chinese** | **0.843** | 0.669 | — |
| BERT + DeepSeek 标签 | — | **0.892** | — |
| **BERT 方面级（6 维度）** | — | — | **0.865** |
| BiLSTM 蒸馏（81MB） | — | 0.860 | 0.827 |

> 🏆 关键发现：31770 条 AI 标注的干净标签训练效果远超 16 万条评分映射的噪声标签（0.892 vs 0.669），验证了 **标注质量 > 数据量**。

## 🏗️ 技术架构

```
数据清洗 → 传统ML(RF) → 浅层DL(FastText) → 预训练(BERT) → 方面级分析 → 蒸馏(BiLSTM) → Web部署
```

### 模型演进

| 阶段 | 模型 | 技术要点 |
|------|------|------|
| 基线 | TF-IDF + 随机森林 | 词袋模型 · GridSearchCV 调参 · 自定义 tokenizer |
| 升级 | FastText | 词向量 · 子词 n-gram · OOV 处理 |
| 深度 | BERT-base-chinese | Transformer · Self-Attention · 单头/双头/多标签 |
| 蒸馏 | BERT → BiLSTM | 知识蒸馏 · KL 散度 · 软硬标签混合（α=0.9, T=3.0） |

### 方面级情感分析

定义了 6 个业务维度，每个维度独立判断正面/负面/未提及：

| 商品质量 | 外观设计 | 使用体验 | 物流配送 | 价格 | 客服服务 |
|:--:|:--:|:--:|:--:|:--:|:--:|
| 0.845 | 0.863 | 0.830 | 0.889 | 0.946 | 0.818 |

> 标注数据：DeepSeek API 辅助标注 50000 条，提示词工程驱动

### 双头 BERT 对比实验

同一模型、同一数据、两个输出头——公平对比"直接分类"和"方面反推整体"：

| 预测方式 | 三分类 F1 |
|------|:--:|
| 直接三分类 | **0.892** |
| 方面反推（理想上界） | 0.721 |
| 方面反推（模型预测） | 0.679 |

> 结论：整体 ≠ 部分之和。上下文、转折语义无法被维度标签求和替代。

### 模型蒸馏

| | BERT（老师） | BiLSTM（学生） | 保留 |
|------|:--:|:--:|:--:|
| 三分类 F1 | 0.892 | 0.860 | 96% |
| 方面级 F1 | 0.865 | 0.827 | 95% |
| 参数量 | 1.1 亿 | 2000 万 | 5x |
| 模型大小 | 400 MB | 81 MB | 5x |
| 推理速度 | 50ms (GPU) | 5ms (CPU) | 10x |

## 📁 项目结构

```
ec_review_analysis/
├── baseline/                   # TF-IDF + 随机森林 基线模型
│   ├── config.py               # 路径与参数配置
│   ├── process_data.py         # jieba 分词预处理
│   └── train.py                # 训练脚本（含 GridSearchCV）
├── FastTest/                   # FastText 模型
│   ├── config.py
│   ├── dataprocess_fasttext.py # CSV → __label__ 格式
│   └── train_fasttext.py       # 三分类/二分类训练
├── bert/                       # BERT 系列（核心模块）
│   ├── config.py               # 统一路径配置
│   ├── train.py                # 单头 BERT（二分类/三分类）
│   ├── train_aspect.py         # 方面级多标签 BERT
│   ├── train_combined.py       # 双头 BERT（三分类+方面级）
│   ├── distill.py              # 知识蒸馏（BERT→BiLSTM）
│   ├── predict.py              # 推理预测脚本
│   ├── app.py                  # Flask API 服务
│   └── frontend.py             # Streamlit 前端页面
├── deepseek_api/               # DeepSeek API 标注 & 对比
│   ├── predict.py              # 零样本情感分类对比
│   ├── aspect_label.py         # 方面级数据标注
│   ├── label_aspect_3class.py  # 联合标注（三分类+方面级）
│   └── test_1vs5.py            # 消融实验（1星 vs 5星）
├── Initial_data/               # 原始数据目录（300万条）
├── Cleaned_data_*/             # 清洗后各版本数据
└── cleandata*.py               # 数据清洗脚本（7步Pipeline）
```

## 🚀 快速开始

### 安装依赖

```bash
pip install torch transformers scikit-learn fasttext jieba flask streamlit pandas numpy tqdm
```

### 启动推理服务

```bash
cd bert
python app.py
# Flask 服务启动于 http://127.0.0.1:5000
```

### 启动前端页面

```bash
cd bert
streamlit run frontend.py
# 访问 http://localhost:8501
```

### API 调用示例

```bash
# 单条预测
curl -X POST http://127.0.0.1:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text":"屏幕很清晰但电池不耐用"}'

# 批量预测
curl -X POST http://127.0.0.1:5000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts":["物流很快","质量太差","还行吧"]}'

# 健康检查
curl http://127.0.0.1:5000/health
```

### 响应格式

```json
{
    "sentiment": "中评",
    "sentiment_id": 1,
    "aspects": {
        "商品质量": "正面",
        "外观设计": "未提及",
        "使用体验": "负面",
        "物流配送": "未提及",
        "价格": "未提及",
        "客服服务": "未提及"
    }
}
```

## 🛠️ 技术栈

`Python` `PyTorch` `Transformers` `BERT` `FastText` `scikit-learn` `jieba` `Flask` `Streamlit` `DeepSeek API`

## 📝 数据说明

本项目使用京东公开评论数据集用于学术研究。数据集需自行获取，不包含在此仓库中。原始数据格式：

```
评分（1-5星）, 商品名, 评论文本
```

## ⚠️ 注意事项

- BERT 模型文件较大（~400MB），训练需 GPU（推荐 RTX 4090/5090）
- DeepSeek API 调用需自行申请 API Key
- 数据清洗脚本中的路径为 Windows 绝对路径，Linux 环境需调整
- 训练脚本中 `pretrained_model` 本地指向 `D:\dev\NLP\day04\bert-base-chinese`，服务器上改为 `"bert-base-chinese"`

## 📄 License

MIT License
