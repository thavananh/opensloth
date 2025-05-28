# HyperSloth - Framework "Thần Tốc" Huấn Luyện LLM 🦥⚡

<p align="center">
    <img src="../images/hpsloth.webp" alt="Con lười nhưng nhanh" width="200" />
</p>

## 🚨 KHÔNG PHẢI QUẢNG CÁO KHÓA HỌC!

**Dự án này hoàn toàn miễn phí và mở nguồn. Không bán khóa học, không xin ly cà phê.** Mục đích đơn thuần là đóng góp cho nền khoa học "nước nhà" 🇻🇳  
_Ai bảo train LLM phải chậm như rùa? Ai bảo phải mua khóa học mới biết cách train?_

## 🎯 Tuyên Ngôn HyperSloth

**HyperSloth không chỉ là một phần mềm crack. Đó là tuyên ngôn của bản thân tôi trong hành trình của một kỹ sư AI Việt Nam khát khao tạo ra giá trị lõi, và tham gia sâu rộng hơn vào chuỗi giá trị công nghệ AI toàn cầu.**

_Đây không chỉ là code, mà là khát vọng góp phần đưa AI Việt Nam lên bản đồ thế giới._ 🇻🇳🚀

---

## 📞 Hỗ Trợ (Miễn Phí Luôn Nha)

🔥 **GitHub Issues**: [Báo lỗi tại đây](https://github.com/anhvth/HyperSloth/issues) _(không phải group Facebook đâu)_

---

## 🚀 HyperSloth Là Cái Gì Vậy?

**HyperSloth** = Con lười + Tốc độ ánh sáng ⚡

Nói đơn giản: Framework này sinh ra để train LLM với tốc độ vượt trội. Dựa trên Unsloth nhưng "buff thêm steroid" để train trên nhiều GPU.

_Tại sao tên "Sloth" (con lười) mà lại nhanh? Vì đời là thế, nghịch lý mà!_ 🤷‍♂️

### ✨ Tính Năng "Thần Thánh"

#### 🔄 Memory-mapped Gradient Sync

- Hỗ trợ cả 2 backend: **NCCL** (production) và **/dev/shm** (học tập/thử nghiệm)
- Communication qua shared memory _(nhanh hơn cả tin đồn)_
- Băng thông tối ưu đến mức "không thể tin nổi"

#### ⚡ Multi-GPU mà Không Đau Đầu

- Sync trọng số tự động _(không cần cầu nguyện)_
- Consistency đảm bảo 100% _(hoặc refund)_
- Overhead communication = gần như 0 _(không phải bug đâu)_

#### 🎯 Loss Types Linh Hoạt

- **Response-only**: Chỉ train phần trả lời _(thông minh đúng không?)_
- **Full sequence**: Train cả chuỗi _(cho ai thích khổ)_

#### 🛠️ Chat Template Fixes

- Sửa bug tokenizer _(tự nhiên các framework khác không biết)_
- Xử lý "think" tags đúng cách _(rocket science!)_

---

## 📊 So Sánh "Tàn Khốc"

### Thời Gian Huấn Luyện (4x RTX 4090)

| Framework      | Thời Gian   | VRAM     | Ghi Chú                   |
| -------------- | ----------- | -------- | ------------------------- |
| **HyperSloth** | **19 phút** | **6 GB** | _Nhanh đến không tin nổi_ |
| LlamaFactory   | 30 phút     | 21 GB    | _Chậm + tốn RAM_          |
| Unsloth (1X)   | ~70 phút    | 6 GB     | _Chậm như sên bò_         |

> 🔥 **HyperSloth**: Nhanh hơn 37%, ít VRAM hơn 71%... _các framework khác đi về đi!_

### Chất Lượng Training

Đường cong loss y hệt LlamaFactory. _Vậy mà nhanh hơn và ít tốn RAM hơn. Ma thuật à?_ 🪄

| HyperSloth Tensorboard                         | LlamaFactory Tensorboard                           |
| ---------------------------------------------- | -------------------------------------------------- |
| ![HyperSloth TB](../images/hyper-sloth-tb.png) | ![LlamaFactory TB](../images/llama-factory-tb.png) |

---

## 🛠️ Cài Đặt _(3 Bước Đơn Giản)_

```bash
# Bước 1: Cài đặt (không cần tháng đầu miễn phí gì cả)
pip install git+https://github.com/anhvth/HyperSloth.git

# Bước 2: Tạo config (AI sẽ làm hộ bạn)
hypersloth-init

# Bước 3: Train thôi! (ngồi uống cà phê chờ kết quả)
hypersloth-train ./hs_training_config.py
```

**Yêu cầu hệ thống:** Python 3.8+, PyTorch, CUDA, 2+ GPU NVIDIA _(không hỗ trợ CPU vì... chậm quá!)_

---

## 🚀 Hướng Dẫn _(Cho Người Lười)_

### Config File Mẫu _(Copy Paste Thôi)_

```python
from HyperSloth.hypersloth_config import *

# Cấu hình "thần thánh"
hyper_config_model = HyperConfig(
    grad_dir="/dev/shm/hypersloth",  # Nhanh nhất, tin tôi đi
    data=DataConfig(
        dataset_name_or_path="your_dataset.json",
        num_samples=5000,  # Ít thôi, đừng tham lam
        instruction_part="<|im_start|>user\n",
        response_part="<|im_start|>assistant\n",
    ),
    training=TrainingConfig(
        gpus=[0, 1, 2, 3],  # Nhiều GPU = nhiều tiền điện
        loss_type="response_only",  # Thông minh chọn cái này
    ),
    fast_model_args=FastModelArgs(
        model_name="Qwen/Qwen3-8B",  # Hoặc model nào đó
        max_seq_length=2048,
    ),
)
```

### Kaggle Notebook _(Miễn Phí GPU)_

Vào link này train free: `https://www.kaggle.com/code/anhvth226/kaggle-mistral-7b-hypersloth-notebook`

---

## 🛡️ Troubleshooting _(Khi Máy "Nổi Điên")_

### CUDA Out of Memory

```python
# Giảm batch size (đừng tham lam)
per_device_train_batch_size=2
gradient_accumulation_steps=16
```

### GPU Sync Issues

```bash
# Restart lại đi, đời có gì khó khăn
rm -rf /dev/shm/hypersloth/*
nvidia-smi  # Xem GPU còn sống không
```

---

## 🤝 Đóng Góp _(Miễn Phí + Tự Nguyện)_

- **Bug Report**: [GitHub Issues](https://github.com/anhvth/HyperSloth/issues) _(mô tả chi tiết, đừng chỉ viết "không chạy được")_
- **Feature Request**: Thảo luận trong GitHub Discussions
- **Code**: Fork, code, pull request _(chuẩn GitHub workflow)_

---

## 📞 Liên Hệ

👨‍💻 **Author**: [@anhvth](https://github.com/anhvth)  
⚡ **Response Time**: Khi nào rảnh _(không cam kết 24/7)_

---

## 📄 License & Lời Cuối

**MIT License** - Dùng free, sửa free, redistribute free. _Chỉ xin đừng claim là của bạn làm._

---

**🇻🇳 Made in Vietnam với tình yêu khoa học (và hơi nhiều cà phê) ☕**

_Đây là đóng góp cho cộng đồng AI Việt Nam. Không bán khóa học, không thu phí, không spam inbox, không xin ly cà phê._  
_Mục tiêu: Làm cho AI training nhanh hơn, rẻ hơn, dễ tiếp cận hơn!_ 🚀

```bash
# Bắt đầu ngay hôm nay!
pip install git+https://github.com/anhvth/HyperSloth.git
hypersloth-init && hypersloth-train ./hs_training_config.py
```
