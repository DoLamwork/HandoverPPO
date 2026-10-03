# Báo cáo triển khai P2 — observation ablation E04-E06

Ngày triển khai: 03/10/2026.

Branch: `codex/p2-observation-ablation`.

Parent: P1 đã khóa `candidate_count=3`, `candidate_order=ranked_v1`, structural
mask và reward legacy. P2 không đổi action trigger, STAY semantics, reward,
PPO hyperparameter hay MLP 64x64.

## 1. Kết quả triển khai

| Arm | Schema CLI | Observation | Thay đổi duy nhất so với parent |
|---|---|---:|---|
| E03 | `legacy_v1` | 39 | Baseline P1 |
| E04 | `rolling_v1` | 39 | History theo physical cell ID |
| E05 | `monotonic_v1` | 39 | Bỏ clip SE sớm, normalization monotonic |
| E06 | `qos_full_v1` | 62 | 17 global + 15 feature cho mỗi C1-C3 |

Checkpoint lưu `observation_schema_key`, feature names và observation size.
Evaluate/benchmark tự khôi phục schema từ checkpoint; loader từ chối checkpoint
khác schema, kể cả khi action space giống nhau.

## 2. E04 — rolling history theo physical cell ID

- Short window: `[t-100 ms,t]`, 11 sample khi đủ lịch sử.
- Long window: `[t-500 ms,t]`, 51 sample khi đủ lịch sử.
- History được lưu cho mọi measured sector, không bắt đầu lại khi A3 bắt đầu.
- Serving/candidate history được truy xuất bằng physical cell ID hiện tại.
- RSRP gap lịch sử tính lại trên cùng timestamp của serving và candidate.
- Giữ layout 19 feature C1 + 10 feature cho C2/C3, tổng 39 chiều.

Do đó E04 chỉ kiểm tra ảnh hưởng của history; không đồng thời tăng input size.

## 3. E05 — normalization monotonic

E05 giữ nguyên history/layout 39 chiều của E04 nhưng:

- SE dùng SINR thật, không clip trước về `[-10,10]` dB.
- Tính SE ổn định số bằng `logaddexp`, tránh overflow ở SINR lớn.
- Feature unsigned dùng `x/(s+x)`.
- Feature signed dùng `x/(s+abs(x))`.
- Cùng scale được dùng cho serving và mọi candidate; không normalize riêng slot.
- Scale được cố định trong config/checkpoint, không fit trên validation/test.

Scale chính: SE=4 bit/s/Hz, speed=90 km/h, quality gap=20 dB, slope=100
dB/s, RSRP margin=10 dB, std=5 dB.

## 4. E06 — 17 global + 15 feature/candidate

Global block gồm: serving SE/Qout gap, slope 100/500 ms, std và Qout persistence,
N310, N311, T310 active/remaining, residence, time từ HO gần nhất, PP window
remaining, số HO thành công 1 giây, best measured SE, history coverage và speed.

Mỗi candidate C1-C3 có cùng block: valid, SE, delta-SE, Qin gap, slope 100/500
ms, std, Qin suitability fraction, p10-Qin, RSRP gap/slope, projected Qin margin
+90 ms, previous-cell, same-site và A3-target flags.

Projected SINR dùng hoàn toàn dữ liệu causal:

```text
mean SINR 100 ms + 0,09 × slope 100 ms - 1 × std 100 ms
```

Không đọc SINR thật tại `t+90 ms` để tạo observation hoặc action. Giá trị thật
tương lai chỉ được dùng offline để audit sai số dự báo.

Speed được truyền dưới dạng metadata của `RadioTrace`. E06 từ chối trace thiếu
speed thay vì âm thầm gán một giá trị có thể nhầm với UAV đứng yên.

## 5. Audit train/validation, không dùng test

Lệnh `audit_observations` chạy Always-HO-C1 làm continuation cố định, lấy 20
route/tốc độ và tối đa 4 decision/route. Mỗi split có 80 route, 320 decision và
960 dự báo candidate.

| Split | Prediction samples | MAE | RMSE | Bias |
|---|---:|---:|---:|---:|
| Train | 960 | 4,124 dB | 6,614 dB | +0,017 dB |
| Validation | 960 | 4,114 dB | 6,952 dB | -0,510 dB |

Validation MAE theo speed: 30=3,392; 50=4,059; 70=3,541; 90=5,465 dB. Sai số
lớn nhất ở 90 km/h phù hợp với việc extrapolation tuyến tính khó hơn khi radio
thay đổi nhanh. Feature projected vì vậy chỉ là tín hiệu xu hướng nhiễu, không
được mô tả là dự đoán chắc chắn BS tốt trong tương lai.

Saturation audit cho thấy các role bit (`valid`, `is_a3_target`) nằm ở biên theo
định nghĩa. `N311 progress` bằng 0 ở 100% train và khoảng 99% validation
decision; đây là feature thưa cần theo dõi khi screening. Các feature liên tục
monotonic không xuất hiện saturation hàng loạt trong mẫu audit.

Artifacts cục bộ:

- `results/audit/p2_observations_train_s17/`
- `results/audit/p2_observations_validation_s17/`

Ví dụ tái lập train audit:

```powershell
.\.venv\Scripts\python.exe run.py audit_observations `
  --data-dir data/uav_100m_grid100m_dev_processed `
  --speeds 30 50 70 90 `
  --split train --split-seed 17 --sample-seed 17 `
  --routes-per-speed 20 --max-decisions-per-route 4 `
  --candidate-count 3 --candidate-order ranked_v1 `
  --observation-schema qos_full_v1 `
  --output-dir results/audit/p2_observations_train_s17
```

## 6. Smoke end-to-end

Ba arm chạy 128 macro-step, một PPO update, K=3/ranked_v1, reward legacy và
`skip-final-test`. Smoke dùng split mechanics 98/1/1 để validation nhỏ; không
được dùng làm kết quả nghiên cứu.

| Arm | Obs/action | Entropy | Validation rate | Hành vi argmax |
|---|---:|---:|---:|---|
| E04 | 39/4 | 1,3863 | 16,570 Mbps | 90,8% STAY |
| E05 | 39/4 | 1,3863 | 16,328 Mbps | 95,1% STAY |
| E06 | 62/4 | 1,3863 | 16,453 Mbps | 100% STAY |

Các số này chỉ xác nhận shape, collection, update, validation, safe checkpoint
load và metadata. Một update với deterministic argmax không cho biết hội tụ.

Toàn bộ test sau P2: 155 test, tất cả pass.

## 7. Lệnh screening 100k tiếp theo

Cấu hình chung:

```powershell
.\.venv\Scripts\python.exe run.py train_ppo `
  --data-dir data/uav_100m_grid100m_dev_processed `
  --speeds 30 50 70 90 `
  --candidate-count 3 `
  --candidate-order ranked_v1 `
  --total-macro-steps 100000 `
  --rollout-steps 1024 `
  --batch-size 128 `
  --update-epochs 4 `
  --learning-rate 0.00005 `
  --evaluation-interval 10240 `
  --seed 17 `
  --split-seed 17 `
  --skip-final-test `
  <ARM_FLAGS>
```

Arm flags:

```powershell
# E04: chỉ đổi history so với E03
--observation-schema rolling_v1 `
--experiment-id E04_rolling_k3_s17 `
--parent-id E03_ranked_k3_s17 `
--changed-factor observation_history `
--output-dir results/models/ablation_v2/E04_rolling_k3_s17

# E05: chỉ đổi normalization so với E04
--observation-schema monotonic_v1 `
--experiment-id E05_monotonic_k3_s17 `
--parent-id E04_rolling_k3_s17 `
--changed-factor observation_normalization `
--output-dir results/models/ablation_v2/E05_monotonic_k3_s17

# E06: chỉ đổi feature block/size so với E05
--observation-schema qos_full_v1 `
--experiment-id E06_qos_full_k3_s17 `
--parent-id E05_monotonic_k3_s17 `
--changed-factor observation_feature_block `
--output-dir results/models/ablation_v2/E06_qos_full_k3_s17
```

Chạy tuần tự. Không warm-start vì observation schema khác nhau. Không dùng test
hoặc zero-shot để chọn arm. So sánh selected checkpoint và last scheduled
validation để phát hiện lại checkpoint-selection effect của P1.
