# A3-triggered topology-aware multi-candidate PPO

## Mục tiêu

A3+TTT vẫn quyết định **khi nào agent được quyền hành động**, nhưng A3 không còn
ép PPO chỉ được handover tới một target. Ở mỗi decision, PPO chọn một trong 11
action:

```text
0       STAY
1..10   HO_C1 .. HO_C10
```

`STAY` từ chối event A3 hiện tại. `HO_Ci` thực hiện handover tới cell/sector đã
được chốt trong slot `Ci`. Đây là cell ID (`bs_0..bs_20`), không phải site ID.

## Cách tạo 10 candidate

Candidate được tính hoàn toàn từ dữ liệu ở hiện tại và quá khứ rồi giữ nguyên
cho tới khi action được áp dụng:

1. `C1` luôn là target đã hoàn tất A3+TTT. Vì vậy baseline 3GPP Always-HO dùng
   `HO_C1` và vẫn giữ nguyên ý nghĩa cũ.
2. Thêm các sector còn lại trên cùng physical site với serving cell.
3. Tìm các site vòng một từ tọa độ topology; xếp hạng site theo RSRP tốt nhất
   hiện tại, rồi thêm toàn bộ sector của hai site mạnh nhất.
4. Nếu chưa đủ 10, điền bằng các cell còn lại theo RSRP hiện tại giảm dần.
5. Loại trùng, loại serving cell. Nếu dataset có ít hơn 11 cell, slot thiếu
   được mask, không cho PPO chọn.

Topology của bộ 1.060 route được đọc từ `dataset_provenance.json`: 7 site,
21 sector, ba sector/site với hướng 0/120/240 độ. Tọa độ chỉ dùng để xác định
site lân cận; không được đưa trực tiếp vào observation và không tiết lộ tương
lai của UAV. SHA-256 của file provenance được lưu trong dataset split manifest;
evaluate/benchmark từ checkpoint mới sẽ từ chối nếu topology bị thiếu hoặc đổi.

## Observation 109 chiều

Khối đầu giữ nguyên 19 feature cũ cho serving và `C1`. Mỗi candidate `C2..C10`
có thêm 10 feature:

1. spectral efficiency hiện tại;
2. chênh lệch spectral efficiency so với serving;
3. A3 margin hiện tại;
4. A3 margin trung bình từ đầu event;
5. A3 margin nhỏ nhất;
6. slope A3 margin;
7. độ lệch chuẩn A3 margin;
8. slope SINR;
9. khoảng cách SINR tới `Qin`;
10. cờ candidate có phải previous cell hay không.

Tổng số chiều là `19 + 9 × 10 = 109`. Mọi slope/history chỉ sử dụng mẫu tới
decision hiện tại, không đọc mẫu tương lai. Agent có thể học xu hướng một sector
đang tốt dần hay xấu dần dù hướng bay của UAV được chọn ngẫu nhiên.

## Action mask và xác suất ban đầu

Action mask đi cùng observation trong quá trình sample, lưu rollout, PPO update
và deterministic evaluation. Nhờ vậy xác suất cũ và mới luôn được tính trên
cùng tập action hợp lệ.

Policy head được khởi tạo với weight và bias bằng 0. Trên dataset 21 cell, cả 11
action hợp lệ nên policy ban đầu có xác suất đúng `1/11` cho từng action. Đây chỉ
là phân phối lúc khởi tạo; sau update PPO, xác suất phải trở nên không đều theo
observation và reward.

## Những phần cố ý giữ nguyên

- MLP PPO, SMDP decision-to-decision và duration-aware GAE.
- A3, TTT, HO preparation/execution, HOF, RLF và ping-pong trong protocol engine.
- Reward legacy hiện tại. Thay action trước giúp tách ảnh hưởng của target
  selection khỏi ảnh hưởng của một reward mới.
- Cấu hình runtime: carrier 2,1 GHz, bandwidth 10 MHz, `dt = 10 ms`.

## Tương thích checkpoint

Checkpoint cũ có input 19 và 2 action không thể benchmark trong environment
109/11. Cần train model mới. Loader kiểm tra số action, còn PPO kiểm tra cả shape
observation trước rollout/evaluation để tránh chạy nhầm model mà không báo lỗi.

## Train và benchmark

```powershell
.\.venv\Scripts\python.exe run.py train_ppo `
  --data-dir data/uav_100m_grid100m_dev_processed `
  --speeds 30 50 70 90 `
  --total-macro-steps 100000 `
  --rollout-steps 512 `
  --batch-size 64 `
  --update-epochs 4 `
  --learning-rate 0.00005 `
  --evaluation-interval 2048 `
  --seed 17 `
  --split-seed 17 `
  --output-dir results/models/uav_100m_grid100m_topology109_v1_s17

.\.venv\Scripts\python.exe run.py benchmark `
  --data-dir data/uav_100m_grid100m_dev_processed `
  --checkpoint results/models/uav_100m_grid100m_topology109_v1_s17/smdp_ppo_final.pt `
  --split test `
  --speeds 30 50 70 90 `
  --zero-shot-speeds 120 150 `
  --output-dir results/benchmark/uav_100m_grid100m_topology109_v1_s17
```

Benchmark ghi tổng số `HO_C1..HO_C10` riêng để kiểm tra PPO có thật sự dùng các
target ngoài A3 hay chỉ học lại policy Always-HO.
