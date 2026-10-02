# Báo cáo P1 — audit và ablation số lượng candidate PPO

Ngày xác nhận: 02/10/2026.

Branch triển khai: `codex/p1-candidate-ablation`.

Tài liệu kế hoạch: `docs/research/ppo_ablation_training_plan_vi.md`.

## 1. Kết luận ngắn

P1 đã hoàn thành phần code, audit offline và smoke end-to-end. Không gian action
hiện hỗ trợ cùng một pool vật lý 10 cell với ba kích thước prefix:

| K | Action | Observation |
|---:|---:|---:|
| 10 | 11 | 109 |
| 5 | 6 | 59 |
| 3 | 4 | 39 |

Ranked selector giữ C1 là A3 target, đưa alternative có SINR hiện tại tốt nhất
vào C2, rồi đưa một alternative khác có SINR dự báo nhân quả +90 ms tốt nhất
vào C3. Các cell còn lại giữ thứ tự legacy. Vì vậy K=3 là prefix của K=5 và
K=5 là prefix của K=10; selector không thay pool topology ban đầu.

Audit train-only cho thấy K=3 giữ được ít nhất một alternative hữu ích trong
100% positive state quan sát được. Tuy nhiên số positive chỉ khoảng 5-7% tổng
state, nên **chưa được kết luận K=3 là tốt nhất**. Cần chạy E00-E03 đủ 100k trên
validation để chọn K*. Các smoke 1.024 step chỉ xác nhận mechanics, không phải
kết quả nghiên cứu.

## 2. Ranked selector đã khóa

Thứ tự `ranked_v1` tại mỗi completed-A3 decision:

1. C1: target của completed A3+TTT.
2. C2: cell khác C1 trong pool có SINR hiện tại lớn nhất.
3. C3: cell khác C1/C2 có projected SINR +90 ms lớn nhất.
4. C4-C10: các cell còn lại theo đúng thứ tự legacy.

Projected SINR dùng hồi quy tuyến tính trên tối đa 100 ms lịch sử SINR của
physical cell ID. Chỉ dữ liệu có `tic <= decision tic` được dùng; test thay toàn
bộ suffix tương lai xác nhận observation, candidate và mask tại decision không
đổi.

Không có hard Qin/suitability mask ở P1. Mask chỉ đánh dấu slot vật lý tồn tại.
Một cell chất lượng thấp vẫn còn trong K lớn; nó chỉ bị đẩy xuống thấp bởi
selector. Đây là một inductive bias có chủ ý và phải công bố trong bài báo:
actor không tự tìm trong toàn bộ 21 cell, mà chọn trong shortlist nhân quả do
selector topology/quality cung cấp.

## 3. Audit counterfactual train-only

Lệnh đã chạy:

```powershell
.\.venv\Scripts\python.exe run.py audit_candidates `
  --data-dir data/uav_100m_grid100m_dev_processed `
  --speeds 30 50 70 90 `
  --routes-per-speed 20 `
  --max-decisions-per-route 4 `
  --decision-stride 10 `
  --horizons 0.5 1.0 2.0 `
  --checkpoint results/models/uav_100m_grid100m_topology109_v1_s17/smdp_ppo_final.pt `
  --output-dir results/audit/p1_candidates_s17
```

### 3.1. Phạm vi và chống leakage

- Chỉ lấy 80 route thuộc split train: 20 route cho mỗi tốc độ 30/50/70/90.
- Split seed 17, route-sampling seed 17.
- Tối đa 4 decision mỗi route, lấy decision index 0/10/20/30 để route nhiều
  handover không chi phối mẫu.
- Hai nguồn state: Always-HO-C1 và policy 109/11 hiện có.
- Thu được 589 state: 295 từ Always-HO-C1 và 294 từ policy.
- Mỗi state clone toàn bộ protocol state rồi thử STAY và C1-C10.
- Sau action đầu, mọi branch cùng dùng continuation Always-HO-C1.
- Future 0,5/1/2 s chỉ dùng để tạo nhãn counterfactual, không tham gia rank,
  observation hoặc action mask.
- Tập validation/test không được load vào audit.

Một alternative C2-C10 được gọi là hữu ích khi nó tạo **nhiều delivered Mbit
hơn** reference, đồng thời không tăng HOF count, RLF count, ping-pong count hay
RLF recovery time. Báo hai reference: C1 và best(STAY,C1).

### 3.2. Coverage tổng hợp

| Horizon | Positive vs C1 | K3 | K5 | K10 | Positive vs best(STAY,C1) | K3 | K5 | K10 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0,5 s | 39/589 | 100% | 100% | 100% | 34/589 | 100% | 100% | 100% |
| 1,0 s | 38/589 | 100% | 100% | 100% | 30/589 | 100% | 100% | 100% |
| 2,0 s | 38/589 | 100% | 100% | 100% | 31/589 | 100% | 100% | 100% |

Positive state tồn tại ở cả 4 tốc độ và cả hai state source. Các physical cell
hữu ích nằm ở mọi vị trí legacy C2-C10; vì vậy coverage K=3 không phải do
alternative hữu ích vốn luôn nằm sẵn trong ba slot legacy. Ranked selector đã
đưa ít nhất một cell hữu ích vào C2/C3 trong mẫu audit.

Caveat quan trọng:

- 100% là conditional coverage trên số positive tương đối nhỏ, không phải xác
  suất bảo đảm trên mọi route.
- Nhiều action có thể cùng hữu ích trong một state, nên tổng số lần xuất hiện
  slot lớn hơn số positive state.
- Continuation Always-HO-C1 là một giả định đánh giá cố định; nó không thay thế
  train policy dài hạn.
- Nếu E03 100k thua rõ E02/E01 về QoS hoặc convergence thì không chọn K=3 chỉ
  vì audit coverage bằng 100%.

Artifacts (không commit vì nằm trong `results/`):

- `results/audit/p1_candidates_s17/candidate_audit_summary.json`
- `results/audit/p1_candidates_s17/candidate_audit_coverage.csv`
- `results/audit/p1_candidates_s17/candidate_audit_states.csv`
- `results/audit/p1_candidates_s17/candidate_audit_states.json`

Summary lưu SHA-256 của 160 CSV đã dùng và `dataset_provenance.json`, danh sách
route, split/sampling rule, checkpoint nguồn và định nghĩa selector/useful.

## 4. Thay đổi code

- `train_ppo` có `--candidate-count {3,5,10}` và
  `--candidate-order {legacy,ranked_v1}`.
- Observation/action/mask thay đổi đồng bộ theo K.
- Checkpoint lưu K, order, action names, observation schema; schema không tương
  thích bị từ chối.
- `evaluate_ppo` và `benchmark` tự đọc K/order từ checkpoint, không còn ép
  checkpoint phải có 11 action.
- Thêm command `audit_candidates` để tái lập audit train-only.
- Giữ tương thích checkpoint legacy: K=10 legacy vẫn dùng version
  `a3_topology109_v1`; checkpoint cũ thiếu field order/count vẫn được hiểu là
  K=10 legacy.

## 5. Kiểm thử và smoke

Toàn bộ test sau P1:

```text
Ran 143 tests
OK
```

Các test mới cover prefix K3/K5/K10, causal ranked ordering, dimension/action
space, giá trị config không hợp lệ, clone counterfactual không làm đổi engine
gốc và coverage chỉ tính trên positive state.

Ba smoke dùng cùng seed/split/data, 1.024 macro-step, một PPO update, không đọc
test:

| Arm smoke | K | Obs/action | Entropy ban đầu | Validation rate | RR ratio vs 3GPP | Ghi chú |
|---|---:|---:|---:|---:|---:|---|
| E01 | 10 | 109/11 | 2,3979 ≈ ln(11) | 20,925 Mbps | 89,6% | mechanics pass |
| E02 | 5 | 59/6 | 1,7917 ≈ ln(6) | 20,732 Mbps | 88,4% | mechanics pass |
| E03 | 3 | 39/4 | 1,3862 ≈ ln(4) | 19,284 Mbps | 81,7% | mechanics pass |

Normalized entropy đều bằng 1,000, xác nhận policy khởi tạo chia đều trên số
action hợp lệ. Cả ba run có cùng dataset content hash
`0c50e6cbb899a9fcef27231cc7bb508539c53e209ded2a53738fc57f2873fcb4` và
`test_evaluated=false`.

Không dùng các metric sau một update để chọn K. Deterministic argmax ở policy
gần-uniform rất nhạy với sai khác nhỏ của logits; E03 chọn STAY 90,1% trong
smoke không chứng minh K=3 sẽ hội tụ về STAY ở 100k.

Benchmark validation E03 đã chạy đủ 150 route và load đúng checkpoint 39/4.
Kết quả 1.024-step PPO kém 3GPP về rate như dự kiến đối với smoke; mục đích run
này chỉ là xác nhận loader, mapping và output benchmark.

## 6. Bước tiếp theo bắt buộc: screening E00-E03 100k

Chưa khóa K*. Chạy tuần tự bốn arm từ đầu, không resume/warm-start, giữ
`--skip-final-test`. E00 là baseline F1 K10 legacy; E01 chỉ đổi order; E02 chỉ
đổi K 10→5; E03 chỉ đổi K 5→3.

Các phần chung của mỗi lệnh:

```powershell
.\.venv\Scripts\python.exe run.py train_ppo `
  --data-dir data/uav_100m_grid100m_dev_processed `
  --speeds 30 50 70 90 `
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

Thay `<ARM_FLAGS>` lần lượt bằng:

```powershell
# E00
--candidate-count 10 --candidate-order legacy `
--experiment-id E00_f1_k10_legacy_s17 --parent-id F1_smoke_v3_s17 `
--changed-factor screening_budget `
--output-dir results/models/ablation_v2/E00_f1_k10_legacy_s17

# E01
--candidate-count 10 --candidate-order ranked_v1 `
--experiment-id E01_ranked_k10_s17 --parent-id E00_f1_k10_legacy_s17 `
--changed-factor candidate_order `
--output-dir results/models/ablation_v2/E01_ranked_k10_s17

# E02
--candidate-count 5 --candidate-order ranked_v1 `
--experiment-id E02_ranked_k5_s17 --parent-id E01_ranked_k10_s17 `
--changed-factor candidate_count_10_to_5 `
--output-dir results/models/ablation_v2/E02_ranked_k5_s17

# E03
--candidate-count 3 --candidate-order ranked_v1 `
--experiment-id E03_ranked_k3_s17 --parent-id E02_ranked_k5_s17 `
--changed-factor candidate_count_5_to_3 `
--output-dir results/models/ablation_v2/E03_ranked_k3_s17
```

Chỉ so sánh validation: QoS gate aggregate/từng speed, relative rate và p10,
RLF-time, HOF, PP, learning curve, action distribution và uncertainty theo
route. Không benchmark test/zero-shot cho đến khi đã freeze candidate K*.
