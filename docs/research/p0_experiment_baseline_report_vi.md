# Báo cáo chuẩn hóa nền thí nghiệm PPO — P0

Ngày xác nhận: 02/10/2026.

Tài liệu kế hoạch gốc: `docs/research/ppo_ablation_training_plan_vi.md`.

## 1. Kết luận ngắn

Nền F1 đã đủ để chạy các arm ablation **từ đầu** với cùng một hợp đồng thí
nghiệm. Mỗi run hiện lưu được cấu hình, source/data/topology fingerprint,
validation reference, toàn bộ validation checkpoint, log huấn luyện và trạng
thái QoS. Tập test có thể bị khóa bằng `--skip-final-test`.

Không được gọi checkpoint hiện tại là exact-resume. Checkpoint đã lưu
model/optimizer/counter và RNG Python/NumPy/Torch, nhưng chưa lưu toàn bộ trạng
thái protocol/episode/collector/checkpoint scheduler. Run xác nhận 300k phải
chạy từ đầu cho đến khi exact-resume có test tương đương bit-for-bit.

## 2. Baseline F1 đã khóa

| Thành phần | Giá trị |
|---|---|
| Dataset | `data/uav_100m_grid100m_dev_processed` |
| Split | 700 train / 150 validation / 150 test, split seed 17 |
| Speed train/validation/test | 30, 50, 70, 90 km/h |
| Action | STAY + HO_C1...HO_C10, 11 action |
| Observation | `a3_topology109_v1`, 109 feature |
| Candidate K | 10 |
| Mask | Chỉ structural slot availability |
| STAY | Từ chối completed-A3 event hiện tại |
| Reward | `legacy_handover_reward_v1` |
| Discount | `intra_macro_time_consistent_v1` |
| MLP | 64x64, tanh |
| PPO | LR 5e-5, rollout 1024, batch 128, 4 epoch |
| PPO khác | clip 0,2; entropy 0,01; value 0,5; grad 0,5 |
| SMDP | gamma/second 0,99; lambda/second 0,95 |
| Validation | 10.240 macro-step một lần |
| QoS gate | Mbps và trajectory-relative-rate đều >=98% aggregate và từng speed |

Lưu ý provenance: trace lịch sử được sinh với 4 GHz/20 MHz; runtime handover
đang dùng 2,1 GHz/10 MHz. Hai cấu hình phải được báo riêng, không mô tả như thể
trace đã được sinh lại ở 2,1 GHz/10 MHz.

## 3. Những thay đổi đã triển khai

### 3.1. Hợp đồng và provenance

- Mỗi run sinh `experiment_config.json` với experiment/parent/changed factor,
  K, candidate policy/order, mask, observation, history, normalization, reward,
  discount, STAY semantics, model, seed, budget và evaluation policy.
- Sinh `source_state.json` và `source_worktree.patch`; lưu Git revision/status,
  tracked patch hash và hash từng file source. Untracked source vì vậy vẫn nằm
  trong source manifest dù không xuất hiện trong Git patch.
- Dataset manifest có thêm `topology_sha256` và
  `dataset_content_sha256`. Content hash loại timestamp/absolute path nên ổn
  định giữa các lần chạy trên cùng data/split.
- Metadata đầy đủ được nhúng vào validation, latest và selected checkpoint.

### 3.2. Bảo vệ validation/test

- `--skip-final-test` không khởi tạo và không evaluate test environment.
- Test không tham gia checkpoint selection trong mọi chế độ.
- QoS gate yêu cầu đồng thời mean Mbps ratio và mean trajectory-relative-rate
  ratio, cả aggregate lẫn từng speed.
- Nếu không có checkpoint đạt gate, report ghi `QOS_INFEASIBLE` và selection
  reason `infeasible_*_best_effort`; không tạo `smdp_ppo_best_qos.pt`.
- Checkpoint được export bằng tên `smdp_ppo_selected.pt`; `latest` vẫn là weights
  ở cuối budget, tránh dùng tên `final` nhập nhằng.

### 3.3. SMDP discount nhất quán

Với `dt` giây:

```text
gamma_tick = gamma_per_second ** dt
R_macro = sum_l gamma_tick ** l * r_l
bootstrap_discount = gamma_per_second ** duration_seconds
```

Tick đầu dùng exponent 0. Environment giữ hai giá trị:

- `info["reward"]`: breakdown thô, không discount, dùng audit/report.
- `info["discounted_reward"]` và scalar reward trả cho PPO: reward F1 dùng để
  update.

GAE tiếp tục dùng discount theo duration. Có test segmentation invariance và
test environment trả đúng discounted reward.

### 3.4. Checkpoint contract

Checkpoint lưu thêm RNG Python, NumPy, Torch CPU và CUDA (nếu có), cùng contract:

```text
save_boundary = ppo_update_boundary
resume_capability = optimizer_and_rng_with_episode_restart
exact_resume = false
```

Khi load optimizer, RNG được restore. Collector bị reset có chủ ý. Checkpoint
có objective metadata khác candidate order/observation/action sẽ bị từ chối rõ
ràng khi đưa vào environment.

### 3.5. Logging

Mỗi update có thêm:

- primitive ticks, simulated seconds và macro-step theo speed/route;
- action count từng slot, mask availability và số action hợp lệ trung bình;
- probability mean/p10/p50/p90 từng action;
- normalized entropy `H/log(n_valid)` và conditional HO-candidate entropy;
- reward breakdown thô và discounted;
- macro-duration p50/p90/p99 theo action.

Validation có relative rate aggregate/từng speed, p10 relative rate, action
count/fraction từng slot, Mbps, RLF-time, HOF, PP và hai loại ratio tới 3GPP.

## 4. Kiểm thử

Toàn bộ test project ngày 02/10/2026:

```text
Ran 138 tests
OK
```

Sau khi sửa lỗi helper test, nhóm P0 trực tiếp đã chạy lại:

```text
Ran 40 tests in 3.348s
OK
```

Các invariants được cover gồm Always-HO/C1 protocol reference, causal
observation, candidate ID duy nhất và không phải serving, frozen action target,
mask khi sample/update/evaluate, uniform initial masked policy, SMDP GAE,
discount segmentation, checkpoint safe-load, QoS gate kép và schema mismatch.

## 5. Smoke thật 2.048 macro-step

Artifact chuẩn:

`results/models/ablation_v2/F1_smoke_v3_s17/`

Fingerprint:

- Dataset content SHA-256:
  `0c50e6cbb899a9fcef27231cc7bb508539c53e209ded2a53738fc57f2873fcb4`
- Source manifest SHA-256:
  `ee988475783d008f760ea7dd674e55ea143ae6a886615acf3c26ac6b7739a15b`

Kết quả mechanics:

- 2 rollout x 1.024, tổng 2.048 macro-step, 2 PPO update.
- Policy ban đầu có normalized entropy gần 1, đúng kỳ vọng phân bố gần đều.
- Validation 150 route chạy thành công.
- 3GPP validation mean rate: 23,024 Mbps.
- PPO smoke: 19,220 Mbps; Mbps ratio 83,47%; relative-rate ratio 81,09%.
- P10 trajectory-relative-rate: 59,35%.
- RLF-time 1,718%; HOF 50%; PP 0%; HO 0,2%; STAY 99,8%.
- Trạng thái đúng: `QOS_INFEASIBLE`.
- `test_evaluated=false`, `test=null`; không có `smdp_ppo_best_qos.pt`.

Đây chỉ là smoke mechanics, không phải kết quả nghiên cứu. 2.048 step quá ngắn
và deterministic policy gần như chọn STAY hoàn toàn; không dùng các số trên để
so sánh chất lượng PPO với 3GPP.

Hai lần smoke có manifest artifact hash khác nhau do timestamp, nhưng cùng
`dataset_content_sha256`, xác nhận fingerprint content hoạt động đúng.

## 6. Việc P0 còn chủ ý để mở

1. Exact-resume chưa triển khai. Không nối 100k lên 300k rồi gọi đó là cùng một
   run chính xác.
2. Chưa có fixed diagnostic trace ghi full observation/logits/event outcome;
   logger hiện có thống kê probability theo rollout. Thêm trace này trước khi
   cần phân tích sâu một policy cụ thể, không bắt buộc để chạy E00 từ đầu.
3. F0 lịch sử và F1 khác discount aggregation. Muốn định lượng riêng tác động
   sửa discount thì chạy matched F0/F1 100k; checkpoint cũ chỉ là historical
   reference.
4. Rolling history theo physical cell ID là thay đổi observation của P2/E04,
   không được lén đưa vào E00 vì sẽ phá nguyên tắc một thay đổi mỗi arm.

## 7. Lệnh chạy E00 screening

```powershell
.\.venv\Scripts\python.exe run.py train_ppo `
  --data-dir data/uav_100m_grid100m_dev_processed `
  --speeds 30 50 70 90 `
  --experiment-id E00_f1_k10_legacy_s17 `
  --parent-id F1_smoke_v3_s17 `
  --changed-factor screening_budget `
  --total-macro-steps 100000 `
  --rollout-steps 1024 `
  --batch-size 128 `
  --update-epochs 4 `
  --learning-rate 0.00005 `
  --evaluation-interval 10240 `
  --seed 17 `
  --split-seed 17 `
  --skip-final-test `
  --output-dir results/models/ablation_v2/E00_f1_k10_legacy_s17
```

Không benchmark test/zero-shot sau E00. Chỉ dùng validation để quyết định bước
tiếp theo trong P1.
