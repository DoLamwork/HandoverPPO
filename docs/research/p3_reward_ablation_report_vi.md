# Báo cáo triển khai P3 — reward ablation E07-E10

Ngày triển khai: 03/10/2026.

Trạng thái: code, unit test, smoke train và checkpoint reload đã hoàn tất.
Screening 100k E07-E10 chưa chạy; các số smoke trong tài liệu này không phải
kết quả nghiên cứu.

## 1. Phạm vi được khóa

P3 chỉ thay reward. Các thành phần sau giữ nguyên từ E06:

- K=3, candidate order `ranked_v1`;
- action `STAY`, `HO_C1`, `HO_C2`, `HO_C3` và structural mask;
- observation `qos_full_v1`, 62 chiều;
- A3 completed-TTT là trigger quyết định;
- MLP 64-64, PPO sampler, entropy, learning rate và split seed;
- mỗi arm khởi tạo model mới, không warm-start từ parent.

Như vậy E07-E10 tạo thành chuỗi ablation reward có điều kiện trên E06. Parent
ID dùng để ghi lineage, không có nghĩa là nạp trọng số parent.

## 2. Chuẩn hóa QoS theo trajectory

Với trajectory j:

```text
D*_j  = sum_t dt * R_best,j(t)
rho_j = sum_t dt * R_j(t) / D*_j
u_j,t = 60 * dt * R_j(t) / D*_j
```

`R_best,j(t)` được tính từ SINR tốt nhất trong toàn bộ cell tại tick t bằng
cùng bandwidth 10 MHz. Nếu policy luôn nhận được `R_best`, tổng rate utility
không discount trên trajectory bằng đúng 60. Điều này làm mỗi route có tổng
ngân sách QoS tương đương, thay vì route dài hoặc có rate tuyệt đối cao chi phối
gradient.

`D*_j` dùng toàn bộ trace nên là privileged simulator reward normalizer. Nó
không được đưa vào observation, actor hoặc critic. Đây phải được công bố rõ khi
viết bài. SMDP discount vẫn được áp dụng sau khi tạo primitive reward, do đó
discounted return dùng bởi PPO không nhất thiết bằng 60 ngay cả với oracle.

## 3. Bốn reward schema

| Arm | Reward schema | Thay đổi so với parent |
|---|---|---|
| E07 | `trajectory_qos_reward_v1` | Dùng `u_j,t`; giữ RLF=5, PP=1 như legacy |
| E08 | `trajectory_qos_events_reward_v1` | HOF=0,2; pure RLF=0,2; PP=0,1; HO=0 |
| E09 | `trajectory_qos_recovery_reward_v1` | E08 + recovery cost `1 x dt` mỗi recovery tick |
| E10 | `trajectory_qos_dual_reward_v1` | E09 + QoS dual coefficient riêng theo speed |

HOF và pure RLF không chồng lặp: nếu một tick vừa báo `handover_failed` vừa
`rlf_started`, chỉ HOF cost được cộng. Recovery-time là duration cost riêng,
nên vẫn được tính trên các tick `RLF_RECOVERY`. Không có target-ID penalty,
non-C1 bonus hoặc HO-count penalty trong P3.

Reward decomposition giờ luôn xuất riêng:

```text
rate + handover + handover_failure + rlf + ping_pong + rlf_recovery
```

Các field mới mặc định bằng 0 khi đọc benchmark artifact legacy.

## 4. E10 QoS dual theo tốc độ

Tại cuối PPO update, completed training episodes của rollout được nhóm theo
`speed_kmh`. Với speed s có ít nhất 4 episode hoàn tất:

```text
lambda_s <- clip(lambda_s + 0,5 * (0,98 - mean(rho_s)), 0, 10)
rate reward <- (1 + lambda_s) * u_j,t
```

Các nguyên tắc đã được code khóa:

- lambda được snapshot ở đầu rollout và không đổi trong collection/update;
- chỉ completed episode từ train được dùng, không dùng validation/test;
- speed chưa đủ 4 sample được log nhưng không cập nhật;
- checkpoint lưu `dual_lambdas_by_speed` và evaluate/benchmark khôi phục nó;
- `training_metrics.jsonl` ghi target, sample count, mean rho, shortfall,
  lambda trước/sau và speed thật sự được update;
- log đánh dấu `infeasible_speeds` nếu lambda đã chạm trần mà vẫn thiếu QoS;
- console in thêm dòng `[dual]` sau mỗi PPO update.

E10 là adaptive extension. E09 vẫn là fixed-reward arm chính để so sánh công
bằng với E06.

## 5. Tích hợp train/evaluate/benchmark

CLI `train_ppo` có thêm:

```text
--reward-schema
--dual-target-ratio
--dual-learning-rate
--dual-max-coefficient
--dual-min-episodes-per-speed
```

`experiment_config.json` lưu reward schema, toàn bộ weight, loại normalizer,
khẳng định normalizer không phải actor/critic input và dual update contract.
Objective checkpoint chứa reward schema/config để ngăn nạp nhầm reward. Các
checkpoint selected/latest/Pareto lưu dynamic reward state. `evaluate_ppo` và
`benchmark` tự dựng đúng reward model từ checkpoint; không cần truyền lại CLI
reward flags.

Validation và benchmark xuất thêm mean HOF penalty, HO penalty và recovery
penalty. Các component cũ vẫn giữ nguyên tên nên artifact P0-P2 còn đọc được.

## 6. Kiểm thử đã chạy

- 163 unit/integration tests pass sau P3.
- Test oracle xác nhận tổng utility E07 bằng 60.
- Test HOF/pure-RLF xác nhận không double count.
- Test recovery xác nhận 10 ms tạo cost -0,01.
- Test dual xác nhận threshold sample, rollout freeze và checkpoint round-trip.
- Test checkpoint contract xác nhận reward schema/config là một phần objective.
- Smoke E10: 64 macro-step, một update, validation 11 trajectory, không test.
- Checkpoint smoke được nạp lại thành công bằng `evaluate_ppo`; metric khớp
  validation đã lưu.

Smoke E10 có validation return 35,224, rate 20,460 Mbps và QoS gate pass, nhưng
model gần như chưa học vì chỉ có một update. Một completed training episode ở
50 km/h tạo shortfall 0,1332; lambda giữ 0 vì chưa đủ ngưỡng 4 sample. Đây là
kiểm thử cơ chế, không được dùng để kết luận E10 tốt hơn E06/3GPP.

Artifacts smoke cục bộ:

```text
results/models/p3_smoke/E10_dual_smoke_s17/
```

## 7. Lệnh screening 100k

Cấu hình chung cho cả bốn arm:

```powershell
.\.venv\Scripts\python.exe run.py train_ppo `
  --data-dir data/uav_100m_grid100m_dev_processed `
  --speeds 30 50 70 90 `
  --candidate-count 3 `
  --candidate-order ranked_v1 `
  --observation-schema qos_full_v1 `
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
# E07
--reward-schema trajectory_qos_reward_v1 `
--experiment-id E07_qos_rate_k3_s17 `
--parent-id E06_qos_full_k3_s17 `
--changed-factor trajectory_qos_rate `
--output-dir results/models/ablation_v2/E07_qos_rate_k3_s17

# E08
--reward-schema trajectory_qos_events_reward_v1 `
--experiment-id E08_qos_events_k3_s17 `
--parent-id E07_qos_rate_k3_s17 `
--changed-factor event_cost_rescale `
--output-dir results/models/ablation_v2/E08_qos_events_k3_s17

# E09
--reward-schema trajectory_qos_recovery_reward_v1 `
--experiment-id E09_qos_recovery_k3_s17 `
--parent-id E08_qos_events_k3_s17 `
--changed-factor rlf_recovery_duration_cost `
--output-dir results/models/ablation_v2/E09_qos_recovery_k3_s17

# E10
--reward-schema trajectory_qos_dual_reward_v1 `
--experiment-id E10_qos_dual_k3_s17 `
--parent-id E09_qos_recovery_k3_s17 `
--changed-factor adaptive_qos_dual `
--output-dir results/models/ablation_v2/E10_qos_dual_k3_s17
```

Chạy tuần tự E07→E08→E09→E10. Screening chỉ dùng validation và
`skip-final-test`. Sau khi có kết quả, so sánh selected checkpoint và last
scheduled validation, reward component totals, action distribution, lambda
trajectory theo speed và QoS aggregate/per-speed. Không promote E10 chỉ vì
lambda làm return lớn hơn; tiêu chí vẫn là relative rate, RLF-time, HOF và PP.

## 8. Điều chưa được P3 giải quyết

P3 chưa chứng minh policy đã thoát khỏi C1/STAY bias; đó là câu hỏi thực nghiệm
của screening. Reward-normalized return giữa các schema không thể so trực tiếp
về trị số vì scale khác. Kết luận phải dựa vào physical metrics và paired
validation, không dựa vào mean return. Nếu E07 cải thiện chỉ do tăng scale chứ
không do trajectory normalization, cần chạy arm phụ `R_scale` đã nêu trong kế
hoạch. Nếu E10 chạm lambda=10 mà QoS vẫn fail, đánh dấu constraint infeasible
thay vì tiếp tục tăng lambda hoặc tune theo test.
