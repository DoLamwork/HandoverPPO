# Audit khác biệt T6A seed 17 và seed 29

## Phạm vi

Audit dùng đúng 80 route TRAIN đã khóa trong T2, gồm 20 route cho mỗi tốc độ
30/50/70/90 km/h. Không đọc validation, test hoặc zero-shot. Hai checkpoint
đều có cùng trigger `a3_entry_redecision_v1`, observation
`t4d1_compact_v1`, reward `trajectory_qos_recovery_reward_v1`, K=1 và
redecision 20 ms; yếu tố khác nhau là training seed.

- Seed 17: checkpoint `best_qos`, được chọn tại 100.000 macro-step.
- Seed 29: không có checkpoint QoS-feasible; checkpoint `best_return` tại
  20.480 macro-step được xuất theo chế độ best-effort.

Mỗi checkpoint được replay deterministic trên cùng route set. Hợp các state do
hai policy đi qua được lưu thành bank; sau đó cả hai actor cùng chấm xác suất
trên từng state. Đây là phép so sánh policy trên cùng input, không phải nhãn
counterfactual HO-vs-WAIT.

## State bank và deterministic argmax

- 30.629 state occurrences.
- 27.554 state duy nhất.
- 3.075 state xuất hiện trong rollout của cả hai policy.
- 19 feature/state; toàn bộ action mask đều theo checkpoint objective.

| Phép đo trên 27.554 state | Seed 17 | Seed 29 |
|---|---:|---:|
| Mean P(HO) | 0,49640 | 0,49587 |
| Độ lệch chuẩn P(HO) | 0,01107 | 0,00566 |
| Deterministic HO | 32,89% | 18,67% |
| `|P(HO)-P(STAY)| <= 0,02` | 65,50% | 87,18% |
| `|P(HO)-P(STAY)| <= 0,10` | 99,91% | 100,00% |

Mean `P(HO)_29 - P(HO)_17` chỉ là **-0,00053**, nhưng hai argmax khác nhau
trên **34,09%** state. Trong đó:

- Seed 17 chọn HO, seed 29 chọn STAY: 24,16%.
- Seed 17 chọn STAY, seed 29 chọn HO: 9,93%.
- Tương quan P(HO) giữa hai policy chỉ 0,474.

Trên riêng 3.075 state chung, disagreement vẫn là 36,46%. Vì vậy khác biệt
không thể giải thích chỉ bằng việc hai policy đi qua route state khác nhau.
Boundary actor thực sự khác, nhưng cả hai đều nằm rất sát 0,5; deterministic
argmax khuếch đại sai khác xác suất rất nhỏ thành hành vi khác lớn.

## Feedback do occupancy của policy

Hai policy tạo số decision state rất khác nhau:

- Rollout seed 17: 6.884 occurrences.
- Rollout seed 29: 23.745 occurrences.

Trên state do seed 17 tạo, seed 29 thực ra sẽ chọn HO 64,47%, cao hơn seed 17
41,07%. Nhưng trên state do seed 29 tự tạo, seed 29 chỉ chọn HO 10,01%, trong
khi seed 17 sẽ chọn HO 34,13%.

Điều này bác bỏ diễn giải đơn giản “trọng số seed 29 luôn bias STAY”. Cơ chế
quan sát được là:

1. Một số lựa chọn STAY ban đầu đưa policy sang chuỗi redecision 20 ms.
2. Chuỗi đó tạo nhiều macro-step gần nhau và một phân phối state mới.
3. Boundary của seed 29 trên vùng này tiếp tục nghiêng rất nhẹ về STAY.
4. Argmax lặp lại STAY, khiến vùng redecision tự chiếm ưu thế trong occupancy.

Theo stage, khác biệt rõ nhất nằm trong redecision:

| Stage | State | HO seed 17 | HO seed 29 | Disagreement |
|---|---:|---:|---:|---:|
| Entry | 1.249 | 84,87% | 67,89% | 34,91% |
| Maturity | 2.994 | 56,28% | 54,71% | 48,00% |
| Redecision | 23.204 | 27,06% | 11,25% | 32,23% |
| Scheduled | 107 | 35,51% | 43,93% | 40,19% |

## Full-route TRAIN outcome

| Metric | Seed 17 | Seed 29 | Delta 29−17 | Bootstrap CI 95% |
|---|---:|---:|---:|---:|
| Relative rate | 90,29% | 86,67% | -3,624 pp | [-4,416; -2,880] |
| Mean rate | 21,18 Mbps | 20,50 Mbps | -0,680 Mbps | [-0,831; -0,544] |
| HOF | 8,78% | 11,41% | +2,630 pp | [+1,778; +3,467] |
| PP | 32,80% | 18,86% | -13,943 pp | [-15,981; -12,007] |
| RLF-time | 1,03% | 1,27% | +0,240 pp | [+0,139; +0,342] |
| HO attempts | 2.826 | 2.376 | -450 | — |

Seed 29 không chỉ đổi rate lấy PP thấp hơn. Nó đồng thời làm RR, HOF và
RLF-time xấu hơn seed 17; chỉ PP được cải thiện do giảm HO mạnh.

## Tín hiệu PPO tại checkpoint được chọn

Ở update tương ứng checkpoint được chọn:

| Tín hiệu | Seed 17, update 98 | Seed 29, update 20 |
|---|---:|---:|
| Entropy | 0,69249 | 0,69304 |
| Explained variance critic | -0,00791 | -0,21740 |
| P(HO) trước update trên rollout | 0,51178 | 0,50199 |
| Mean normalized advantage STAY | +0,05520 | +0,01144 |
| Mean normalized advantage HO | -0,05653 | -0,01083 |
| Delta P(STAY) sau update | +0,00507 | +0,00018 |

Hai điểm cần giữ đúng khi diễn giải:

1. Advantage là on-policy rollout signal, không phải nhãn HO/WAIТ tối ưu cho
   từng state.
2. Seed 29 được chọn rất sớm vì không checkpoint nào qua QoS gate. So sánh
   “10 update cuối” của cả hai run không phải so sánh hai selected checkpoint;
   update 20 và 98 mới là mốc phù hợp.

Critic của seed 29 tại mốc chọn có explained variance âm sâu hơn, trong khi
actor vẫn gần entropy cực đại. PPO chưa hình thành boundary timing có margin;
checkpoint selection chỉ chọn nghiệm ít xấu nhất trong run thất bại.

## Kết luận nguyên nhân

Kết quả hỗ trợ đồng thời ba cơ chế:

1. **Boundary gần tie:** gần như mọi state có P(HO) và P(STAY) sát nhau.
2. **Argmax amplification:** thay đổi xác suất trung bình dưới 0,1 điểm % vẫn
   làm 34% action deterministic khác nhau.
3. **Redecision occupancy feedback:** WAIT 20 ms sinh thêm rất nhiều decision
   epoch; seed 29 đi vào vùng đó và tự lặp STAY, làm giảm HO quá mức.

Không có bằng chứng rằng observation hoàn toàn thiếu thông tin, hoặc reward
luôn khuyến khích STAY. Vấn đề trước mắt là policy boundary/credit yếu kết hợp
với phân phối macro-step phụ thuộc vào chính hành vi WAIT.

## Gate và bước tiếp theo

Không nên tăng thẳng lên 300k từ seed 29. Bước tiếp theo cần là một pilot T6B
thay đúng một yếu tố để phá feedback loop, giữ nguyên observation, reward và
trigger:

1. So sánh control T6A với redecision cadence lớn hơn hoặc event-driven
   redecision, nhằm giảm việc một chuỗi WAIT ngắn chiếm phần lớn rollout.
2. Giữ checkpoint gate QoS nhưng bổ sung gate policy margin và cross-seed:
   không chấp nhận policy có phần lớn state nằm sát 0,5.
3. Trước khi train dài, tạo nhãn counterfactual HO_NOW/WAIT trên một mẫu state
   disagreement để đo seed nào đúng, vì same-state probability hiện chỉ chứng
   minh khác biệt chứ chưa xác định action tối ưu.

Chỉ sau khi T6B pilot duy trì RR và giảm HOF/PP ở ít nhất ba seed mới mở lại
held-out benchmark.

## Artifact

- `results/analysis/early_ho_timing/t6a_seed17_vs_seed29_v1/main_report.json`
- `same_state_bank.npz`
- `disagreement_states.jsonl`
- `route_outcomes.jsonl`
- `training_signal_rows.jsonl`
- `run_manifest.json`
