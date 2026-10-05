# D61 — A3-gated redecision 100 ms

## Phạm vi thay đổi

D61 chỉ thay semantics của action STAY. Chế độ cũ
`reject_event_v1` vẫn là mặc định và khóa completed-A3 event cho tới khi A3
rời hoặc target đổi. Chế độ mới `a3_redecision_v1` mở lại cùng event sau 100 ms
nếu serving cell, target và điều kiện A3 vẫn hợp lệ.

Candidate scorer, K=3, ranking, observation, reward E07 và PPO
hyperparameter giữ nguyên như D60.

## Hợp đồng protocol

- Deadline được tính từ tic của decision đã bị STAY.
- 100 ms tương ứng đúng 10 primitive tick với `dt=10 ms`.
- A3 rời trước deadline: hủy redecision.
- Neighbor target đổi: hủy deadline cũ, reset A3 và target mới phải hoàn thành
  TTT như bình thường.
- Cùng event còn hợp lệ tại deadline: phát đúng một decision mới, không chạy
  lại TTT.
- Baseline 3GPP Always-HO không bị thay đổi.

`stay_mode` và `stay_redecision_ms` nằm trong objective/checkpoint contract.
Evaluate/benchmark tự khôi phục hai giá trị từ checkpoint; checkpoint cũ thiếu
field được hiểu là `reject_event_v1`, 100 ms.

## Kiểm thử

- Re-offer đúng tic và đúng thời lượng macro-transition 100 ms.
- Hủy khi A3 leaving.
- Target đổi chạy TTT mới và không tạo duplicate decision.
- Delay phải dương và chia hết cho `delta_t_ms`.
- Checkpoint objective từ hai STAY mode khác nhau bị từ chối.
- Counterfactual transition audit dùng đúng redecision delay.

Full suite sau triển khai: **189 test pass**.

## Pilot được phép chạy

Pilot duy nhất là `D61_a3_redecision_pilot_b10240_s17`, 10.240 macro-step,
không test/zero-shot. Runner tái lập:

```powershell
.\scripts\run_d61_redecision_pilot.ps1
```

Không promote lên 100k trước khi pilot đạt: deterministic non-STAY > 0,
useful-alternative argmax recall > 0, relative rate cải thiện rõ so với D60 và
RLF/HOF không suy giảm mất kiểm soát.

## Kết quả pilot

Pilot hoàn tất đủ 10.240 macro-step, seed/split seed 17 và không đọc test hay
zero-shot.

| Model | Return | Rate Mbps | Relative rate | RLF-time | HOF | PP | STAY |
|---|---:|---:|---:|---:|---:|---:|---:|
| D30 shared | 10,820 | 19,214 | 73,99% | 1,725% | 0% | 0% | 100% |
| D60 scorer | 10,820 | 19,214 | 73,99% | 1,725% | 0% | 0% | 100% |
| D61 redecision | 31,614 | 22,440 | 88,25% | 0,387% | 4,069% | 13,237% | 84,59% |
| 3GPP Always-C1 | -18,978 | 23,024 | 91,27% | 1,313% | 10,903% | 37,057% | 0% |

D61 tăng rate 3,225 Mbps và relative rate 14,26 điểm phần trăm so với D60,
đồng thời giảm RLF-time mạnh. So với 3GPP, nó giảm HOF khoảng 6,83 điểm phần
trăm và PP khoảng 23,82 điểm phần trăm. Đổi lại, QoS còn thiếu nhẹ: rate ratio
97,46% và relative-rate ratio 96,69%, nên checkpoint vẫn `QOS_INFEASIBLE` với
gate 98%.

Theo tốc độ, relative rate D61 là 89,67 / 88,38 / 86,86 / 85,64% ở
30 / 50 / 70 / 90 km/h. Mức hụt tăng theo vận tốc và không thể che bằng
aggregate.

## Candidate-learning audit sau pilot

Fixed TRAIN bank 3.117 state:

| Model | P(STAY) | P(C1) | P(C2) | P(C3) | Argmax |
|---|---:|---:|---:|---:|---|
| D30 | 37,66% | 32,37% | 15,75% | 14,22% | STAY 3.117 |
| D60 | 33,50% | 24,16% | 21,51% | 20,84% | STAY 3.117 |
| D61 | 27,07% | 26,43% | 23,60% | 22,90% | STAY 2.802; C1 315 |

D61 đã vượt `STAY 100%`, nhưng 2.091 action HO trên validation đều là C1;
C2/C3 vẫn bằng 0. Vì vậy useful-alternative argmax recall vẫn 0%.

Same-transition audit đúng semantics redecision tìm 56 positive state trong
rollout D61. C2 hữu ích được sample 12 lần nhưng normalized advantage trung bình
-0,226; C3 hữu ích chỉ được sample 2 lần, advantage +0,455. Sau update, xác
suất trên useful state vẫn giảm lần lượt -0,00168 và -0,00388. Redecision giải
quyết tần suất quyết định và timing, chưa giải quyết credit assignment giữa các
candidate.

## Quyết định gate sau D61

D61 **qua gate timing/reliability** nhưng **không qua gate QoS/candidate**. Không
promote trực tiếp cấu hình này lên 100k.

Nhánh tiếp theo có bằng chứng mạnh nhất là D62 actor phân cấp nhưng vẫn là PPO:

- head thứ nhất học `STAY` so với tổng xác suất `HANDOVER`;
- head thứ hai học `C1/C2/C3 | HANDOVER` bằng candidate scorer chia sẻ;
- transition STAY không backprop gradient vào candidate-choice head;
- joint probability vẫn là một phân phối 4 action để PPO ratio/entropy và API
  environment không đổi.

Factor này trực tiếp tách timing credit khỏi target-selection credit. Nếu pilot
D62 vẫn chỉ chọn C1, khi đó mới cân nhắc auxiliary TRAIN-only ranking hoặc D5
QoS constraint; không nên tăng entropy vì D61 đã giữ normalized entropy 0,998.

## Artefact

- Checkpoint/report: `results/models/d6_pilots/D61_a3_redecision_pilot_b10240_s17/`
- Fixed-bank probe: `results/analysis/p23_diagnostics/d61_pilot_v1/fixed_bank_probe/`
- Transition audit: `results/analysis/p23_diagnostics/d61_pilot_v1/transition_signal/`
- Runner: `scripts/run_d61_redecision_pilot.ps1`
