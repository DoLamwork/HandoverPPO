# T5 timing-only PPO pilot — 10.240 macro-step

## Cấu hình khóa

- Trigger: completed A3+TTT.
- Action: `STAY = WAIT_20` và `HO_C1 = HO_NOW`; target luôn là target A3.
- Observation: `t4d1_compact_v1`, 19 feature causal, khớp feature set T4D.1.
- Reward: `trajectory_qos_recovery_reward_v1` (E09 cố định).
- PPO: MLP dùng chung 64-64, learning rate 5e-5, rollout 1.024, batch 128,
  4 epoch/update, 10 update, seed/split-seed 17.
- Tốc độ: 30/50/70/90 km/h.
- Test/zero-shot: không chạy.

## Kết quả

Training hoàn tất đủ 10.240 macro-step. Checkpoint được chọn theo best-return
best-effort vì không checkpoint nào đạt QoS gate.

| Metric validation | PPO | Tỷ lệ so với 3GPP |
|---|---:|---:|
| Mean rate | 21,9673 Mbps | 95,41% |
| Mean relative rate | 86,2701% | 94,52% |
| HOF | 7,7778% | — |
| Ping-pong | 32,0209% | — |
| RLF time | 0,6647% | — |
| STAY / WAIT_20 | 95,33% | — |
| HO_NOW | 4,67% | — |

QoS theo từng tốc độ đều fail; relative-rate retention lần lượt là 95,63%,
94,49%, 93,76% và 92,21% tại 30/50/70/90 km/h. P10 relative rate còn
75,42%. Trạng thái pilot là `QOS_INFEASIBLE`; không được promote lên 100k.

## Vì sao rollout gần 50/50 nhưng validation lại gần như luôn WAIT?

Ở rollout cuối, xác suất trung bình vẫn gần như bằng nhau:

- `P(WAIT_20) = 0,50198`;
- `P(HO_NOW) = 0,49802`;
- entropy chuẩn hóa = 0,99998.

Training lấy mẫu stochastic nên action count vẫn gần 50/50. Validation dùng
argmax deterministic; chỉ cần logit WAIT cao hơn rất ít là WAIT thắng trên phần
lớn state. Vì WAIT tạo decision mới sau 20 ms, sai lệch nhỏ này còn bị lặp nhiều
lần trên cùng A3 episode, dẫn tới 55.319 WAIT và chỉ 2.710 HO_NOW.

## Diagnostic học

- KL gần 0 và clip fraction 0 trong cả pilot: policy thay đổi rất ít.
- Actor gradient norm ở update cuối: khoảng 0,0713.
- Critic gradient norm: khoảng 22,111.
- Global clipping factor: khoảng 0,0226.
- Explained variance cuối: -0,0305.

Critic loss lớn chi phối shared encoder và kích hoạt global gradient clipping;
actor vốn đã yếu bị scale xuống thêm khoảng 44 lần. Do đó chạy dài hơn cùng cấu
hình có nguy cơ chỉ củng cố một sai lệch argmax rất nhỏ, không học ranh giới
SINR có ích đã được audit vật lý xác nhận.

## Quyết định

Không mở 100k từ checkpoint này. Bước kế tiếp nên là pilot đối chứng cùng ngân
sách với critic được ổn định trước: `fixed_scale_v1` hoặc actor/critic tách riêng,
giữ nguyên action, observation, reward, seed và dữ liệu. Gate phải bao gồm mean
RR, p10 RR, HOF, PP, RLF-time và tỷ lệ WAIT; checkpoint chỉ được promote nếu QoS
>=98% aggregate và từng tốc độ.
