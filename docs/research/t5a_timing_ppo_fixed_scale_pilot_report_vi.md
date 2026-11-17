# T5A — PPO timing pilot với critic `fixed_scale_v1`

## Mục tiêu

T5A là đối chứng một-yếu-tố với T5. Mọi thành phần được giữ nguyên, ngoại trừ
critic target được chia cho một scale cố định fit từ TRAIN. Mục tiêu là kiểm tra
giả thuyết gradient critic quá lớn đã làm gradient clipping chung triệt tiêu cập
nhật actor trong T5.

## Cấu hình khóa

- Dataset: `data/uav_100m_grid100m_dev_processed`.
- Split: seed 17; 700 TRAIN, 150 validation, 150 test.
- Action: `STAY`, `HO_C1`; K=1, `ranked_v1`.
- Observation: `t4d1_compact_v1`, 19 feature.
- Reward: `trajectory_qos_recovery_reward_v1`.
- STAY: `a3_redecision_v1`, 20 ms.
- Network: `shared_mlp_v1`.
- PPO: 10.240 macro-step, rollout 1.024, batch 128, 4 epoch, LR 5e-5.
- Thay đổi duy nhất: `value_normalization=fixed_scale_v1`.
- Scale: 24,569382831; fit từ 80 TRAIN routes (20 route/tốc độ), không đọc
  validation/test để fit scale.
- Test/zero-shot: không chạy.

Runner: `scripts/run_t5a_timing_ppo_fixed_scale_pilot.ps1`.

## Kết quả validation deterministic

| Metric | 3GPP Always-HO | T5 control | T5A fixed-scale |
|---|---:|---:|---:|
| Rate (Mbps) | 23,0245 | 21,9673 | 23,0355 |
| Rate / 3GPP | 100% | 95,409% | 100,048% |
| Relative rate (%) | 91,2718 | 86,2701 | 91,3440 |
| Relative-rate / 3GPP | 100% | 94,520% | 100,079% |
| HOF (%) | 10,9031 | 7,7778 | 10,2440 |
| Ping-pong (%) | 37,0569 | 32,0209 | 33,3333 |
| RLF-time (%) | 1,3130 | 0,6647 | 1,1260 |
| STAY (%) | 0 | 95,3299 | 23,7986 |
| HO (%) | 100 | 4,6701 | 76,2014 |
| P10 relative rate (%) | — | 75,4163 | 84,8660 |

T5A vượt gate relative-rate 98% ở aggregate và từng tốc độ. Relative-rate ratio
theo tốc độ lần lượt là 100,035%, 100,112%, 100,057% và 100,179% cho
30/50/70/90 km/h. Đồng thời HOF, ping-pong và RLF-time đều thấp hơn baseline
3GPP trên cùng 150 validation routes.

## Chẩn đoán tối ưu

| Diagnostic | T5 update 1 | T5A update 1 | T5 update 10 | T5A update 10 |
|---|---:|---:|---:|---:|
| Actor gradient norm | 0,1843 | 0,1086 | 0,0713 | 0,3077 |
| Critic gradient norm | 22,8905 | 0,8509 | 22,1110 | 0,1860 |
| Global clipping factor | 0,02184 | 0,58289 | 0,02261 | 1,0 |

Chuẩn hóa critic đã xử lý đúng lỗi mất cân bằng gradient: đến update 10 không
còn clipping toàn cục và actor gradient lớn hơn critic gradient. Đây là bằng
chứng ủng hộ giả thuyết kỹ thuật của T5.

Tuy nhiên, policy chưa hội tụ mạnh sau 10 update:

- entropy update 10 vẫn là 0,69307, gần cực đại của hai action;
- approximate KL chỉ khoảng 2,43e-6;
- rollout cuối có `P(HO)` trung bình 0,50274, P10/P50/P90 lần lượt
  0,49685/0,50227/0,51030;
- deterministic `argmax` có thể khuếch đại margin xác suất nhỏ.

## Kết luận và gate

T5A **qua gate QoS pilot** và tốt hơn rõ rệt T5 control. `fixed_scale_v1` nên
được giữ cho nhánh timing tiếp theo. Kết quả chưa đủ để tuyên bố PPO đã học một
quy luật timing ổn định vì action margin còn rất nhỏ.

Bước tiếp theo nên là một run xác nhận trung gian hoặc 100k có checkpoint/validation
dày hơn, đồng thời theo dõi deterministic và stochastic policy. Không cần mở
`separate_mlp_v1` trước khi xác nhận liệu T5A có tiếp tục tách xác suất theo
state khi được thêm budget hay không.

