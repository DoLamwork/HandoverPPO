# Fresh TRAIN confirmation — luật post-TTT SINR

## Phạm vi

Một split gồm 40 TRAIN route mới được chọn outcome-blind và khóa trước khi chạy
full-route audit. Split gồm đúng 10 route cho mỗi tốc độ 30, 50, 70 và 90 km/h,
không giao với T2 main/smoke, post-TTT pilot hoặc post-TTT bank v1. Không đọc
validation, test hoặc zero-shot.

- Split seed: 17.
- Route-selection seed: 101.
- Split hash: `98e47b6f296c2f25d70573b352cad603b3b00b227d9467777615f8595b1aabe5`.
- Bootstrap: 2.000 route-cluster resamples, phân tầng speed, seed 43.
- Luật được giữ nguyên: sau A3 đủ TTT, `WAIT 20 ms` nếu target SINR thấp hơn
  serving SINR; ngược lại `HO_NOW` tới target A3.
- Không fit model và không thay threshold sau khi khóa route.

## Kết quả

Luật SINR chọn `WAIT` 264/1.797 lần quyết định, tương đương 14,69%.

| Metric | B0 / 3GPP | Luật SINR | Chênh lệch |
|---|---:|---:|---:|
| Mean relative rate | 90,2143% | 90,4168% | +0,2025 điểm % |
| Mean estimated rate | 21,6911 Mbps | 21,7312 Mbps | +0,0401 Mbps |
| p10 route rate | 11,1703 Mbps | 11,1427 Mbps | -0,0276 Mbps |
| HO attempts | 1.614 | 1.533 | -81 |
| HOF | 9,4176% | 8,4801% | -0,9375 điểm % |
| Ping-pong | 38,0301% | 35,4954% | -2,5347 điểm % |
| RLF time | 1,2667% | 1,0833% | -0,1833 điểm % |

Paired bootstrap CI95 của chênh lệch luật trừ B0:

- Relative rate: `[+0,1109; +0,3060]` điểm %.
- HOF: `[-1,5366; -0,3887]` điểm %.
- Ping-pong: `[-3,2520; -1,7746]` điểm %.
- RLF time: `[-0,2750; -0,1000]` điểm %.

Mean relative-rate retention theo tốc độ đều trên 100%:

- 30 km/h: 100,1957%.
- 50 km/h: 100,1723%.
- 70 km/h: 100,1547%.
- 90 km/h: 100,3814%.

## Gate và diễn giải

- B0/B0A parity: PASS.
- QoS retention >=98% aggregate và từng tốc độ: PASS.
- Mean QoS gain với CI dương: PASS.
- HOF non-worse: PASS.
- Ping-pong non-worse: PASS.
- RLF-time non-worse: PASS.
- Strict-all: PASS.

Kết quả mới lặp lại tín hiệu của 40 development routes trước đó và lần này HOF
cũng vượt gate. Vì split được khóa trước khi xem outcome và không giao với các bank
đã khai báo, đây là bằng chứng đủ để mở một PPO pilot ngắn. Nó vẫn là TRAIN-only
evidence, không thay thế validation/test cuối cùng.

P10 route rate giảm nhẹ 0,0276 Mbps dù mean rate tăng. Vì vậy PPO pilot phải khóa
thêm gate QoS đuôi (p10 hoặc tỷ lệ route vi phạm QoS), tránh tối ưu mean bằng cách
làm xấu một nhóm route khó.

Trường `ppo_status=CLOSED` trong artifact phản ánh audit không tự động cho phép hay
khởi động training. Sau khi review kết quả strict-all này, khuyến nghị nghiên cứu
là **READY FOR PPO PILOT**, chưa phải READY FOR 100k/300k.

## Bước kế tiếp

Chạy PPO pilot 10.240 macro-step với action `HO_NOW/WAIT_20`, target A3 giữ nguyên,
observation causal T4D.1 và cùng reward hiện tại. So sánh trên cùng TRAIN routes với
B0 và luật SINR; bắt buộc kiểm tra action collapse, RR mean, p10 rate, HOF, PP và
RLF time trước khi cân nhắc screening 100k.
