# Early-HO T1 — pilot B1 giảm ngưỡng A3

Ngày: 10/10/2026. Nhánh: `codex/early-ho-timing`.
Artifact chính: `results/analysis/early_ho_timing/b1_lower_threshold_pilot_v2`.
Trạng thái: **PILOT COMPLETE; chưa đánh giá gate nghiên cứu**.

## Contract

B1 kiểm tra liệu chỉ giảm ngưỡng A3 có đủ tạo lợi ích hay không:

| Tham số | B0 | B1 |
|---|---:|---:|
| A3 offset | 1 dB | 0 dB |
| A3 hysteresis | 1 dB | 1 dB |
| TTT | 40 ms | 40 ms |
| Entering delta khi offset khác bằng 0 | >2 dB | >1 dB |
| Leaving delta của B1 | — | <-1 dB |

Mọi tham số radio, L3, RLM, HO preparation/execution, RLF recovery và PP giữ
nguyên. B1 vẫn là A3 rule cố định, không có agent và không dùng future.

## Dữ liệu

- Đúng 8 TRAIN route đã dùng để xác nhận B0/B0A: 2 route/tốc độ
  30/50/70/90 km/h, tổng 48.000 primitive tick.
- Route/file hash được kế thừa và kiểm tra lại từ manifest B0/B0A.
- Không đọc validation/test/zero-shot và không train model.
- Đây là pilot cơ chế, không bootstrap CI và không dùng để tuyên bố thành công.

## Kết quả aggregate

| Metric | B0 | B1 | B1 − B0 |
|---|---:|---:|---:|
| Mean route RR | 90,4968% | 90,5038% | +0,0070 điểm % |
| Mean rate | 23,7753 Mbps | 23,7762 Mbps | +0,00094 Mbps |
| HO attempts | 328 | 329 | +1 |
| HOF | 10,6707% | 10,0304% | -0,6403 điểm % |
| Ping-pong | 43,6860% | 43,9189% | +0,2329 điểm % |
| RLF-time | 1,4583% | 1,3750% | -0,0833 điểm % |

Theo tốc độ, delta RR lần lượt là:

- 30 km/h: +0,0101 điểm %; event metrics không đổi.
- 50 km/h: +0,0207 điểm %; event metrics không đổi.
- 70 km/h: -0,0965 điểm %; HOF/RLF giảm nhưng PP tăng.
- 90 km/h: +0,0936 điểm %; event metrics không đổi.

## Cơ chế theo route

Sáu trong tám route không đổi số HO/HOF/PP/RLF. Hai route 70 km/h tạo phần
lớn khác biệt:

- `70:124`: thêm một TTT và một HO attempt, RR giảm khoảng 0,35 điểm %.
- `70:142`: thêm một TTT; giảm hai HOF và 40 recovery tick nhưng tăng hai PP,
  RR tăng khoảng 0,16 điểm %.

Vì trajectory có thể phân kỳ sau lần HO khác thời điểm, không diễn giải việc
thêm/bớt event như cùng một HO được dịch sớm hơn nếu chưa replay chuỗi chi tiết.

## Kết luận

Giảm A3 offset 1→0 dB tạo thay đổi vật lý đúng như mong đợi, nhưng pilot chưa
cho thấy cải thiện QoS rõ ràng: RR aggregate gần như không đổi. Reliability có
trade-off: HOF và RLF-time giảm nhẹ, trong khi PP tăng nhẹ. Point estimate cũng
chưa đạt gate đã đăng ký: RR +0,5 điểm %, HOF và PP cùng giảm ít nhất 10% tương
đối, RLF không xấu.

Do mẫu chỉ có hai route mỗi tốc độ, không kết luận B1 tốt/xấu tổng quát. B1 cần
được giữ làm comparator rule đơn giản trong pilot B2/B3/Rpost/Rearly. Không nên
train PPO chỉ dựa trên kết quả này.

## Tái lập

Output `v1` là lần chạy collector lỗi trước route đầu do đọc sai trường TTT;
không chứa kết quả hợp lệ và được giữ lại, không ghi đè. `v2` là artifact chuẩn.

```powershell
.\scripts\run_early_ho_b1_pilot.ps1 `
  -OutputDir results/analysis/early_ho_timing/b1_lower_threshold_pilot_v3
```
