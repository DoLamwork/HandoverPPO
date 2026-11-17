# Audit chất lượng dữ liệu P0 Canonical pilot

## Phạm vi

Audit dùng toàn bộ 120 trajectory của `data/uav_p0_canonical_v1_pilot`, gồm
20 trajectory ở mỗi tốc độ 30/50/70/90/120/150 km/h. Mỗi trajectory có 6.000
mẫu, `dt=10 ms`, tương đương 60 giây. Không train PPO và không đọc dataset lịch
sử, validation hay test của các thí nghiệm trước.

Nguồn radio được khóa ở 2,1 GHz, bandwidth 10 MHz; topology 7 site/21 sector;
UAV cao 100 m; grid đường 100 m; urban profile có area ratio 0,30, mật độ 500
công trình/km² và Rayleigh height scale 15 m.

## Kết quả toàn cục

| Chỉ số | Kết quả |
|---|---:|
| Trajectory / samples | 120 / 720.000 |
| Best-cell RSRP p5 / p50 / p95 | -98,11 / -86,68 / -70,46 dBm |
| Best-cell SINR p5 / p50 / p95 | -2,20 / 4,30 / 21,43 dB |
| Top-1 minus top-2 RSRP <= 2 dB | 19,03% |
| Top-1 minus top-2 RSRP <= 5 dB | 42,45% |
| Cell-rank switches/phút | 63,67 |
| Cell từng đứng đầu | 20/21 |
| Normalized winner entropy | 0,955 |
| 3GPP mean relative rate | 89,19% |
| 3GPP HOF | 13,07% |
| 3GPP ping-pong | 36,86% |
| 3GPP RLF-time | 1,81% |
| TTT start / abort / maturity | 5.781 / 636 / 5.006 |

Phân phối best-cell có dynamic range rộng nhưng không mất coverage hàng loạt.
Winner entropy cao và cell thắng lớn nhất chỉ chiếm 10,16% tổng thời gian, nên
dataset không bị một sector duy nhất thống trị. Khoảng 19% timestep có hai cell
cách nhau không quá 2 dB, tạo đủ trạng thái HO khó; đồng thời median gap 6,16 dB
cho thấy không phải mọi timestep đều mơ hồ.

Giá trị rất thấp trong phân phối **all-cell** (không phải best-cell), như RSRP
p1 khoảng -184,77 dBm và SINR p1 khoảng -106,3 dB, chủ yếu đến từ sector ngoài
hướng búp sóng/xa UAV. Chúng không được dùng để kết luận mất phủ vì best-cell
distribution vẫn nằm trong miền hữu dụng.

## Kết quả theo tốc độ

| km/h | RR % | p10 RR % | HOF % | PP % | RLF-time % | HO | rank switch/phút |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | 93,11 | 85,46 | 10,91 | 36,15 | 1,02 | 559 | 48,05 |
| 50 | 93,06 | 87,51 | 10,61 | 33,27 | 1,00 | 575 | 47,70 |
| 70 | 90,11 | 83,94 | 11,20 | 36,38 | 1,45 | 777 | 56,45 |
| 90 | 88,11 | 82,32 | 10,18 | 41,07 | 1,50 | 894 | 64,05 |
| 120 | 86,14 | 81,18 | 14,74 | 33,63 | 2,62 | 1.065 | 79,90 |
| 150 | 84,63 | 77,88 | 17,39 | 39,08 | 3,28 | 1.133 | 85,85 |

Tốc độ cao tạo stress đúng hướng: rank switch, số HO, HOF và RLF-time tăng,
trong khi relative rate giảm. Tuy vậy, 150 km/h vẫn giữ RR trung bình 84,63%
và RLF-time 3,28%; P0 do đó khó hơn ở high-speed nhưng chưa rơi vào trạng thái
mọi policy đều thất bại.

## Gate

- `G_integrity`: PASS — đúng 120 route, 20 route/tốc độ.
- `G_radio_dynamic_range`: PASS — best-cell RSRP và SINR đều có p99-p1 lớn
  hơn 10 dB.
- `G_decision_opportunity`: PASS — mọi tốc độ đều có rank switch, ít nhất ba
  winner cell, TTT start và A3 maturity.
- `G_protocol_viability`: PASS — ở từng tốc độ RR >=70%, HOF <=30% và
  RLF-time <=10%.

Kết luận: **P0 pilot đạt yêu cầu làm canonical baseline**. Nó có đủ biến động
và đủ cơ hội timing/HO, đồng thời baseline 3GPP vẫn hoạt động được. Pilot chỉ có
20 route/tốc độ nên phù hợp correctness, calibration và train ngắn; chưa đủ để
thay bộ main cho kết luận PPO cuối cùng.

## Khuyến nghị

1. Giữ nguyên toàn bộ tham số P0; không tinh chỉnh dựa trên PPO.
2. Sinh P0 main bằng route ID mới và đủ route cho train/validation/test.
3. Khóa route split trước khi train; 120/150 km/h nên giữ làm OOD nếu mục tiêu
   train chính vẫn là 30/50/70/90 km/h.
4. Chỉ sau khi P0 main được khóa mới tạo P1 dense-urban, thay đúng urban profile
   và giữ topology, mobility seed, radio bandwidth/carrier giống P0.

