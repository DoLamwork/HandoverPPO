# P0: biên độ RSRP của cùng một sector dọc route

## Câu hỏi và cách tính

Lần phân tích trước lấy `max − min` sau khi gộp 21 sector của một route. Phép đó trộn chênh lệch giữa các BS với thay đổi mà một đường truyền thực sự trải qua. Phân tích này cố định **một route và một sector**, rồi mới tính:

\[
\Delta_{r,s}(x)=\max_{t=0}^{5999}x(r,s,t)-\min_{t=0}^{5999}x(r,s,t).
\]

Áp dụng riêng cho Gtx, FSPL, suy hao dư LoS/NLoS, shadow fading, fast fading và RSRP. Mỗi route dài 6.000 bước × 10 ms = 60 s. Chọn route 0 và 10 ở từng tốc độ 30, 50, 70, 90, 120, 150 km/h: tổng cộng **12 route × 21 sector = 252 vệt sector**. Trung vị trong các bảng là trung vị của 42 vệt sector ở cùng tốc độ; cực đại là giá trị lớn nhất trong 42 vệt đó. Đây là cực đại **trong mẫu 12 route**, chưa phải cực đại của toàn bộ 120 route P0.

Trong generator, có thể viết gọn

\[
RSRP=\underbrace{PTX+GRX-10\log_{10}(51\times12)+\text{offset}}_{\text{cố định theo sector/route}}+Gtx-FSPL-L_{\mathrm{LoS/NLoS}}-X_{\mathrm{shadow}}+F_{\mathrm{fast}}.
\]

Vì phần trong ngoặc cố định theo thời gian, nó không đóng góp vào `max − min` của cùng một sector trên một route. Các cực trị của những thành phần còn lại có thể rơi vào những bước khác nhau.

RSRP phát lại từ generator khớp toàn bộ 12 CSV sau khi làm tròn 0,01 dB, sai số lớn nhất **0 dB**. Với từng vệt sector, biên độ FSPL cũng khớp \(20\log_{10}(d_{\max}/d_{\min})\); sai số số học lớn nhất \(1,82\times10^{-14}\) dB.

Mã nguồn hiện tại có hash khác hash ghi trong manifest lúc tạo dataset vì đã có thay đổi sau đó. Kiểm tra khớp CSV ở trên xác nhận phép phát lại RSRP cho đúng 12 route đang dùng ở đây; không suy rộng kiểm tra đó sang các route chưa phát lại.

## Quãng đường, hình học và RSRP

Mỗi ô biên độ ghi **trung vị / cực đại** (dB), trừ cột khoảng cách 3D có đơn vị mét. Quãng đường được cộng từ các vị trí ghi mỗi 10 ms nên có thể thấp hơn quãng đường điều khiển vài mét khi UAV rẽ bên trong một bước.

| Tốc độ (km/h) | Quãng đường trong 60 s (m) | Biên độ khoảng cách 3D (m) | Biên độ Gtx (dB) | Biên độ FSPL (dB) | Biên độ RSRP (dB) |
|---:|---:|---:|---:|---:|---:|
| 30 | 500,0 | 153,9 / 222,3 | 25,84 / 109,02 | 1,70 / 4,76 | 70,70 / 124,83 |
| 50 | 833,3 | 196,8 / 345,3 | 11,92 / 102,14 | 1,77 / 3,89 | 65,91 / 115,51 |
| 70 | 1.166,3 | 303,3 / 393,7 | 80,48 / 115,31 | 2,93 / 11,97 | 100,84 / 138,22 |
| 90 | 1.500,0 | 336,4 / 416,9 | 29,68 / 115,48 | 2,50 / 7,94 | 75,33 / 125,76 |
| 120 | 2.000,0 | 262,5 / 357,4 | 20,22 / 98,56 | 2,37 / 11,78 | 71,82 / 123,43 |
| 150 | 2.499,1 | 428,0 / 775,5 | 37,99 / 97,71 | 4,00 / 7,75 | 83,54 / 138,09 |

**FSPL tăng theo tỉ số khoảng cách, không theo tổng quãng đường bay.** Ví dụ, tại 70 km/h, route 10, sector 12, UAV cách BS gần nhất 81,75 m và xa nhất 324,38 m; do đó FSPL biến thiên \(20\log_{10}(324,38/81,75)=11,97\) dB. Tốc độ 150 km/h đi xa hơn nhưng không nhất thiết có FSPL cực đại lớn hơn: một route có thể di chuyển ngang quanh BS hoặc ở xa BS suốt route.

**Gtx phụ thuộc vào góc nhìn từ sector tới UAV.** UAV đi trên lưới Manhattan, đổi hướng ở giao điểm. Cùng một sector có thể lần lượt nhìn UAV qua búp sóng chính, búp phụ và điểm triệt của mảng anten 3D. Mô hình tính bình phương tổng trường phức của mảng 8 × 4 phần tử; điểm triệt có thể làm gain rất thấp. Vì vậy Gtx của một sector có thể biến thiên 115,48 dB trong mẫu, dù suy hao do khoảng cách của chính sector đó thường chỉ vài dB. Mức lớn này phản ánh cực trị của **mô hình anten số**, không nên xem là biên độ điển hình của mọi sector.

## Các thành phần kênh khác

Mỗi ô là **trung vị / cực đại** của biên độ trên 42 vệt sector cùng tốc độ, đơn vị dB.

| Tốc độ (km/h) | LoS/NLoS | Shadow fading | Fast fading |
|---:|---:|---:|---:|
| 30 | 19,00 / 19,00 | 44,21 / 53,96 | 20,15 / 26,55 |
| 50 | 19,00 / 19,00 | 50,15 / 59,51 | 17,48 / 22,07 |
| 70 | 19,00 / 19,00 | 52,14 / 60,00 | 14,45 / 20,29 |
| 90 | 19,00 / 19,00 | 54,26 / 60,00 | 12,76 / 16,87 |
| 120 | 19,00 / 19,00 | 53,07 / 60,00 | 10,10 / 16,01 |
| 150 | 19,00 / 19,00 | 55,76 / 60,00 | 9,59 / 14,50 |

- **LoS/NLoS:** generator cộng 1 dB suy hao dư khi LoS và 20 dB khi NLoS. Vì thế một sector đã trải qua cả hai trạng thái có biên độ đúng 19 dB. Chuyển trạng thái tạo bước nhảy tức thời, không phải thay đổi trơn theo khoảng cách.
- Trên 252 vệt sector có tổng cộng **32.853 lần chuyển LoS/NLoS** trong 60 s mỗi vệt, trung bình khoảng 130 lần/vệt. Điều này giải thích vì sao thành phần 19 dB có thể gây bước nhảy RSRP lặp lại, dù biên độ trọn route của nó vẫn luôn là 19 dB khi cả hai trạng thái xuất hiện.
- **Shadow fading:** biến ngẫu nhiên có tương quan không gian và được chặn ở ±3σ. Với NLoS, σ = 10 dB nên biên độ lý thuyết tối đa trong cùng nhánh là 60 dB; nhiều route trong mẫu chạm cả hai biên. Khi đọc cột cực đại, cần nhớ đây là hai cực trị trong 6.000 mẫu, không phải thay đổi ở một bước.
- **Fast fading:** biên độ quan sát trên route giảm từ 20,15 xuống 9,59 dB theo trung vị khi tốc độ tăng từ 30 lên 150 km/h. Generator lấy trung bình công suất theo các mẫu nội bộ 1 ms trong mỗi bước 10 ms; Doppler và route/seed đều ảnh hưởng. Chỉ có hai route mỗi tốc độ nên không thể quy toàn bộ xu hướng này riêng cho tốc độ.

## Tổng hợp trên 252 vệt sector

| Thành phần | Trung vị biên độ (dB) | Phân vị 95% (dB) | Cực đại (dB) | Trung bình độ lớn thay đổi mỗi 10 ms (dB) |
|---|---:|---:|---:|---:|
| Gtx | 25,21 | 101,27 | 115,48 | 0,042 |
| FSPL | 2,43 | 7,62 | 11,97 | 0,002 |
| LoS/NLoS | 19,00 | 19,00 | 19,00 | 0,413 |
| Shadow fading | 52,82 | 60,00 | 60,00 | 0,813 |
| Fast fading | 13,01 | 21,77 | 26,55 | 1,337 |
| RSRP | 74,65 | 122,54 | 138,22 | — |

Theo **biên độ trọn route**, shadow và Gtx nổi bật; FSPL nhỏ hơn nhiều. Theo **mức thay đổi trung bình mỗi 10 ms**, fast fading lớn nhất, còn FSPL gần như đứng yên. Không cộng các biên độ thành phần để dự đoán biên độ RSRP: cực đại/cực tiểu của từng thành phần thường xảy ra ở thời điểm khác nhau.

## Cực đại có liên quan đến handover không?

Biên độ RSRP lớn nhất, **138,22 dB**, nằm ở 70 km/h, route 0, sector 2. Sector này **không nằm trong top 3 RSRP ở bất kỳ bước nào** của route; RSRP trung vị của nó là −155,82 dBm. Cực đại này mô tả một sector rất yếu đi qua điểm triệt anten, ít nói lên bài toán chọn target HO.

Ngược lại, tại 90 km/h, route 0, sector 19, Gtx biến thiên **115,48 dB** và RSRP biến thiên **125,76 dB**. Sector này có mặt trong top 3 ở **78,3%** số bước và đứng thứ nhất ở **27,9%** số bước. Do đó điểm triệt anten cũng có thể xuất hiện ở một candidate HO thực sự. Để đánh giá Agent, bước phân tích kế tiếp nên cố định serving sector và chỉ xét các candidate trong cửa sổ quyết định HO, thay vì lấy cả 60 s kể cả thời gian sector không còn cạnh tranh.

## Nguồn số liệu

- `docs/research/p0_rsrp_per_sector_12routes_vi.md`: bảng **21 sector cho từng route**, đủ 252 hàng, kèm biên độ cả route, biến thiên giữa timestep và tỉ lệ vào top 3.
- `results/analysis/p0_rsrp_same_sector_step_12routes/sector_route_metrics.csv`: cùng dữ liệu ở dạng bảng để lọc/sắp xếp theo sector, site, tốc độ hoặc route; có `mean`, P95, P99 và max của `abs(δ)`.
- `results/analysis/p0_rsrp_same_sector_step_12routes/factor_report.json`: thống kê gộp và theo tốc độ.
- `results/analysis/p0_rsrp_same_sector_step_12routes/route_factor_summary.jsonl`: 12 hàng route, mỗi hàng chứa biên độ cả route, khoảng cách 3D và thống kê `abs(δ)` của từng sector.
- `data_generator/src/uav_simulation/validation/rsrp_factor_audit.py`: phép phát lại và tính thống kê.

