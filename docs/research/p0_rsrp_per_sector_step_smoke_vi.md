# P0: từng sector trên từng route

Mỗi hàng cố định một `bs_i` trong một route 60 giây (6.000 mẫu). Các cột Δ là `max − min` theo thời gian của **chính sector đó**. Không lấy cực trị sau khi gộp 21 sector. Khoảng cách là khoảng cách 3D từ UAV tới site phát. Gtx, FSPL, LoS/NLoS, shadow, fast và RSRP có đơn vị dB; khoảng cách có đơn vị mét.

Biến thiên giữa hai timestep liên tiếp dùng ký hiệu `|δ| = |x[t] − x[t−1]|`, với khoảng lấy mẫu 10 ms. `mean |δ|` là thay đổi điển hình mỗi bước; P95 là mức chỉ vượt ở 5% bước; P99 và max cho thấy các bước nhảy hiếm. Các chỉ số `|δRSRP|` tính trực tiếp từ CSV; các chỉ số thành phần tính từ phép phát lại generator.

Có 7 site, mỗi site có 3 sector hướng 0°, 120°, 240° (`bs_0`–`bs_2` cùng site 0, v.v.). Ba sector cùng site chia sẻ vị trí, nên khoảng cách và FSPL giống nhau; cấu hình P0 còn chia sẻ trạng thái LoS và shadow theo site. Gtx khác nhau vì hướng anten khác nhau; fast fading có trạng thái riêng theo sector. Vì vậy cần đọc cả `bs_i`, không chỉ site.

`Top 3 %` là tỉ lệ bước mà sector nằm trong ba sector có RSRP cao nhất, tính từ CSV đã làm tròn. Biên độ Δ vẫn dùng **toàn bộ route**, kể cả khi sector không nằm trong top 3. CSV chi tiết có thêm RSRP nhỏ nhất/lớn nhất và tỉ lệ đứng thứ nhất.

Ngay dưới mỗi bảng có phần đánh giá riêng cho route. “Candidate” trong phần này chỉ là sector nằm trong top 3 ít nhất 10% thời gian; đây là tiêu chí đọc dữ liệu, không phải luật HO. “Sector mạnh nhất theo mẫu” cũng chỉ là argmax RSRP tức thời, chưa áp dụng A3, hysteresis hay TTT.

ΔRSRP, RSRP nhỏ nhất/lớn nhất và thứ hạng được tính trực tiếp từ CSV dataset (đã làm tròn 0,01 dB); biên độ các thành phần được tính từ phép phát lại generator. Chênh lệch biên độ RSRP giữa hai nguồn được kiểm tra không quá 0,02 dB.

Tổng cộng **21 hàng sector–route**. Dấu chấm trong bảng là dấu thập phân.

## 30 km/h · route 0

UAV đi khoảng **500.0 m**. Biên độ RSRP lớn nhất thuộc `bs_8`: **119.93 dB**, xuất hiện trong top 3 **0.0%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_11` (**84.2%**), với biên độ RSRP **59.82 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **6/21 sector**. Trong nhóm đó, `bs_7` có biên độ RSRP lớn nhất (**116.21 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 536.4 → 614.5 | 5.56 | 1.18 | 19 | 33.33 | 18.22 | 58.67 | 1.60 | 4.15 | 72.8 |
| `bs_1` (0/120°) | 536.4 → 614.5 | 25.83 | 1.18 | 19 | 33.33 | 17.24 | 51.06 | 1.59 | 4.37 | 0.1 |
| `bs_2` (0/240°) | 536.4 → 614.5 | 15.07 | 1.18 | 19 | 33.33 | 20.07 | 50.42 | 1.54 | 4.15 | 0.2 |
| `bs_3` (1/0°) | 512.3 → 727.3 | 7.85 | 3.04 | 19 | 49.17 | 22.12 | 65.34 | 2.05 | 6.61 | 41.8 |
| `bs_4` (1/120°) | 512.3 → 727.3 | 5.02 | 3.04 | 19 | 49.17 | 20.23 | 67.08 | 2.03 | 6.17 | 0.0 |
| `bs_5` (1/240°) | 512.3 → 727.3 | 10.14 | 3.04 | 19 | 49.17 | 22.67 | 68.91 | 1.97 | 5.88 | 0.0 |
| `bs_6` (2/0°) | 795.2 → 1017.5 | 101.07 | 2.14 | 19 | 53.85 | 20.23 | 114.45 | 2.19 | 6.54 | 0.0 |
| `bs_7` (2/120°) | 795.2 → 1017.5 | 101.62 | 2.14 | 19 | 53.85 | 25.00 | 116.21 | 2.23 | 6.78 | 22.3 |
| `bs_8` (2/240°) | 795.2 → 1017.5 | 102.57 | 2.14 | 19 | 53.85 | 21.42 | 119.93 | 2.25 | 6.94 | 0.0 |
| `bs_9` (3/0°) | 311.0 → 412.4 | 12.58 | 2.45 | 19 | 29.71 | 15.51 | 59.53 | 1.55 | 3.85 | 11.8 |
| `bs_10` (3/120°) | 311.0 → 412.4 | 28.83 | 2.45 | 19 | 29.71 | 14.29 | 66.33 | 1.57 | 4.01 | 57.5 |
| `bs_11` (3/240°) | 311.0 → 412.4 | 13.07 | 2.45 | 19 | 29.71 | 17.66 | 59.82 | 1.55 | 4.00 | 84.2 |
| `bs_12` (4/0°) | 897.1 → 1046.2 | 96.13 | 1.33 | 19 | 48.98 | 26.06 | 117.53 | 2.52 | 7.59 | 0.0 |
| `bs_13` (4/120°) | 897.1 → 1046.2 | 93.93 | 1.33 | 19 | 48.98 | 22.82 | 119.36 | 2.41 | 7.12 | 0.0 |
| `bs_14` (4/240°) | 897.1 → 1046.2 | 95.08 | 1.33 | 19 | 48.98 | 20.05 | 118.31 | 2.44 | 7.23 | 8.8 |
| `bs_15` (5/0°) | 1094.5 → 1256.0 | 5.41 | 1.20 | 19 | 51.24 | 19.44 | 63.36 | 2.01 | 5.65 | 0.0 |
| `bs_16` (5/120°) | 1094.5 → 1256.0 | 5.72 | 1.20 | 19 | 51.24 | 19.06 | 64.65 | 2.00 | 5.42 | 0.1 |
| `bs_17` (5/240°) | 1094.5 → 1256.0 | 8.52 | 1.20 | 19 | 51.24 | 21.24 | 65.42 | 2.02 | 5.62 | 0.0 |
| `bs_18` (6/0°) | 1151.4 → 1243.3 | 19.10 | 0.67 | 19 | 53.96 | 22.28 | 67.56 | 1.92 | 4.95 | 0.0 |
| `bs_19` (6/120°) | 1151.4 → 1243.3 | 25.85 | 0.67 | 19 | 53.96 | 23.01 | 74.78 | 1.94 | 5.00 | 0.0 |
| `bs_20` (6/240°) | 1151.4 → 1243.3 | 12.41 | 0.67 | 19 | 53.96 | 21.68 | 68.97 | 1.94 | 4.97 | 0.3 |

### Biến thiên giữa timestep 10 ms của candidate

| BS | mean abs(δRSRP) | P95 abs(δRSRP) | P99 abs(δRSRP) | max abs(δRSRP) | Thành phần có mean abs(δ) lớn nhất |
|---|---:|---:|---:|---:|---|
| `bs_11` | 1.55 | 4.00 | 5.66 | 30.05 | fast fading: 1.50 dB/10 ms |
| `bs_0` | 1.60 | 4.15 | 8.60 | 41.82 | fast fading: 1.48 dB/10 ms |
| `bs_10` | 1.57 | 4.01 | 5.66 | 34.83 | fast fading: 1.51 dB/10 ms |
| `bs_3` | 2.05 | 6.61 | 12.81 | 42.93 | fast fading: 1.80 dB/10 ms |
| `bs_7` | 2.23 | 6.78 | 16.57 | 55.54 | fast fading: 1.86 dB/10 ms |
| `bs_9` | 1.55 | 3.85 | 5.80 | 32.81 | fast fading: 1.49 dB/10 ms |

### Đánh giá route

- **Các candidate đo đạc:** 6 sector ở 4 site có mặt trong top 3 ít nhất 10% thời gian: `bs_11` 84.2%, `bs_0` 72.8%, `bs_10` 57.5%, `bs_3` 41.8%, `bs_7` 22.3%, `bs_9` 11.8%.
- **Sector mạnh nhất theo từng mẫu:** có 6 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_11` 50.6%, `bs_0` 24.5%, `bs_10` 12.3%, `bs_3` 10.8%, `bs_7` 1.3%, `bs_14` 0.5%. Identity của sector có RSRP lớn nhất đổi 1100 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_7` (site 2, 120°) đi từ -199.47 đến -83.26 dBm, ΔRSRP = 116.21 dB. Các biên độ thành phần riêng: ΔGtx 101.62 dB, Δshadow 53.85 dB, Δfast fading 25.00 dB, ΔLoS/NLoS 19.00 dB, ΔFSPL 2.14 dB.
- **Hình học của candidate:** `bs_3` (site 1, 0°) có ΔFSPL lớn nhất trong nhóm candidate, 3.04 dB, khi khoảng cách 3D đổi từ 512.3 đến 727.3 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_7` (site 2, 120°) có ΔGtx = 101.62 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_7` (site 2, 120°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.23 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_7` (site 2, 120°), 6.78 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_8` (site 2, 240°) có ΔRSRP lớn nhất toàn bảng (119.93 dB) nhưng chỉ vào top 3 0.0% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

