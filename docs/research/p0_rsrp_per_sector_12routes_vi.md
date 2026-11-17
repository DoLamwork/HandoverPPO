# P0: từng sector trên từng route

Mỗi hàng cố định một `bs_i` trong một route 60 giây (6.000 mẫu). Các cột Δ là `max − min` theo thời gian của **chính sector đó**. Không lấy cực trị sau khi gộp 21 sector. Khoảng cách là khoảng cách 3D từ UAV tới site phát. Gtx, FSPL, LoS/NLoS, shadow, fast và RSRP có đơn vị dB; khoảng cách có đơn vị mét.

Biến thiên giữa hai timestep liên tiếp dùng ký hiệu `|δ| = |x[t] − x[t−1]|`, với khoảng lấy mẫu 10 ms. `mean |δ|` là thay đổi điển hình mỗi bước; P95 là mức chỉ vượt ở 5% bước; P99 và max cho thấy các bước nhảy hiếm. Các chỉ số `|δRSRP|` tính trực tiếp từ CSV; các chỉ số thành phần tính từ phép phát lại generator.

Có 7 site, mỗi site có 3 sector hướng 0°, 120°, 240° (`bs_0`–`bs_2` cùng site 0, v.v.). Ba sector cùng site chia sẻ vị trí, nên khoảng cách và FSPL giống nhau; cấu hình P0 còn chia sẻ trạng thái LoS và shadow theo site. Gtx khác nhau vì hướng anten khác nhau; fast fading có trạng thái riêng theo sector. Vì vậy cần đọc cả `bs_i`, không chỉ site.

`Top 3 %` là tỉ lệ bước mà sector nằm trong ba sector có RSRP cao nhất, tính từ CSV đã làm tròn. Biên độ Δ vẫn dùng **toàn bộ route**, kể cả khi sector không nằm trong top 3. CSV chi tiết có thêm RSRP nhỏ nhất/lớn nhất và tỉ lệ đứng thứ nhất.

Ngay dưới mỗi bảng có phần đánh giá riêng cho route. “Candidate” trong phần này chỉ là sector nằm trong top 3 ít nhất 10% thời gian; đây là tiêu chí đọc dữ liệu, không phải luật HO. “Sector mạnh nhất theo mẫu” cũng chỉ là argmax RSRP tức thời, chưa áp dụng A3, hysteresis hay TTT.

ΔRSRP, RSRP nhỏ nhất/lớn nhất và thứ hạng được tính trực tiếp từ CSV dataset (đã làm tròn 0,01 dB); biên độ các thành phần được tính từ phép phát lại generator. Chênh lệch biên độ RSRP giữa hai nguồn được kiểm tra không quá 0,02 dB.

Tổng cộng **252 hàng sector–route**. Dấu chấm trong bảng là dấu thập phân.

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

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_11` | 0.005 | 0.001 | 0.025 | 0.320 | 1.498 | 1.547 |
| `bs_0` | 0.003 | 0.001 | 0.095 | 0.386 | 1.483 | 1.597 |
| `bs_10` | 0.007 | 0.001 | 0.025 | 0.320 | 1.507 | 1.565 |
| `bs_3` | 0.002 | 0.001 | 0.225 | 0.496 | 1.798 | 2.051 |
| `bs_7` | 0.061 | 0.001 | 0.326 | 0.566 | 1.863 | 2.225 |
| `bs_9` | 0.003 | 0.001 | 0.025 | 0.320 | 1.489 | 1.550 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_11` | 0.011 | 0.002 | 0.000 | 0.772 | 3.934 | 4.000 |
| `bs_0` | 0.006 | 0.001 | 0.000 | 0.950 | 3.902 | 4.151 |
| `bs_10` | 0.015 | 0.002 | 0.000 | 0.772 | 3.920 | 4.010 |
| `bs_3` | 0.004 | 0.001 | 0.000 | 1.189 | 5.976 | 6.610 |
| `bs_7` | 0.116 | 0.001 | 0.000 | 1.320 | 5.943 | 6.782 |
| `bs_9` | 0.011 | 0.002 | 0.000 | 0.772 | 3.824 | 3.851 |

### Đánh giá route

- **Các candidate đo đạc:** 6 sector ở 4 site có mặt trong top 3 ít nhất 10% thời gian: `bs_11` 84.2%, `bs_0` 72.8%, `bs_10` 57.5%, `bs_3` 41.8%, `bs_7` 22.3%, `bs_9` 11.8%.
- **Sector mạnh nhất theo từng mẫu:** có 6 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_11` 50.6%, `bs_0` 24.5%, `bs_10` 12.3%, `bs_3` 10.8%, `bs_7` 1.3%, `bs_14` 0.5%. Identity của sector có RSRP lớn nhất đổi 1100 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_7` (site 2, 120°) đi từ -199.47 đến -83.26 dBm, ΔRSRP = 116.21 dB. Các biên độ thành phần riêng: ΔGtx 101.62 dB, Δshadow 53.85 dB, Δfast fading 25.00 dB, ΔLoS/NLoS 19.00 dB, ΔFSPL 2.14 dB.
- **Hình học của candidate:** `bs_3` (site 1, 0°) có ΔFSPL lớn nhất trong nhóm candidate, 3.04 dB, khi khoảng cách 3D đổi từ 512.3 đến 727.3 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_7` (site 2, 120°) có ΔGtx = 101.62 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_7` (site 2, 120°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.23 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_7` (site 2, 120°), 6.78 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_8` (site 2, 240°) có ΔRSRP lớn nhất toàn bảng (119.93 dB) nhưng chỉ vào top 3 0.0% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

## 30 km/h · route 10

UAV đi khoảng **500.0 m**. Biên độ RSRP lớn nhất thuộc `bs_4`: **124.83 dB**, xuất hiện trong top 3 **0.0%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_0` (**84.2%**), với biên độ RSRP **48.18 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **6/21 sector**. Trong nhóm đó, `bs_11` có biên độ RSRP lớn nhất (**123.70 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 719.6 → 904.3 | 12.76 | 1.98 | 19 | 29.22 | 18.43 | 48.18 | 1.64 | 4.36 | 84.2 |
| `bs_1` (0/120°) | 719.6 → 904.3 | 13.46 | 1.98 | 19 | 29.22 | 20.54 | 46.67 | 1.66 | 4.54 | 0.0 |
| `bs_2` (0/240°) | 719.6 → 904.3 | 10.21 | 1.98 | 19 | 29.22 | 19.87 | 48.48 | 1.66 | 4.57 | 0.0 |
| `bs_3` (1/0°) | 865.5 → 1082.8 | 77.79 | 1.95 | 19 | 47.12 | 23.85 | 113.72 | 2.27 | 6.90 | 0.9 |
| `bs_4` (1/120°) | 865.5 → 1082.8 | 78.57 | 1.95 | 19 | 47.12 | 16.81 | 124.83 | 2.18 | 6.45 | 0.0 |
| `bs_5` (1/240°) | 865.5 → 1082.8 | 79.86 | 1.95 | 19 | 47.12 | 21.23 | 120.37 | 2.22 | 6.76 | 0.0 |
| `bs_6` (2/0°) | 637.0 → 743.1 | 19.10 | 1.34 | 19 | 36.96 | 15.94 | 69.85 | 1.76 | 4.70 | 0.0 |
| `bs_7` (2/120°) | 637.0 → 743.1 | 70.90 | 1.34 | 19 | 36.96 | 19.45 | 71.55 | 1.79 | 4.63 | 28.9 |
| `bs_8` (2/240°) | 637.0 → 743.1 | 74.91 | 1.34 | 19 | 36.96 | 17.91 | 84.24 | 1.79 | 4.64 | 0.0 |
| `bs_9` (3/0°) | 153.1 → 264.9 | 91.99 | 4.76 | 0 | 26.62 | 10.68 | 95.88 | 1.48 | 3.71 | 21.7 |
| `bs_10` (3/120°) | 153.1 → 264.9 | 109.02 | 4.76 | 0 | 26.62 | 9.45 | 113.40 | 1.52 | 3.72 | 7.1 |
| `bs_11` (3/240°) | 153.1 → 264.9 | 99.88 | 4.76 | 0 | 26.62 | 9.23 | 123.70 | 1.46 | 3.67 | 81.8 |
| `bs_12` (4/0°) | 1002.3 → 1106.6 | 25.10 | 0.86 | 19 | 43.36 | 21.23 | 80.96 | 2.54 | 7.16 | 0.0 |
| `bs_13` (4/120°) | 1002.3 → 1106.6 | 83.62 | 0.86 | 19 | 43.36 | 24.20 | 124.60 | 2.73 | 7.87 | 0.0 |
| `bs_14` (4/240°) | 1002.3 → 1106.6 | 70.00 | 0.86 | 19 | 43.36 | 21.39 | 95.23 | 2.60 | 7.29 | 0.0 |
| `bs_15` (5/0°) | 684.1 → 904.3 | 14.92 | 2.42 | 19 | 45.07 | 16.10 | 65.02 | 1.58 | 3.72 | 0.8 |
| `bs_16` (5/120°) | 684.1 → 904.3 | 13.15 | 2.42 | 19 | 45.07 | 18.68 | 63.96 | 1.58 | 3.78 | 58.1 |
| `bs_17` (5/240°) | 684.1 → 904.3 | 17.69 | 2.42 | 19 | 45.07 | 16.76 | 66.73 | 1.57 | 3.70 | 0.0 |
| `bs_18` (6/0°) | 865.5 → 1024.2 | 91.13 | 1.46 | 19 | 42.93 | 22.16 | 114.25 | 1.52 | 4.01 | 0.0 |
| `bs_19` (6/120°) | 865.5 → 1024.2 | 90.65 | 1.46 | 19 | 42.93 | 26.55 | 111.68 | 1.50 | 3.88 | 0.0 |
| `bs_20` (6/240°) | 865.5 → 1024.2 | 90.10 | 1.46 | 19 | 42.93 | 21.62 | 110.39 | 1.55 | 3.96 | 16.4 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_0` | 0.004 | 0.001 | 0.165 | 0.416 | 1.492 | 1.644 |
| `bs_11` | 0.076 | 0.002 | 0.000 | 0.297 | 1.407 | 1.463 |
| `bs_16` | 0.004 | 0.001 | 0.162 | 0.400 | 1.388 | 1.580 |
| `bs_7` | 0.032 | 0.000 | 0.095 | 0.380 | 1.648 | 1.790 |
| `bs_9` | 0.053 | 0.002 | 0.000 | 0.297 | 1.425 | 1.476 |
| `bs_20` | 0.091 | 0.001 | 0.089 | 0.376 | 1.398 | 1.550 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_0` | 0.008 | 0.001 | 0.000 | 0.927 | 4.121 | 4.361 |
| `bs_11` | 0.173 | 0.004 | 0.000 | 0.741 | 3.546 | 3.670 |
| `bs_16` | 0.008 | 0.001 | 0.000 | 0.823 | 3.535 | 3.782 |
| `bs_7` | 0.082 | 0.001 | 0.000 | 0.913 | 4.384 | 4.631 |
| `bs_9` | 0.116 | 0.004 | 0.000 | 0.741 | 3.613 | 3.711 |
| `bs_20` | 0.230 | 0.001 | 0.000 | 0.868 | 3.622 | 3.961 |

### Đánh giá route

- **Các candidate đo đạc:** 6 sector ở 5 site có mặt trong top 3 ít nhất 10% thời gian: `bs_0` 84.2%, `bs_11` 81.8%, `bs_16` 58.1%, `bs_7` 28.9%, `bs_9` 21.7%, `bs_20` 16.4%.
- **Sector mạnh nhất theo từng mẫu:** có 6 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_11` 56.6%, `bs_0` 25.8%, `bs_16` 10.4%, `bs_20` 3.9%, `bs_9` 2.6%, `bs_7` 0.7%. Identity của sector có RSRP lớn nhất đổi 546 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_11` (site 3, 240°) đi từ -179.61 đến -55.91 dBm, ΔRSRP = 123.70 dB. Các biên độ thành phần riêng: ΔGtx 99.88 dB, Δshadow 26.62 dB, Δfast fading 9.23 dB, ΔFSPL 4.76 dB, ΔLoS/NLoS 0.00 dB.
- **Hình học của candidate:** `bs_9` (site 3, 0°) có ΔFSPL lớn nhất trong nhóm candidate, 4.76 dB, khi khoảng cách 3D đổi từ 153.1 đến 264.9 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_11` (site 3, 240°) có ΔGtx = 99.88 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_7` (site 2, 120°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 1.79 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_7` (site 2, 120°), 4.63 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_4` (site 1, 120°) có ΔRSRP lớn nhất toàn bảng (124.83 dB) nhưng chỉ vào top 3 0.0% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

## 50 km/h · route 0

UAV đi khoảng **833.3 m**. Biên độ RSRP lớn nhất thuộc `bs_7`: **115.51 dB**, xuất hiện trong top 3 **0.3%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_2` (**99.9%**), với biên độ RSRP **47.09 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **6/21 sector**. Trong nhóm đó, `bs_20` có biên độ RSRP lớn nhất (**74.93 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 344.1 → 478.0 | 14.08 | 2.85 | 19 | 22.44 | 12.27 | 46.17 | 1.31 | 3.33 | 0.2 |
| `bs_1` (0/120°) | 344.1 → 478.0 | 7.09 | 2.85 | 19 | 22.44 | 14.76 | 50.84 | 1.29 | 3.17 | 1.8 |
| `bs_2` (0/240°) | 344.1 → 478.0 | 11.64 | 2.85 | 19 | 22.44 | 13.14 | 47.09 | 1.28 | 3.18 | 99.9 |
| `bs_3` (1/0°) | 1178.5 → 1293.1 | 13.87 | 0.81 | 19 | 50.23 | 16.96 | 60.97 | 1.88 | 5.39 | 0.0 |
| `bs_4` (1/120°) | 1178.5 → 1293.1 | 11.49 | 0.81 | 19 | 50.23 | 20.14 | 62.31 | 1.84 | 4.83 | 0.0 |
| `bs_5` (1/240°) | 1178.5 → 1293.1 | 16.70 | 0.81 | 19 | 50.23 | 18.42 | 62.29 | 1.81 | 4.75 | 31.6 |
| `bs_6` (2/0°) | 804.2 → 1011.5 | 101.79 | 1.99 | 19 | 53.31 | 19.17 | 112.47 | 2.04 | 6.13 | 0.0 |
| `bs_7` (2/120°) | 804.2 → 1011.5 | 102.14 | 1.99 | 19 | 53.31 | 16.60 | 115.51 | 2.01 | 5.56 | 0.3 |
| `bs_8` (2/240°) | 804.2 → 1011.5 | 101.51 | 1.99 | 19 | 53.31 | 14.82 | 111.66 | 1.97 | 5.21 | 0.0 |
| `bs_9` (3/0°) | 1157.7 → 1304.0 | 4.54 | 1.03 | 19 | 47.53 | 14.47 | 54.45 | 2.01 | 5.35 | 0.0 |
| `bs_10` (3/120°) | 1157.7 → 1304.0 | 7.91 | 1.03 | 19 | 47.53 | 18.51 | 54.20 | 2.15 | 6.22 | 0.0 |
| `bs_11` (3/240°) | 1157.7 → 1304.0 | 7.27 | 1.03 | 19 | 47.53 | 15.77 | 59.24 | 2.14 | 6.20 | 48.9 |
| `bs_12` (4/0°) | 1836.9 → 1973.3 | 3.39 | 0.62 | 19 | 56.86 | 17.48 | 70.44 | 2.49 | 7.10 | 0.0 |
| `bs_13` (4/120°) | 1836.9 → 1973.3 | 2.70 | 0.62 | 19 | 56.86 | 17.95 | 62.31 | 2.35 | 6.33 | 0.0 |
| `bs_14` (4/240°) | 1836.9 → 1973.3 | 0.92 | 0.62 | 19 | 56.86 | 22.07 | 67.14 | 2.38 | 6.36 | 71.9 |
| `bs_15` (5/0°) | 1602.7 → 1796.8 | 3.46 | 0.99 | 19 | 56.11 | 21.90 | 66.46 | 2.54 | 7.08 | 0.0 |
| `bs_16` (5/120°) | 1602.7 → 1796.8 | 7.66 | 0.99 | 19 | 56.11 | 17.74 | 69.73 | 2.56 | 7.00 | 0.0 |
| `bs_17` (5/240°) | 1602.7 → 1796.8 | 1.68 | 0.99 | 19 | 56.11 | 18.29 | 65.63 | 2.61 | 7.49 | 13.0 |
| `bs_18` (6/0°) | 2003.0 → 2161.5 | 5.66 | 0.66 | 19 | 59.51 | 17.88 | 74.51 | 2.36 | 6.75 | 0.0 |
| `bs_19` (6/120°) | 2003.0 → 2161.5 | 10.61 | 0.66 | 19 | 59.51 | 18.18 | 74.45 | 2.46 | 7.37 | 0.0 |
| `bs_20` (6/240°) | 2003.0 → 2161.5 | 9.10 | 0.66 | 19 | 59.51 | 16.58 | 74.93 | 2.40 | 6.82 | 32.5 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_2` | 0.005 | 0.002 | 0.019 | 0.400 | 1.202 | 1.276 |
| `bs_14` | 0.001 | 0.000 | 0.333 | 0.765 | 2.001 | 2.378 |
| `bs_11` | 0.004 | 0.001 | 0.348 | 0.674 | 1.749 | 2.143 |
| `bs_20` | 0.004 | 0.000 | 0.443 | 0.771 | 1.915 | 2.401 |
| `bs_5` | 0.008 | 0.000 | 0.396 | 0.627 | 1.383 | 1.813 |
| `bs_17` | 0.001 | 0.000 | 0.431 | 0.834 | 2.089 | 2.605 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_2` | 0.020 | 0.003 | 0.000 | 0.979 | 3.007 | 3.180 |
| `bs_14` | 0.001 | 0.001 | 0.000 | 1.924 | 5.597 | 6.360 |
| `bs_11` | 0.010 | 0.001 | 0.000 | 1.695 | 5.229 | 6.200 |
| `bs_20` | 0.007 | 0.000 | 0.000 | 1.865 | 5.648 | 6.821 |
| `bs_5` | 0.019 | 0.001 | 0.000 | 1.377 | 3.749 | 4.752 |
| `bs_17` | 0.002 | 0.001 | 0.000 | 1.986 | 6.442 | 7.493 |

### Đánh giá route

- **Các candidate đo đạc:** 6 sector ở 6 site có mặt trong top 3 ít nhất 10% thời gian: `bs_2` 99.9%, `bs_14` 71.9%, `bs_11` 48.9%, `bs_20` 32.5%, `bs_5` 31.6%, `bs_17` 13.0%.
- **Sector mạnh nhất theo từng mẫu:** có 3 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_2` 97.2%, `bs_14` 1.8%, `bs_20` 1.0%. Identity của sector có RSRP lớn nhất đổi 125 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_20` (site 6, 240°) đi từ -162.80 đến -87.87 dBm, ΔRSRP = 74.93 dB. Các biên độ thành phần riêng: Δshadow 59.51 dB, ΔLoS/NLoS 19.00 dB, Δfast fading 16.58 dB, ΔGtx 9.10 dB, ΔFSPL 0.66 dB.
- **Hình học của candidate:** `bs_2` (site 0, 240°) có ΔFSPL lớn nhất trong nhóm candidate, 2.85 dB, khi khoảng cách 3D đổi từ 344.1 đến 478.0 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_20` (site 6, 240°) có Δshadow = 59.51 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_17` (site 5, 240°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.61 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_17` (site 5, 240°), 7.49 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_7` (site 2, 120°) có ΔRSRP lớn nhất toàn bảng (115.51 dB) nhưng chỉ vào top 3 0.3% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

## 50 km/h · route 10

UAV đi khoảng **833.3 m**. Biên độ RSRP lớn nhất thuộc `bs_20`: **109.04 dB**, xuất hiện trong top 3 **69.7%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_16` (**92.8%**), với biên độ RSRP **102.85 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **5/21 sector**. Trong nhóm đó, `bs_20` có biên độ RSRP lớn nhất (**109.04 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 1409.9 → 1699.3 | 9.60 | 1.62 | 19 | 50.06 | 18.93 | 64.64 | 1.93 | 5.36 | 51.4 |
| `bs_1` (0/120°) | 1409.9 → 1699.3 | 12.20 | 1.62 | 19 | 50.06 | 16.97 | 65.63 | 1.97 | 5.37 | 0.0 |
| `bs_2` (0/240°) | 1409.9 → 1699.3 | 4.24 | 1.62 | 19 | 50.06 | 14.62 | 61.11 | 2.05 | 6.02 | 0.0 |
| `bs_3` (1/0°) | 1400.9 → 1746.2 | 5.67 | 1.91 | 19 | 46.66 | 18.06 | 62.75 | 2.08 | 5.61 | 30.5 |
| `bs_4` (1/120°) | 1400.9 → 1746.2 | 3.77 | 1.91 | 19 | 46.66 | 17.61 | 61.11 | 2.24 | 6.25 | 0.0 |
| `bs_5` (1/240°) | 1400.9 → 1746.2 | 6.02 | 1.91 | 19 | 46.66 | 19.15 | 61.45 | 2.24 | 6.29 | 0.0 |
| `bs_6` (2/0°) | 1028.9 → 1228.3 | 16.88 | 1.54 | 19 | 53.27 | 21.31 | 58.60 | 1.87 | 5.42 | 0.1 |
| `bs_7` (2/120°) | 1028.9 → 1228.3 | 21.85 | 1.54 | 19 | 53.27 | 15.52 | 78.66 | 1.82 | 4.86 | 0.0 |
| `bs_8` (2/240°) | 1028.9 → 1228.3 | 10.95 | 1.54 | 19 | 53.27 | 17.29 | 62.85 | 1.83 | 4.93 | 0.0 |
| `bs_9` (3/0°) | 604.7 → 912.9 | 17.85 | 3.58 | 19 | 50.08 | 18.38 | 73.65 | 2.17 | 5.93 | 47.7 |
| `bs_10` (3/120°) | 604.7 → 912.9 | 15.88 | 3.58 | 19 | 50.08 | 16.57 | 72.37 | 2.29 | 6.38 | 0.0 |
| `bs_11` (3/240°) | 604.7 → 912.9 | 23.46 | 3.58 | 19 | 50.08 | 17.19 | 79.61 | 2.27 | 6.24 | 0.0 |
| `bs_12` (4/0°) | 1002.4 → 1347.1 | 25.32 | 2.57 | 19 | 53.89 | 18.09 | 69.81 | 2.05 | 5.86 | 0.2 |
| `bs_13` (4/120°) | 1002.4 → 1347.1 | 21.48 | 2.57 | 19 | 53.89 | 20.52 | 67.29 | 2.12 | 6.29 | 0.0 |
| `bs_14` (4/240°) | 1002.4 → 1347.1 | 17.06 | 2.57 | 19 | 53.89 | 17.49 | 66.20 | 2.09 | 6.03 | 0.0 |
| `bs_15` (5/0°) | 343.3 → 526.9 | 32.07 | 3.72 | 19 | 21.15 | 9.13 | 53.51 | 1.15 | 2.75 | 4.8 |
| `bs_16` (5/120°) | 343.3 → 526.9 | 85.11 | 3.72 | 19 | 21.15 | 7.56 | 102.85 | 1.13 | 2.78 | 92.8 |
| `bs_17` (5/240°) | 343.3 → 526.9 | 83.55 | 3.72 | 19 | 21.15 | 8.39 | 102.57 | 1.15 | 2.77 | 2.7 |
| `bs_18` (6/0°) | 377.4 → 590.8 | 21.78 | 3.89 | 19 | 40.45 | 12.84 | 50.62 | 1.18 | 2.85 | 0.0 |
| `bs_19` (6/120°) | 377.4 → 590.8 | 88.91 | 3.89 | 19 | 40.45 | 18.13 | 97.93 | 1.20 | 2.84 | 0.0 |
| `bs_20` (6/240°) | 377.4 → 590.8 | 93.69 | 3.89 | 19 | 40.45 | 16.44 | 109.04 | 1.22 | 2.83 | 69.7 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_16` | 0.029 | 0.002 | 0.032 | 0.387 | 1.036 | 1.132 |
| `bs_20` | 0.030 | 0.002 | 0.038 | 0.411 | 1.102 | 1.223 |
| `bs_0` | 0.003 | 0.000 | 0.355 | 0.683 | 1.511 | 1.932 |
| `bs_9` | 0.004 | 0.001 | 0.399 | 0.714 | 1.650 | 2.170 |
| `bs_3` | 0.003 | 0.000 | 0.304 | 0.684 | 1.702 | 2.082 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_16` | 0.058 | 0.003 | 0.000 | 0.939 | 2.596 | 2.781 |
| `bs_20` | 0.058 | 0.003 | 0.000 | 0.980 | 2.649 | 2.830 |
| `bs_0` | 0.005 | 0.001 | 0.000 | 1.599 | 4.541 | 5.360 |
| `bs_9` | 0.013 | 0.002 | 0.000 | 1.795 | 4.750 | 5.931 |
| `bs_3` | 0.004 | 0.001 | 0.000 | 1.807 | 4.895 | 5.610 |

### Đánh giá route

- **Các candidate đo đạc:** 5 sector ở 5 site có mặt trong top 3 ít nhất 10% thời gian: `bs_16` 92.8%, `bs_20` 69.7%, `bs_0` 51.4%, `bs_9` 47.7%, `bs_3` 30.5%.
- **Sector mạnh nhất theo từng mẫu:** có 5 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_16` 38.5%, `bs_20` 33.3%, `bs_0` 16.0%, `bs_9` 6.3%, `bs_3` 5.8%. Identity của sector có RSRP lớn nhất đổi 847 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_20` (site 6, 240°) đi từ -173.30 đến -64.26 dBm, ΔRSRP = 109.04 dB. Các biên độ thành phần riêng: ΔGtx 93.69 dB, Δshadow 40.45 dB, ΔLoS/NLoS 19.00 dB, Δfast fading 16.44 dB, ΔFSPL 3.89 dB.
- **Hình học của candidate:** `bs_20` (site 6, 240°) có ΔFSPL lớn nhất trong nhóm candidate, 3.89 dB, khi khoảng cách 3D đổi từ 377.4 đến 590.8 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_20` (site 6, 240°) có ΔGtx = 93.69 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_9` (site 3, 0°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.17 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_9` (site 3, 0°), 5.93 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị cũng là candidate:** `bs_20` (site 6, 240°) vừa có ΔRSRP lớn nhất toàn bảng (109.04 dB), vừa nằm trong top 3 69.7% thời gian. Đây là sector cần ưu tiên khi phân tích quyết định HO của route này.

## 70 km/h · route 0

UAV đi khoảng **1166.4 m**. Biên độ RSRP lớn nhất thuộc `bs_2`: **138.22 dB**, xuất hiện trong top 3 **0.0%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_3` (**88.0%**), với biên độ RSRP **60.95 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **7/21 sector**. Trong nhóm đó, `bs_16` có biên độ RSRP lớn nhất (**119.43 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 1552.0 → 1930.6 | 80.05 | 1.90 | 19 | 48.22 | 17.71 | 104.27 | 2.43 | 6.42 | 5.0 |
| `bs_1` (0/120°) | 1552.0 → 1930.6 | 21.44 | 1.90 | 19 | 48.22 | 15.90 | 71.87 | 2.65 | 7.39 | 0.0 |
| `bs_2` (0/240°) | 1552.0 → 1930.6 | 115.31 | 1.90 | 19 | 48.22 | 17.95 | 138.22 | 2.61 | 6.95 | 0.0 |
| `bs_3` (1/0°) | 1358.7 → 1657.1 | 4.15 | 1.72 | 19 | 49.39 | 14.86 | 60.95 | 1.84 | 5.35 | 88.0 |
| `bs_4` (1/120°) | 1358.7 → 1657.1 | 6.81 | 1.72 | 19 | 49.39 | 14.60 | 57.63 | 1.82 | 5.20 | 0.0 |
| `bs_5` (1/240°) | 1358.7 → 1657.1 | 8.50 | 1.72 | 19 | 49.39 | 15.01 | 63.25 | 1.87 | 5.47 | 0.1 |
| `bs_6` (2/0°) | 1299.7 → 1693.4 | 10.14 | 2.30 | 19 | 52.12 | 13.18 | 64.18 | 1.87 | 4.94 | 8.9 |
| `bs_7` (2/120°) | 1299.7 → 1693.4 | 9.25 | 2.30 | 19 | 52.12 | 14.79 | 69.42 | 1.94 | 5.29 | 5.9 |
| `bs_8` (2/240°) | 1299.7 → 1693.4 | 4.98 | 2.30 | 19 | 52.12 | 12.76 | 63.24 | 1.85 | 4.65 | 0.0 |
| `bs_9` (3/0°) | 691.2 → 1066.1 | 94.73 | 3.76 | 19 | 51.29 | 14.75 | 107.96 | 1.90 | 4.93 | 13.1 |
| `bs_10` (3/120°) | 691.2 → 1066.1 | 97.52 | 3.76 | 19 | 51.29 | 13.48 | 113.40 | 1.88 | 4.89 | 0.0 |
| `bs_11` (3/240°) | 691.2 → 1066.1 | 92.59 | 3.76 | 19 | 51.29 | 13.98 | 109.01 | 1.91 | 5.10 | 0.0 |
| `bs_12` (4/0°) | 674.8 → 987.5 | 78.13 | 3.31 | 19 | 55.27 | 13.00 | 110.38 | 1.73 | 4.51 | 26.3 |
| `bs_13` (4/120°) | 674.8 → 987.5 | 80.73 | 3.31 | 19 | 55.27 | 20.29 | 116.60 | 1.85 | 5.52 | 0.0 |
| `bs_14` (4/240°) | 674.8 → 987.5 | 90.67 | 3.31 | 19 | 55.27 | 15.20 | 117.16 | 1.83 | 5.32 | 0.0 |
| `bs_15` (5/0°) | 705.3 → 1013.5 | 90.56 | 3.15 | 19 | 45.66 | 14.57 | 102.49 | 1.82 | 5.30 | 0.0 |
| `bs_16` (5/120°) | 705.3 → 1013.5 | 99.10 | 3.15 | 19 | 45.66 | 15.71 | 119.43 | 1.75 | 4.45 | 11.8 |
| `bs_17` (5/240°) | 705.3 → 1013.5 | 101.61 | 3.15 | 19 | 45.66 | 15.09 | 134.83 | 1.86 | 5.33 | 0.0 |
| `bs_18` (6/0°) | 90.1 → 216.7 | 98.58 | 7.62 | 19 | 30.65 | 9.61 | 91.15 | 1.37 | 3.20 | 48.8 |
| `bs_19` (6/120°) | 90.1 → 216.7 | 107.30 | 7.62 | 19 | 30.65 | 12.77 | 113.03 | 1.35 | 3.13 | 60.2 |
| `bs_20` (6/240°) | 90.1 → 216.7 | 107.93 | 7.62 | 19 | 30.65 | 11.47 | 113.21 | 1.36 | 3.20 | 31.8 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_3` | 0.002 | 0.001 | 0.475 | 0.748 | 1.261 | 1.842 |
| `bs_19` | 0.173 | 0.006 | 0.057 | 0.499 | 1.116 | 1.351 |
| `bs_18` | 0.184 | 0.006 | 0.057 | 0.499 | 1.124 | 1.374 |
| `bs_20` | 0.189 | 0.006 | 0.057 | 0.499 | 1.115 | 1.364 |
| `bs_12` | 0.066 | 0.001 | 0.336 | 0.715 | 1.309 | 1.730 |
| `bs_9` | 0.121 | 0.001 | 0.481 | 0.724 | 1.302 | 1.899 |
| `bs_16` | 0.094 | 0.001 | 0.431 | 0.747 | 1.216 | 1.749 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_3` | 0.003 | 0.001 | 0.000 | 1.825 | 3.815 | 5.350 |
| `bs_19` | 0.461 | 0.010 | 0.000 | 1.174 | 2.694 | 3.130 |
| `bs_18` | 0.567 | 0.010 | 0.000 | 1.174 | 2.752 | 3.201 |
| `bs_20` | 0.576 | 0.010 | 0.000 | 1.174 | 2.740 | 3.200 |
| `bs_12` | 0.159 | 0.002 | 0.000 | 1.749 | 3.626 | 4.511 |
| `bs_9` | 0.354 | 0.002 | 0.000 | 1.690 | 3.397 | 4.931 |
| `bs_16` | 0.232 | 0.002 | 0.000 | 1.673 | 3.371 | 4.452 |

### Đánh giá route

- **Các candidate đo đạc:** 7 sector ở 5 site có mặt trong top 3 ít nhất 10% thời gian: `bs_3` 88.0%, `bs_19` 60.2%, `bs_18` 48.8%, `bs_20` 31.8%, `bs_12` 26.3%, `bs_9` 13.1%, `bs_16` 11.8%.
- **Sector mạnh nhất theo từng mẫu:** có 7 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_19` 33.8%, `bs_3` 31.0%, `bs_20` 17.6%, `bs_18` 17.4%, `bs_7` 0.1%, `bs_9` 0.0%, `bs_12` 0.0%. Identity của sector có RSRP lớn nhất đổi 298 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_16` (site 5, 120°) đi từ -208.79 đến -89.36 dBm, ΔRSRP = 119.43 dB. Các biên độ thành phần riêng: ΔGtx 99.10 dB, Δshadow 45.66 dB, ΔLoS/NLoS 19.00 dB, Δfast fading 15.71 dB, ΔFSPL 3.15 dB.
- **Hình học của candidate:** `bs_18` (site 6, 0°) có ΔFSPL lớn nhất trong nhóm candidate, 7.62 dB, khi khoảng cách 3D đổi từ 90.1 đến 216.7 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_20` (site 6, 240°) có ΔGtx = 107.93 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_9` (site 3, 0°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 1.90 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_3` (site 1, 0°), 5.35 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_2` (site 0, 240°) có ΔRSRP lớn nhất toàn bảng (138.22 dB) nhưng chỉ vào top 3 0.0% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

## 70 km/h · route 10

UAV đi khoảng **1166.1 m**. Biên độ RSRP lớn nhất thuộc `bs_7`: **124.35 dB**, xuất hiện trong top 3 **6.1%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_16` (**67.6%**), với biên độ RSRP **65.88 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **8/21 sector**. Trong nhóm đó, `bs_14` có biên độ RSRP lớn nhất (**122.80 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 1194.8 → 1531.9 | 9.53 | 2.16 | 19 | 53.51 | 14.06 | 63.30 | 2.17 | 5.89 | 15.6 |
| `bs_1` (0/120°) | 1194.8 → 1531.9 | 7.84 | 2.16 | 19 | 53.51 | 12.62 | 59.43 | 2.15 | 5.74 | 0.1 |
| `bs_2` (0/240°) | 1194.8 → 1531.9 | 6.32 | 2.16 | 19 | 53.51 | 14.34 | 61.37 | 2.17 | 5.86 | 0.0 |
| `bs_3` (1/0°) | 667.8 → 1009.6 | 94.84 | 3.59 | 19 | 54.39 | 14.08 | 105.30 | 1.93 | 5.00 | 37.2 |
| `bs_4` (1/120°) | 667.8 → 1009.6 | 97.05 | 3.59 | 19 | 54.39 | 16.88 | 108.24 | 2.17 | 6.82 | 0.0 |
| `bs_5` (1/240°) | 667.8 → 1009.6 | 90.27 | 3.59 | 19 | 54.39 | 13.29 | 106.24 | 2.00 | 5.57 | 0.0 |
| `bs_6` (2/0°) | 1438.2 → 1704.1 | 21.77 | 1.47 | 19 | 55.40 | 18.65 | 77.94 | 2.55 | 7.96 | 0.0 |
| `bs_7` (2/120°) | 1438.2 → 1704.1 | 87.50 | 1.47 | 19 | 55.40 | 16.91 | 124.35 | 2.21 | 6.12 | 6.1 |
| `bs_8` (2/240°) | 1438.2 → 1704.1 | 80.24 | 1.47 | 19 | 55.40 | 15.96 | 105.25 | 2.22 | 6.07 | 0.0 |
| `bs_9` (3/0°) | 580.3 → 842.8 | 28.42 | 3.24 | 19 | 53.85 | 10.53 | 58.37 | 1.31 | 3.24 | 0.0 |
| `bs_10` (3/120°) | 580.3 → 842.8 | 100.42 | 3.24 | 19 | 53.85 | 12.94 | 98.70 | 1.28 | 3.14 | 33.9 |
| `bs_11` (3/240°) | 580.3 → 842.8 | 79.03 | 3.24 | 19 | 53.85 | 11.92 | 82.58 | 1.29 | 3.14 | 0.0 |
| `bs_12` (4/0°) | 81.7 → 324.4 | 112.92 | 11.97 | 0 | 22.44 | 6.60 | 114.97 | 1.29 | 3.03 | 49.0 |
| `bs_13` (4/120°) | 81.7 → 324.4 | 97.80 | 11.97 | 0 | 22.44 | 6.87 | 118.94 | 1.29 | 3.13 | 12.0 |
| `bs_14` (4/240°) | 81.7 → 324.4 | 110.26 | 11.97 | 0 | 22.44 | 7.12 | 122.80 | 1.28 | 3.07 | 58.5 |
| `bs_15` (5/0°) | 1279.5 → 1473.8 | 6.58 | 1.23 | 19 | 60.00 | 17.13 | 67.81 | 2.59 | 7.09 | 0.0 |
| `bs_16` (5/120°) | 1279.5 → 1473.8 | 4.69 | 1.23 | 19 | 60.00 | 15.89 | 65.88 | 2.69 | 7.53 | 67.6 |
| `bs_17` (5/240°) | 1279.5 → 1473.8 | 9.45 | 1.23 | 19 | 60.00 | 16.54 | 67.96 | 2.71 | 8.08 | 0.0 |
| `bs_18` (6/0°) | 647.6 → 884.7 | 93.25 | 2.71 | 19 | 52.16 | 12.05 | 99.19 | 1.75 | 4.22 | 0.0 |
| `bs_19` (6/120°) | 647.6 → 884.7 | 72.19 | 2.71 | 19 | 52.16 | 16.17 | 82.58 | 1.78 | 4.80 | 19.9 |
| `bs_20` (6/240°) | 647.6 → 884.7 | 31.58 | 2.71 | 19 | 52.16 | 12.54 | 77.14 | 1.70 | 4.07 | 0.0 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_16` | 0.002 | 0.001 | 0.820 | 1.086 | 1.804 | 2.693 |
| `bs_14` | 0.262 | 0.006 | 0.000 | 0.460 | 1.056 | 1.283 |
| `bs_12` | 0.281 | 0.006 | 0.000 | 0.460 | 1.052 | 1.292 |
| `bs_3` | 0.055 | 0.001 | 0.513 | 0.817 | 1.276 | 1.934 |
| `bs_10` | 0.064 | 0.001 | 0.146 | 0.561 | 1.040 | 1.281 |
| `bs_19` | 0.085 | 0.001 | 0.412 | 0.738 | 1.237 | 1.776 |
| `bs_0` | 0.003 | 0.001 | 0.548 | 0.891 | 1.568 | 2.165 |
| `bs_13` | 0.286 | 0.006 | 0.000 | 0.460 | 1.056 | 1.293 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_16` | 0.004 | 0.001 | 0.000 | 2.636 | 5.097 | 7.531 |
| `bs_14` | 0.821 | 0.010 | 0.000 | 1.131 | 2.610 | 3.070 |
| `bs_12` | 0.873 | 0.010 | 0.000 | 1.131 | 2.562 | 3.030 |
| `bs_3` | 0.115 | 0.002 | 0.000 | 1.983 | 3.788 | 5.001 |
| `bs_10` | 0.121 | 0.003 | 0.000 | 1.294 | 2.786 | 3.141 |
| `bs_19` | 0.231 | 0.002 | 0.000 | 1.721 | 3.623 | 4.801 |
| `bs_0` | 0.007 | 0.001 | 0.000 | 2.176 | 4.423 | 5.892 |
| `bs_13` | 0.924 | 0.010 | 0.000 | 1.131 | 2.621 | 3.131 |

### Đánh giá route

- **Các candidate đo đạc:** 8 sector ở 6 site có mặt trong top 3 ít nhất 10% thời gian: `bs_16` 67.6%, `bs_14` 58.5%, `bs_12` 49.0%, `bs_3` 37.2%, `bs_10` 33.9%, `bs_19` 19.9%, `bs_0` 15.6%, `bs_13` 12.0%.
- **Sector mạnh nhất theo từng mẫu:** có 8 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_16` 41.6%, `bs_12` 24.2%, `bs_14` 22.3%, `bs_3` 8.0%, `bs_19` 2.0%, `bs_10` 1.3%, `bs_0` 0.5%, `bs_7` 0.1%. Identity của sector có RSRP lớn nhất đổi 587 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_14` (site 4, 240°) đi từ -190.51 đến -67.71 dBm, ΔRSRP = 122.80 dB. Các biên độ thành phần riêng: ΔGtx 110.26 dB, Δshadow 22.44 dB, ΔFSPL 11.97 dB, Δfast fading 7.12 dB, ΔLoS/NLoS 0.00 dB.
- **Hình học của candidate:** `bs_12` (site 4, 0°) có ΔFSPL lớn nhất trong nhóm candidate, 11.97 dB, khi khoảng cách 3D đổi từ 81.7 đến 324.4 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_12` (site 4, 0°) có ΔGtx = 112.92 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_16` (site 5, 120°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.69 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_16` (site 5, 120°), 7.53 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_7` (site 2, 120°) có ΔRSRP lớn nhất toàn bảng (124.35 dB) nhưng chỉ vào top 3 6.1% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

## 90 km/h · route 0

UAV đi khoảng **1500.0 m**. Biên độ RSRP lớn nhất thuộc `bs_19`: **125.77 dB**, xuất hiện trong top 3 **78.3%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_19` (**78.3%**), với biên độ RSRP **125.77 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **6/21 sector**. Trong nhóm đó, `bs_19` có biên độ RSRP lớn nhất (**125.77 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 1391.8 → 1712.4 | 5.66 | 1.80 | 19 | 50.87 | 11.65 | 66.91 | 1.96 | 4.70 | 11.0 |
| `bs_1` (0/120°) | 1391.8 → 1712.4 | 18.67 | 1.80 | 19 | 50.87 | 12.52 | 72.34 | 2.08 | 5.44 | 0.0 |
| `bs_2` (0/240°) | 1391.8 → 1712.4 | 7.48 | 1.80 | 19 | 50.87 | 14.35 | 64.12 | 2.12 | 5.66 | 0.0 |
| `bs_3` (1/0°) | 884.7 → 1273.8 | 75.02 | 3.17 | 19 | 54.70 | 14.02 | 89.26 | 1.76 | 4.66 | 47.0 |
| `bs_4` (1/120°) | 884.7 → 1273.8 | 77.85 | 3.17 | 19 | 54.70 | 13.02 | 88.60 | 1.69 | 4.36 | 0.0 |
| `bs_5` (1/240°) | 884.7 → 1273.8 | 72.15 | 3.17 | 19 | 54.70 | 10.27 | 86.16 | 1.67 | 4.24 | 0.0 |
| `bs_6` (2/0°) | 1489.2 → 1747.6 | 31.28 | 1.39 | 19 | 51.73 | 11.27 | 80.76 | 2.17 | 6.25 | 0.0 |
| `bs_7` (2/120°) | 1489.2 → 1747.6 | 7.22 | 1.39 | 19 | 51.73 | 13.30 | 64.01 | 2.30 | 6.67 | 9.8 |
| `bs_8` (2/240°) | 1489.2 → 1747.6 | 15.44 | 1.39 | 19 | 51.73 | 12.34 | 70.92 | 2.31 | 6.70 | 0.0 |
| `bs_9` (3/0°) | 678.1 → 927.5 | 29.28 | 2.72 | 19 | 59.59 | 10.30 | 69.30 | 1.54 | 3.49 | 6.5 |
| `bs_10` (3/120°) | 678.1 → 927.5 | 21.47 | 2.72 | 19 | 59.59 | 11.14 | 72.85 | 1.50 | 3.41 | 13.8 |
| `bs_11` (3/240°) | 678.1 → 927.5 | 15.63 | 2.72 | 19 | 59.59 | 12.06 | 65.41 | 1.59 | 3.68 | 0.0 |
| `bs_12` (4/0°) | 235.4 → 587.7 | 75.96 | 7.94 | 19 | 45.87 | 12.69 | 93.89 | 1.63 | 3.87 | 74.9 |
| `bs_13` (4/120°) | 235.4 → 587.7 | 72.68 | 7.94 | 19 | 45.87 | 11.70 | 94.74 | 1.62 | 3.84 | 1.4 |
| `bs_14` (4/240°) | 235.4 → 587.7 | 26.55 | 7.94 | 19 | 45.87 | 12.87 | 80.64 | 1.53 | 3.54 | 1.3 |
| `bs_15` (5/0°) | 1033.0 → 1316.1 | 25.75 | 2.10 | 19 | 53.82 | 12.94 | 66.59 | 1.98 | 5.32 | 0.0 |
| `bs_16` (5/120°) | 1033.0 → 1316.1 | 19.20 | 2.10 | 19 | 53.82 | 14.92 | 61.07 | 2.08 | 5.85 | 56.0 |
| `bs_17` (5/240°) | 1033.0 → 1316.1 | 14.90 | 2.10 | 19 | 53.82 | 12.22 | 56.32 | 1.96 | 5.37 | 0.0 |
| `bs_18` (6/0°) | 293.2 → 694.7 | 66.62 | 7.49 | 19 | 43.69 | 10.76 | 83.37 | 1.34 | 3.20 | 0.2 |
| `bs_19` (6/120°) | 293.2 → 694.7 | 115.48 | 7.49 | 19 | 43.69 | 15.17 | 125.77 | 1.38 | 3.16 | 78.3 |
| `bs_20` (6/240°) | 293.2 → 694.7 | 12.86 | 7.49 | 19 | 43.69 | 14.40 | 61.38 | 1.30 | 3.16 | 0.0 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_19` | 0.144 | 0.004 | 0.101 | 0.612 | 1.049 | 1.384 |
| `bs_12` | 0.109 | 0.003 | 0.323 | 0.718 | 1.147 | 1.628 |
| `bs_16` | 0.008 | 0.001 | 0.561 | 0.935 | 1.377 | 2.079 |
| `bs_3` | 0.049 | 0.001 | 0.421 | 0.799 | 1.148 | 1.757 |
| `bs_10` | 0.013 | 0.002 | 0.285 | 0.709 | 1.027 | 1.500 |
| `bs_0` | 0.002 | 0.001 | 0.393 | 0.922 | 1.393 | 1.955 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_19` | 0.281 | 0.005 | 0.000 | 1.454 | 2.675 | 3.160 |
| `bs_12` | 0.287 | 0.005 | 0.000 | 1.670 | 2.986 | 3.871 |
| `bs_16` | 0.019 | 0.002 | 0.000 | 2.404 | 4.217 | 5.851 |
| `bs_3` | 0.115 | 0.002 | 0.000 | 1.918 | 3.329 | 4.660 |
| `bs_10` | 0.025 | 0.002 | 0.000 | 1.641 | 2.685 | 3.410 |
| `bs_0` | 0.006 | 0.001 | 0.000 | 2.514 | 3.621 | 4.700 |

### Đánh giá route

- **Các candidate đo đạc:** 6 sector ở 6 site có mặt trong top 3 ít nhất 10% thời gian: `bs_19` 78.3%, `bs_12` 74.9%, `bs_16` 56.0%, `bs_3` 47.0%, `bs_10` 13.8%, `bs_0` 11.0%.
- **Sector mạnh nhất theo từng mẫu:** có 8 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_12` 34.7%, `bs_19` 27.9%, `bs_16` 20.2%, `bs_3` 15.0%, `bs_10` 1.6%, `bs_0` 0.5%, `bs_9` 0.1%, `bs_7` 0.0%. Identity của sector có RSRP lớn nhất đổi 629 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_19` (site 6, 120°) đi từ -195.50 đến -69.73 dBm, ΔRSRP = 125.77 dB. Các biên độ thành phần riêng: ΔGtx 115.48 dB, Δshadow 43.69 dB, ΔLoS/NLoS 19.00 dB, Δfast fading 15.17 dB, ΔFSPL 7.49 dB.
- **Hình học của candidate:** `bs_12` (site 4, 0°) có ΔFSPL lớn nhất trong nhóm candidate, 7.94 dB, khi khoảng cách 3D đổi từ 235.4 đến 587.7 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_19` (site 6, 120°) có ΔGtx = 115.48 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_16` (site 5, 120°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.08 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_16` (site 5, 120°), 5.85 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị cũng là candidate:** `bs_19` (site 6, 120°) vừa có ΔRSRP lớn nhất toàn bảng (125.77 dB), vừa nằm trong top 3 78.3% thời gian. Đây là sector cần ưu tiên khi phân tích quyết định HO của route này.

## 90 km/h · route 10

UAV đi khoảng **1500.0 m**. Biên độ RSRP lớn nhất thuộc `bs_13`: **107.36 dB**, xuất hiện trong top 3 **0.0%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_3` (**79.2%**), với biên độ RSRP **61.33 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **6/21 sector**. Trong nhóm đó, `bs_12` có biên độ RSRP lớn nhất (**107.08 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 1742.7 → 2123.0 | 8.41 | 1.71 | 19 | 60.00 | 12.11 | 62.98 | 2.43 | 6.16 | 7.9 |
| `bs_1` (0/120°) | 1742.7 → 2123.0 | 30.08 | 1.71 | 19 | 60.00 | 13.05 | 82.57 | 2.28 | 5.59 | 0.0 |
| `bs_2` (0/240°) | 1742.7 → 2123.0 | 16.06 | 1.71 | 19 | 60.00 | 11.47 | 67.07 | 2.40 | 6.01 | 0.0 |
| `bs_3` (1/0°) | 1372.1 → 1783.1 | 8.17 | 2.28 | 19 | 57.92 | 13.14 | 61.33 | 2.16 | 6.09 | 79.2 |
| `bs_4` (1/120°) | 1372.1 → 1783.1 | 11.76 | 2.28 | 19 | 57.92 | 11.64 | 61.69 | 2.14 | 5.56 | 0.0 |
| `bs_5` (1/240°) | 1372.1 → 1783.1 | 4.39 | 2.28 | 19 | 57.92 | 15.31 | 62.89 | 2.14 | 5.85 | 0.0 |
| `bs_6` (2/0°) | 1647.5 → 1915.9 | 12.99 | 1.31 | 19 | 59.92 | 16.81 | 64.75 | 2.67 | 6.84 | 2.4 |
| `bs_7` (2/120°) | 1647.5 → 1915.9 | 8.69 | 1.31 | 19 | 59.92 | 16.87 | 68.68 | 2.59 | 6.79 | 2.0 |
| `bs_8` (2/240°) | 1647.5 → 1915.9 | 2.98 | 1.31 | 19 | 59.92 | 13.37 | 66.66 | 2.48 | 6.40 | 0.0 |
| `bs_9` (3/0°) | 899.9 → 1262.2 | 71.06 | 2.94 | 19 | 52.36 | 11.55 | 79.42 | 2.16 | 5.64 | 2.4 |
| `bs_10` (3/120°) | 899.9 → 1262.2 | 72.76 | 2.94 | 19 | 52.36 | 14.01 | 86.13 | 2.31 | 6.35 | 0.0 |
| `bs_11` (3/240°) | 899.9 → 1262.2 | 68.00 | 2.94 | 19 | 52.36 | 11.78 | 77.80 | 2.16 | 5.63 | 0.0 |
| `bs_12` (4/0°) | 605.6 → 1022.5 | 75.54 | 4.55 | 19 | 51.48 | 16.08 | 107.08 | 2.17 | 6.15 | 39.9 |
| `bs_13` (4/120°) | 605.6 → 1022.5 | 72.76 | 4.55 | 19 | 51.48 | 12.83 | 107.36 | 2.16 | 5.83 | 0.0 |
| `bs_14` (4/240°) | 605.6 → 1022.5 | 79.29 | 4.55 | 19 | 51.48 | 10.22 | 105.91 | 2.03 | 5.55 | 0.0 |
| `bs_15` (5/0°) | 1102.1 → 1277.3 | 31.10 | 1.28 | 19 | 60.00 | 14.41 | 78.44 | 1.90 | 5.63 | 0.0 |
| `bs_16` (5/120°) | 1102.1 → 1277.3 | 83.83 | 1.28 | 19 | 60.00 | 13.09 | 97.76 | 1.87 | 5.22 | 13.8 |
| `bs_17` (5/240°) | 1102.1 → 1277.3 | 86.76 | 1.28 | 19 | 60.00 | 14.34 | 96.62 | 1.80 | 4.94 | 0.0 |
| `bs_18` (6/0°) | 245.2 → 434.0 | 44.08 | 4.96 | 19 | 54.75 | 8.74 | 70.27 | 1.26 | 2.89 | 59.2 |
| `bs_19` (6/120°) | 245.2 → 434.0 | 79.65 | 4.96 | 19 | 54.75 | 11.01 | 85.42 | 1.31 | 3.14 | 69.2 |
| `bs_20` (6/240°) | 245.2 → 434.0 | 76.38 | 4.96 | 19 | 54.75 | 12.23 | 85.36 | 1.35 | 3.18 | 24.2 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_3` | 0.003 | 0.001 | 0.694 | 1.025 | 1.340 | 2.160 |
| `bs_19` | 0.040 | 0.004 | 0.127 | 0.618 | 1.040 | 1.309 |
| `bs_18` | 0.017 | 0.004 | 0.127 | 0.618 | 1.005 | 1.264 |
| `bs_12` | 0.078 | 0.001 | 0.488 | 0.979 | 1.476 | 2.168 |
| `bs_20` | 0.029 | 0.004 | 0.127 | 0.618 | 1.084 | 1.349 |
| `bs_16` | 0.032 | 0.001 | 0.399 | 0.842 | 1.307 | 1.866 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_3` | 0.005 | 0.001 | 0.000 | 2.636 | 4.078 | 6.090 |
| `bs_19` | 0.075 | 0.006 | 0.000 | 1.442 | 2.548 | 3.140 |
| `bs_18` | 0.045 | 0.006 | 0.000 | 1.442 | 2.466 | 2.890 |
| `bs_12` | 0.229 | 0.003 | 0.000 | 2.489 | 4.716 | 6.151 |
| `bs_20` | 0.071 | 0.006 | 0.000 | 1.442 | 2.699 | 3.181 |
| `bs_16` | 0.058 | 0.002 | 0.000 | 2.167 | 3.986 | 5.222 |

### Đánh giá route

- **Các candidate đo đạc:** 6 sector ở 4 site có mặt trong top 3 ít nhất 10% thời gian: `bs_3` 79.2%, `bs_19` 69.2%, `bs_18` 59.2%, `bs_12` 39.9%, `bs_20` 24.2%, `bs_16` 13.8%.
- **Sector mạnh nhất theo từng mẫu:** có 7 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_3` 34.6%, `bs_19` 27.2%, `bs_18` 25.9%, `bs_12` 11.9%, `bs_16` 0.2%, `bs_0` 0.1%, `bs_7` 0.1%. Identity của sector có RSRP lớn nhất đổi 682 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_12` (site 4, 0°) đi từ -176.47 đến -69.39 dBm, ΔRSRP = 107.08 dB. Các biên độ thành phần riêng: ΔGtx 75.54 dB, Δshadow 51.48 dB, ΔLoS/NLoS 19.00 dB, Δfast fading 16.08 dB, ΔFSPL 4.55 dB.
- **Hình học của candidate:** `bs_18` (site 6, 0°) có ΔFSPL lớn nhất trong nhóm candidate, 4.96 dB, khi khoảng cách 3D đổi từ 245.2 đến 434.0 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_16` (site 5, 120°) có ΔGtx = 83.83 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_12` (site 4, 0°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.17 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_12` (site 4, 0°), 6.15 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_13` (site 4, 120°) có ΔRSRP lớn nhất toàn bảng (107.36 dB) nhưng chỉ vào top 3 0.0% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

## 120 km/h · route 0

UAV đi khoảng **2000.0 m**. Biên độ RSRP lớn nhất thuộc `bs_6`: **119.50 dB**, xuất hiện trong top 3 **0.9%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_7` (**77.5%**), với biên độ RSRP **86.82 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **5/21 sector**. Trong nhóm đó, `bs_0` có biên độ RSRP lớn nhất (**93.20 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 329.4 → 607.1 | 65.25 | 5.31 | 19 | 54.04 | 9.20 | 93.20 | 2.07 | 5.18 | 39.0 |
| `bs_1` (0/120°) | 329.4 → 607.1 | 74.53 | 5.31 | 19 | 54.04 | 8.45 | 100.50 | 2.08 | 5.34 | 0.3 |
| `bs_2` (0/240°) | 329.4 → 607.1 | 24.08 | 5.31 | 19 | 54.04 | 10.89 | 81.92 | 2.11 | 5.48 | 0.0 |
| `bs_3` (1/0°) | 1098.1 → 1346.2 | 17.36 | 1.77 | 19 | 56.54 | 7.84 | 70.78 | 1.60 | 4.09 | 0.0 |
| `bs_4` (1/120°) | 1098.1 → 1346.2 | 11.81 | 1.77 | 19 | 56.54 | 11.22 | 69.11 | 1.62 | 4.10 | 0.0 |
| `bs_5` (1/240°) | 1098.1 → 1346.2 | 7.74 | 1.77 | 19 | 56.54 | 11.90 | 66.69 | 1.65 | 4.19 | 4.3 |
| `bs_6` (2/0°) | 310.9 → 556.2 | 86.40 | 5.05 | 19 | 50.24 | 7.71 | 119.50 | 1.39 | 2.90 | 0.9 |
| `bs_7` (2/120°) | 310.9 → 556.2 | 74.08 | 5.05 | 19 | 50.24 | 6.86 | 86.82 | 1.42 | 3.04 | 77.5 |
| `bs_8` (2/240°) | 310.9 → 556.2 | 29.93 | 5.05 | 19 | 50.24 | 9.20 | 83.35 | 1.25 | 2.69 | 3.3 |
| `bs_9` (3/0°) | 704.9 → 927.5 | 16.71 | 2.38 | 19 | 44.67 | 11.15 | 68.89 | 1.44 | 3.05 | 0.0 |
| `bs_10` (3/120°) | 704.9 → 927.5 | 15.64 | 2.38 | 19 | 44.67 | 9.98 | 64.60 | 1.46 | 3.09 | 0.0 |
| `bs_11` (3/240°) | 704.9 → 927.5 | 15.40 | 2.38 | 19 | 44.67 | 7.81 | 66.26 | 1.42 | 2.98 | 76.5 |
| `bs_12` (4/0°) | 1531.1 → 1747.6 | 7.69 | 1.15 | 19 | 55.55 | 9.73 | 64.31 | 1.96 | 4.92 | 0.0 |
| `bs_13` (4/120°) | 1531.1 → 1747.6 | 2.42 | 1.15 | 19 | 55.55 | 11.71 | 65.00 | 2.04 | 5.33 | 0.0 |
| `bs_14` (4/240°) | 1531.1 → 1747.6 | 5.42 | 1.15 | 19 | 55.55 | 10.46 | 61.23 | 1.95 | 4.93 | 41.9 |
| `bs_15` (5/0°) | 1071.6 → 1267.9 | 12.34 | 1.46 | 19 | 51.60 | 10.89 | 63.18 | 1.68 | 4.46 | 0.0 |
| `bs_16` (5/120°) | 1071.6 → 1267.9 | 23.09 | 1.46 | 19 | 51.60 | 10.82 | 75.08 | 1.75 | 4.63 | 0.0 |
| `bs_17` (5/240°) | 1071.6 → 1267.9 | 8.74 | 1.46 | 19 | 51.60 | 16.01 | 61.86 | 1.85 | 5.00 | 0.9 |
| `bs_18` (6/0°) | 1498.5 → 1712.4 | 2.06 | 1.16 | 19 | 58.13 | 11.09 | 61.45 | 2.03 | 4.96 | 0.0 |
| `bs_19` (6/120°) | 1498.5 → 1712.4 | 9.61 | 1.16 | 19 | 58.13 | 13.14 | 59.50 | 1.93 | 4.94 | 0.0 |
| `bs_20` (6/240°) | 1498.5 → 1712.4 | 6.56 | 1.16 | 19 | 58.13 | 12.86 | 60.41 | 2.15 | 5.54 | 55.5 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_7` | 0.188 | 0.004 | 0.228 | 0.720 | 0.889 | 1.424 |
| `bs_11` | 0.015 | 0.003 | 0.285 | 0.773 | 0.907 | 1.419 |
| `bs_20` | 0.006 | 0.001 | 0.440 | 1.098 | 1.460 | 2.146 |
| `bs_14` | 0.004 | 0.001 | 0.551 | 1.105 | 1.074 | 1.954 |
| `bs_0` | 0.168 | 0.004 | 0.728 | 1.140 | 1.095 | 2.065 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_7` | 0.512 | 0.008 | 0.000 | 1.636 | 2.226 | 3.040 |
| `bs_11` | 0.037 | 0.004 | 0.000 | 1.750 | 2.254 | 2.981 |
| `bs_20` | 0.009 | 0.001 | 0.000 | 2.897 | 3.955 | 5.541 |
| `bs_14` | 0.012 | 0.002 | 0.000 | 2.917 | 3.173 | 4.931 |
| `bs_0` | 0.524 | 0.006 | 0.000 | 2.988 | 2.944 | 5.181 |

### Đánh giá route

- **Các candidate đo đạc:** 5 sector ở 5 site có mặt trong top 3 ít nhất 10% thời gian: `bs_7` 77.5%, `bs_11` 76.5%, `bs_20` 55.5%, `bs_14` 41.9%, `bs_0` 39.0%.
- **Sector mạnh nhất theo từng mẫu:** có 5 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_11` 45.5%, `bs_7` 23.7%, `bs_20` 17.3%, `bs_14` 7.5%, `bs_0` 6.0%. Identity của sector có RSRP lớn nhất đổi 785 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_0` (site 0, 0°) đi từ -172.25 đến -79.05 dBm, ΔRSRP = 93.20 dB. Các biên độ thành phần riêng: ΔGtx 65.25 dB, Δshadow 54.04 dB, ΔLoS/NLoS 19.00 dB, Δfast fading 9.20 dB, ΔFSPL 5.31 dB.
- **Hình học của candidate:** `bs_0` (site 0, 0°) có ΔFSPL lớn nhất trong nhóm candidate, 5.31 dB, khi khoảng cách 3D đổi từ 329.4 đến 607.1 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_7` (site 2, 120°) có ΔGtx = 74.08 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_20` (site 6, 240°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.15 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_20` (site 6, 240°), 5.54 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_6` (site 2, 0°) có ΔRSRP lớn nhất toàn bảng (119.50 dB) nhưng chỉ vào top 3 0.9% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

## 120 km/h · route 10

UAV đi khoảng **2000.0 m**. Biên độ RSRP lớn nhất thuộc `bs_13`: **123.42 dB**, xuất hiện trong top 3 **0.0%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_20` (**71.1%**), với biên độ RSRP **115.19 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **7/21 sector**. Trong nhóm đó, `bs_12` có biên độ RSRP lớn nhất (**123.04 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 1388.4 → 1708.0 | 69.21 | 1.80 | 19 | 49.78 | 15.43 | 78.45 | 1.79 | 4.47 | 3.7 |
| `bs_1` (0/120°) | 1388.4 → 1708.0 | 17.33 | 1.80 | 19 | 49.78 | 10.11 | 63.74 | 1.90 | 5.08 | 0.0 |
| `bs_2` (0/240°) | 1388.4 → 1708.0 | 73.99 | 1.80 | 19 | 49.78 | 10.21 | 79.68 | 1.91 | 4.88 | 0.0 |
| `bs_3` (1/0°) | 1252.2 → 1476.2 | 5.91 | 1.43 | 19 | 54.81 | 10.95 | 59.34 | 2.16 | 5.80 | 48.6 |
| `bs_4` (1/120°) | 1252.2 → 1476.2 | 7.65 | 1.43 | 19 | 54.81 | 11.19 | 61.96 | 2.18 | 6.11 | 0.0 |
| `bs_5` (1/240°) | 1252.2 → 1476.2 | 11.42 | 1.43 | 19 | 54.81 | 10.64 | 61.98 | 2.15 | 5.72 | 0.0 |
| `bs_6` (2/0°) | 1149.7 → 1507.1 | 9.98 | 2.35 | 19 | 41.39 | 10.16 | 57.24 | 1.33 | 2.92 | 0.2 |
| `bs_7` (2/120°) | 1149.7 → 1507.1 | 12.73 | 2.35 | 19 | 41.39 | 8.41 | 65.39 | 1.36 | 2.93 | 7.4 |
| `bs_8` (2/240°) | 1149.7 → 1507.1 | 8.19 | 2.35 | 19 | 41.39 | 7.42 | 61.50 | 1.36 | 2.90 | 0.0 |
| `bs_9` (3/0°) | 532.3 → 844.5 | 77.06 | 4.01 | 19 | 51.62 | 9.68 | 102.10 | 1.76 | 4.68 | 51.4 |
| `bs_10` (3/120°) | 532.3 → 844.5 | 26.20 | 4.01 | 19 | 51.62 | 9.54 | 72.85 | 1.75 | 4.38 | 0.0 |
| `bs_11` (3/240°) | 532.3 → 844.5 | 65.27 | 4.01 | 19 | 51.62 | 10.09 | 87.94 | 1.75 | 4.50 | 0.0 |
| `bs_12` (4/0°) | 644.4 → 992.5 | 72.03 | 3.75 | 19 | 52.09 | 8.16 | 123.04 | 2.00 | 4.97 | 12.9 |
| `bs_13` (4/120°) | 644.4 → 992.5 | 87.06 | 3.75 | 19 | 52.09 | 10.51 | 123.42 | 2.09 | 5.76 | 0.0 |
| `bs_14` (4/240°) | 644.4 → 992.5 | 73.64 | 3.75 | 19 | 52.09 | 9.21 | 122.33 | 2.05 | 5.17 | 0.0 |
| `bs_15` (5/0°) | 606.3 → 936.6 | 32.84 | 3.78 | 19 | 60.00 | 9.85 | 90.73 | 1.66 | 4.11 | 0.0 |
| `bs_16` (5/120°) | 606.3 → 936.6 | 33.35 | 3.78 | 19 | 60.00 | 9.80 | 93.77 | 1.68 | 4.24 | 56.5 |
| `bs_17` (5/240°) | 606.3 → 936.6 | 24.04 | 3.78 | 19 | 60.00 | 11.83 | 83.03 | 1.66 | 4.13 | 0.0 |
| `bs_18` (6/0°) | 96.2 → 373.0 | 77.24 | 11.78 | 19 | 55.41 | 10.07 | 92.52 | 1.43 | 3.21 | 13.6 |
| `bs_19` (6/120°) | 96.2 → 373.0 | 90.38 | 11.78 | 19 | 55.41 | 8.44 | 107.75 | 1.39 | 3.08 | 34.5 |
| `bs_20` (6/240°) | 96.2 → 373.0 | 98.56 | 11.78 | 19 | 55.41 | 9.27 | 115.19 | 1.44 | 3.24 | 71.1 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_20` | 0.215 | 0.008 | 0.196 | 0.732 | 0.906 | 1.439 |
| `bs_16` | 0.018 | 0.002 | 0.450 | 0.978 | 0.966 | 1.676 |
| `bs_9` | 0.051 | 0.003 | 0.500 | 0.964 | 1.100 | 1.759 |
| `bs_3` | 0.006 | 0.002 | 0.621 | 1.164 | 1.264 | 2.159 |
| `bs_19` | 0.199 | 0.008 | 0.196 | 0.732 | 0.852 | 1.391 |
| `bs_18` | 0.205 | 0.008 | 0.196 | 0.732 | 0.921 | 1.429 |
| `bs_12` | 0.082 | 0.003 | 0.621 | 1.086 | 1.139 | 2.001 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_20` | 0.568 | 0.017 | 0.000 | 1.698 | 2.329 | 3.241 |
| `bs_16` | 0.046 | 0.004 | 0.000 | 2.429 | 2.671 | 4.240 |
| `bs_9` | 0.117 | 0.005 | 0.000 | 2.496 | 3.077 | 4.683 |
| `bs_3` | 0.009 | 0.002 | 0.000 | 3.141 | 3.714 | 5.800 |
| `bs_19` | 0.570 | 0.017 | 0.000 | 1.698 | 2.140 | 3.081 |
| `bs_18` | 0.575 | 0.017 | 0.000 | 1.698 | 2.279 | 3.211 |
| `bs_12` | 0.231 | 0.003 | 0.000 | 2.860 | 3.037 | 4.970 |

### Đánh giá route

- **Các candidate đo đạc:** 7 sector ở 5 site có mặt trong top 3 ít nhất 10% thời gian: `bs_20` 71.1%, `bs_16` 56.5%, `bs_9` 51.4%, `bs_3` 48.6%, `bs_19` 34.5%, `bs_18` 13.6%, `bs_12` 12.9%.
- **Sector mạnh nhất theo từng mẫu:** có 8 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_20` 44.5%, `bs_3` 28.9%, `bs_19` 14.4%, `bs_16` 6.7%, `bs_12` 3.0%, `bs_9` 2.2%, `bs_7` 0.2%, `bs_0` 0.0%. Identity của sector có RSRP lớn nhất đổi 523 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_12` (site 4, 0°) đi từ -211.63 đến -88.59 dBm, ΔRSRP = 123.04 dB. Các biên độ thành phần riêng: ΔGtx 72.03 dB, Δshadow 52.09 dB, ΔLoS/NLoS 19.00 dB, Δfast fading 8.16 dB, ΔFSPL 3.75 dB.
- **Hình học của candidate:** `bs_18` (site 6, 0°) có ΔFSPL lớn nhất trong nhóm candidate, 11.78 dB, khi khoảng cách 3D đổi từ 96.2 đến 373.0 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_20` (site 6, 240°) có ΔGtx = 98.56 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_3` (site 1, 0°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.16 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_3` (site 1, 0°), 5.80 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_13` (site 4, 120°) có ΔRSRP lớn nhất toàn bảng (123.42 dB) nhưng chỉ vào top 3 0.0% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

## 150 km/h · route 0

UAV đi khoảng **2498.2 m**. Biên độ RSRP lớn nhất thuộc `bs_14`: **138.09 dB**, xuất hiện trong top 3 **0.1%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_19` (**80.9%**), với biên độ RSRP **88.02 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **7/21 sector**. Trong nhóm đó, `bs_16` có biên độ RSRP lớn nhất (**135.83 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 1385.9 → 2161.5 | 8.74 | 3.86 | 19 | 60.00 | 14.50 | 67.65 | 2.39 | 6.43 | 13.2 |
| `bs_1` (0/120°) | 1385.9 → 2161.5 | 33.57 | 3.86 | 19 | 60.00 | 10.16 | 86.31 | 2.25 | 5.57 | 0.0 |
| `bs_2` (0/240°) | 1385.9 → 2161.5 | 16.60 | 3.86 | 19 | 60.00 | 7.01 | 73.64 | 2.31 | 5.71 | 0.0 |
| `bs_3` (1/0°) | 1053.3 → 1796.8 | 16.22 | 4.64 | 19 | 58.32 | 11.44 | 71.65 | 2.07 | 5.37 | 73.2 |
| `bs_4` (1/120°) | 1053.3 → 1796.8 | 15.93 | 4.64 | 19 | 58.32 | 7.60 | 70.76 | 2.00 | 4.60 | 0.0 |
| `bs_5` (1/240°) | 1053.3 → 1796.8 | 19.54 | 4.64 | 19 | 58.32 | 10.52 | 73.48 | 2.16 | 6.00 | 0.0 |
| `bs_6` (2/0°) | 1369.9 → 1973.3 | 27.32 | 3.17 | 19 | 58.40 | 10.21 | 78.62 | 2.32 | 5.99 | 1.6 |
| `bs_7` (2/120°) | 1369.9 → 1973.3 | 9.17 | 3.17 | 19 | 58.40 | 12.36 | 65.79 | 2.48 | 6.81 | 6.5 |
| `bs_8` (2/240°) | 1369.9 → 1973.3 | 10.89 | 3.17 | 19 | 58.40 | 8.35 | 67.51 | 2.32 | 5.97 | 0.0 |
| `bs_9` (3/0°) | 561.4 → 1303.9 | 71.48 | 7.32 | 19 | 60.00 | 8.30 | 79.51 | 2.28 | 5.99 | 9.1 |
| `bs_10` (3/120°) | 561.4 → 1303.9 | 80.39 | 7.32 | 19 | 60.00 | 8.58 | 111.98 | 2.38 | 6.24 | 1.0 |
| `bs_11` (3/240°) | 561.4 → 1303.9 | 74.17 | 7.32 | 19 | 60.00 | 11.21 | 88.98 | 2.41 | 6.74 | 0.0 |
| `bs_12` (4/0°) | 419.0 → 1022.5 | 78.74 | 7.75 | 19 | 60.00 | 8.59 | 113.83 | 1.94 | 5.05 | 56.7 |
| `bs_13` (4/120°) | 419.0 → 1022.5 | 92.19 | 7.75 | 19 | 60.00 | 9.30 | 136.57 | 1.88 | 4.65 | 1.0 |
| `bs_14` (4/240°) | 419.0 → 1022.5 | 97.70 | 7.75 | 19 | 60.00 | 11.10 | 138.09 | 1.97 | 5.16 | 0.1 |
| `bs_15` (5/0°) | 936.7 → 1293.0 | 81.15 | 2.80 | 19 | 55.30 | 8.55 | 79.08 | 2.31 | 6.23 | 0.0 |
| `bs_16` (5/120°) | 936.7 → 1293.0 | 90.36 | 2.80 | 19 | 55.30 | 7.93 | 135.83 | 2.33 | 6.43 | 13.2 |
| `bs_17` (5/240°) | 936.7 → 1293.0 | 85.94 | 2.80 | 19 | 55.30 | 11.74 | 108.60 | 2.31 | 6.41 | 0.0 |
| `bs_18` (6/0°) | 263.2 → 489.2 | 82.10 | 5.38 | 19 | 56.23 | 9.82 | 96.78 | 1.57 | 3.19 | 32.4 |
| `bs_19` (6/120°) | 263.2 → 489.2 | 69.73 | 5.38 | 19 | 56.23 | 6.79 | 88.02 | 1.61 | 3.40 | 80.9 |
| `bs_20` (6/240°) | 263.2 → 489.2 | 68.66 | 5.38 | 19 | 56.23 | 9.62 | 86.07 | 1.62 | 3.43 | 10.9 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_19` | 0.099 | 0.006 | 0.355 | 0.910 | 0.863 | 1.606 |
| `bs_3` | 0.007 | 0.001 | 0.779 | 1.232 | 0.941 | 2.074 |
| `bs_12` | 0.095 | 0.003 | 0.640 | 1.115 | 0.991 | 1.939 |
| `bs_18` | 0.049 | 0.006 | 0.355 | 0.910 | 0.864 | 1.566 |
| `bs_16` | 0.113 | 0.002 | 0.896 | 1.349 | 1.087 | 2.327 |
| `bs_0` | 0.007 | 0.001 | 0.804 | 1.395 | 1.224 | 2.391 |
| `bs_20` | 0.072 | 0.006 | 0.355 | 0.910 | 0.916 | 1.623 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_19` | 0.285 | 0.010 | 0.000 | 2.050 | 2.161 | 3.400 |
| `bs_3` | 0.017 | 0.003 | 0.000 | 3.327 | 2.835 | 5.373 |
| `bs_12` | 0.256 | 0.007 | 0.000 | 2.898 | 2.718 | 5.051 |
| `bs_18` | 0.106 | 0.010 | 0.000 | 2.050 | 2.175 | 3.190 |
| `bs_16` | 0.289 | 0.004 | 0.000 | 3.673 | 3.036 | 6.431 |
| `bs_0` | 0.018 | 0.002 | 0.000 | 3.814 | 3.661 | 6.433 |
| `bs_20` | 0.172 | 0.010 | 0.000 | 2.050 | 2.318 | 3.432 |

### Đánh giá route

- **Các candidate đo đạc:** 7 sector ở 5 site có mặt trong top 3 ít nhất 10% thời gian: `bs_19` 80.9%, `bs_3` 73.2%, `bs_12` 56.7%, `bs_18` 32.4%, `bs_16` 13.2%, `bs_0` 13.2%, `bs_20` 10.9%.
- **Sector mạnh nhất theo từng mẫu:** có 8 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_19` 33.3%, `bs_3` 33.2%, `bs_12` 23.4%, `bs_18` 7.8%, `bs_16` 1.7%, `bs_9` 0.3%, `bs_10` 0.1%, `bs_0` 0.1%. Identity của sector có RSRP lớn nhất đổi 568 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_16` (site 5, 120°) đi từ -222.99 đến -87.16 dBm, ΔRSRP = 135.83 dB. Các biên độ thành phần riêng: ΔGtx 90.36 dB, Δshadow 55.30 dB, ΔLoS/NLoS 19.00 dB, Δfast fading 7.93 dB, ΔFSPL 2.80 dB.
- **Hình học của candidate:** `bs_12` (site 4, 0°) có ΔFSPL lớn nhất trong nhóm candidate, 7.75 dB, khi khoảng cách 3D đổi từ 419.0 đến 1022.5 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_16` (site 5, 120°) có ΔGtx = 90.36 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_0` (site 0, 0°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.39 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_0` (site 0, 0°), 6.43 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_14` (site 4, 240°) có ΔRSRP lớn nhất toàn bảng (138.09 dB) nhưng chỉ vào top 3 0.1% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

## 150 km/h · route 10

UAV đi khoảng **2500.0 m**. Biên độ RSRP lớn nhất thuộc `bs_8`: **118.34 dB**, xuất hiện trong top 3 **0.1%** số bước. Sector xuất hiện trong top 3 nhiều nhất là `bs_17` (**79.5%**), với biên độ RSRP **114.54 dB**.

Nếu chỉ đánh dấu các sector nằm trong top 3 ít nhất 10% thời gian, route này có **7/21 sector**. Trong nhóm đó, `bs_17` có biên độ RSRP lớn nhất (**114.54 dB**). Ngưỡng 10% chỉ giúp đọc bảng; mọi Δ trong bảng vẫn tính trên toàn route.

| BS (site/hướng) | d3 gần → xa (m) | ΔGtx | ΔFSPL | ΔLoS | ΔShadow | ΔFast | ΔRSRP | mean abs(δRSRP) | P95 abs(δRSRP) | Top 3 % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bs_0` (0/0°) | 1052.7 → 1366.9 | 12.46 | 2.27 | 19 | 54.95 | 9.88 | 69.82 | 1.99 | 5.28 | 57.2 |
| `bs_1` (0/120°) | 1052.7 → 1366.9 | 17.84 | 2.27 | 19 | 54.95 | 11.57 | 68.40 | 2.04 | 5.81 | 0.0 |
| `bs_2` (0/240°) | 1052.7 → 1366.9 | 12.87 | 2.27 | 19 | 54.95 | 10.48 | 71.73 | 1.86 | 4.51 | 0.0 |
| `bs_3` (1/0°) | 1302.2 → 1722.1 | 9.12 | 2.43 | 19 | 54.23 | 11.10 | 60.94 | 2.08 | 4.92 | 9.9 |
| `bs_4` (1/120°) | 1302.2 → 1722.1 | 16.49 | 2.43 | 19 | 54.23 | 9.56 | 65.32 | 2.36 | 6.36 | 0.0 |
| `bs_5` (1/240°) | 1302.2 → 1722.1 | 28.71 | 2.43 | 19 | 54.23 | 9.18 | 72.86 | 2.25 | 6.05 | 0.0 |
| `bs_6` (2/0°) | 428.0 → 690.1 | 72.53 | 4.15 | 19 | 53.37 | 12.38 | 106.13 | 1.65 | 3.64 | 59.7 |
| `bs_7` (2/120°) | 428.0 → 690.1 | 42.41 | 4.15 | 19 | 53.37 | 7.28 | 88.55 | 1.63 | 3.71 | 0.3 |
| `bs_8` (2/240°) | 428.0 → 690.1 | 88.51 | 4.15 | 19 | 53.37 | 10.28 | 118.34 | 1.72 | 3.58 | 0.1 |
| `bs_9` (3/0°) | 454.8 → 890.9 | 20.79 | 5.84 | 19 | 53.37 | 7.69 | 80.98 | 2.01 | 4.71 | 25.5 |
| `bs_10` (3/120°) | 454.8 → 890.9 | 17.20 | 5.84 | 19 | 53.37 | 11.28 | 81.02 | 2.16 | 5.74 | 0.7 |
| `bs_11` (3/240°) | 454.8 → 890.9 | 29.22 | 5.84 | 19 | 53.37 | 7.26 | 87.39 | 2.08 | 5.19 | 1.6 |
| `bs_12` (4/0°) | 1238.7 → 1677.5 | 16.73 | 2.63 | 19 | 60.00 | 11.37 | 76.05 | 1.88 | 4.98 | 0.0 |
| `bs_13` (4/120°) | 1238.7 → 1677.5 | 9.37 | 2.63 | 19 | 60.00 | 8.11 | 67.59 | 1.73 | 4.38 | 0.0 |
| `bs_14` (4/240°) | 1238.7 → 1677.5 | 6.44 | 2.63 | 19 | 60.00 | 9.55 | 66.76 | 1.83 | 4.69 | 15.4 |
| `bs_15` (5/0°) | 195.5 → 467.4 | 87.87 | 7.57 | 19 | 48.76 | 8.50 | 102.96 | 1.35 | 2.98 | 1.9 |
| `bs_16` (5/120°) | 195.5 → 467.4 | 86.00 | 7.57 | 19 | 48.76 | 7.11 | 99.08 | 1.34 | 2.95 | 13.2 |
| `bs_17` (5/240°) | 195.5 → 467.4 | 97.71 | 7.57 | 19 | 48.76 | 7.16 | 114.54 | 1.39 | 3.06 | 79.5 |
| `bs_18` (6/0°) | 846.0 → 1220.2 | 86.71 | 3.18 | 19 | 53.43 | 8.90 | 100.57 | 1.95 | 5.19 | 0.0 |
| `bs_19` (6/120°) | 846.0 → 1220.2 | 77.99 | 3.18 | 19 | 53.43 | 10.37 | 108.61 | 1.98 | 5.51 | 0.0 |
| `bs_20` (6/240°) | 846.0 → 1220.2 | 82.61 | 3.18 | 19 | 53.43 | 11.06 | 100.62 | 1.95 | 5.28 | 34.9 |

### Biến thiên từng thành phần giữa timestep 10 ms của candidate

Mỗi giá trị là `abs(x[t] − x[t−1])`, đơn vị dB/10 ms. Bảng đầu cho mức thay đổi điển hình; bảng sau cho bước nhảy P95. P99 và max của mọi thành phần vẫn có trong CSV chi tiết.

| BS | mean abs(δGtx) | mean abs(δFSPL) | mean abs(δLoS) | mean abs(δShadow) | mean abs(δFast) | mean abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_17` | 0.158 | 0.006 | 0.266 | 0.807 | 0.831 | 1.392 |
| `bs_6` | 0.128 | 0.005 | 0.513 | 1.019 | 0.786 | 1.648 |
| `bs_0` | 0.008 | 0.001 | 0.738 | 1.167 | 0.967 | 1.987 |
| `bs_20` | 0.108 | 0.002 | 0.709 | 1.085 | 0.929 | 1.953 |
| `bs_9` | 0.018 | 0.004 | 0.681 | 1.188 | 1.013 | 2.013 |
| `bs_14` | 0.005 | 0.002 | 0.643 | 1.117 | 0.924 | 1.826 |
| `bs_16` | 0.063 | 0.006 | 0.266 | 0.807 | 0.813 | 1.338 |

| BS | P95 abs(δGtx) | P95 abs(δFSPL) | P95 abs(δLoS) | P95 abs(δShadow) | P95 abs(δFast) | P95 abs(δRSRP) |
|---|---:|---:|---:|---:|---:|---:|
| `bs_17` | 0.500 | 0.013 | 0.000 | 1.822 | 2.036 | 3.061 |
| `bs_6` | 0.391 | 0.007 | 0.000 | 2.616 | 2.122 | 3.640 |
| `bs_0` | 0.024 | 0.003 | 0.000 | 3.173 | 2.709 | 5.282 |
| `bs_20` | 0.277 | 0.004 | 0.000 | 2.946 | 2.361 | 5.280 |
| `bs_9` | 0.034 | 0.005 | 0.000 | 3.338 | 2.695 | 4.711 |
| `bs_14` | 0.009 | 0.003 | 0.000 | 2.838 | 2.588 | 4.690 |
| `bs_16` | 0.109 | 0.013 | 0.000 | 1.822 | 2.111 | 2.951 |

### Đánh giá route

- **Các candidate đo đạc:** 7 sector ở 6 site có mặt trong top 3 ít nhất 10% thời gian: `bs_17` 79.5%, `bs_6` 59.7%, `bs_0` 57.2%, `bs_20` 34.9%, `bs_9` 25.5%, `bs_14` 15.4%, `bs_16` 13.2%.
- **Sector mạnh nhất theo từng mẫu:** có 8 sector từng đứng thứ nhất; tỉ lệ đứng thứ nhất là `bs_17` 46.9%, `bs_0` 19.1%, `bs_6` 18.1%, `bs_9` 6.5%, `bs_16` 4.5%, `bs_20` 4.0%, `bs_3` 0.7%, `bs_14` 0.3%. Identity của sector có RSRP lớn nhất đổi 658 lần trong 5.999 chuyển tiếp mẫu. Đây là thay đổi của phép đo RSRP, không phải số handover mà Agent phải thực hiện.
- **Candidate có RSRP biến thiên lớn nhất:** `bs_17` (site 5, 240°) đi từ -177.43 đến -62.89 dBm, ΔRSRP = 114.54 dB. Các biên độ thành phần riêng: ΔGtx 97.71 dB, Δshadow 48.76 dB, ΔLoS/NLoS 19.00 dB, ΔFSPL 7.57 dB, Δfast fading 7.16 dB.
- **Hình học của candidate:** `bs_16` (site 5, 120°) có ΔFSPL lớn nhất trong nhóm candidate, 7.57 dB, khi khoảng cách 3D đổi từ 195.5 đến 467.4 m. Đây là phần biến thiên xác định trực tiếp từ chuyển động; Gtx/shadow/fading còn phụ thuộc hướng hoặc trạng thái kênh.
- **Thành phần dao động riêng lớn nhất trong nhóm candidate:** `bs_17` (site 5, 240°) có ΔGtx = 97.71 dB. Không cộng các Δ trong bảng để suy ra ΔRSRP, vì cực đại và cực tiểu của các thành phần thường nằm ở các thời điểm khác nhau.
- **Biến thiên giữa timestep:** `bs_9` (site 3, 0°) có trung bình |δRSRP| lớn nhất trong nhóm candidate, 2.01 dB/10 ms. Bước nhảy P95 lớn nhất thuộc `bs_0` (site 0, 0°), 5.28 dB/10 ms. Các số này đo tín hiệu thay đổi cục bộ giữa hai mẫu liền kề, khác với ΔRSRP là biên độ cả route.
- **Cực trị ngoài vùng cạnh tranh:** `bs_8` (site 2, 240°) có ΔRSRP lớn nhất toàn bảng (118.34 dB) nhưng chỉ vào top 3 0.1% thời gian. Nó hữu ích để kiểm tra mô hình kênh, nhưng không nên được dùng một mình để đánh giá khả năng chọn target HO của Agent.

