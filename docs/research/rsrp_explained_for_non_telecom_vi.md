# Hiểu RSRP trong Data Generator UAV

## 1. Mục đích tài liệu

Tài liệu này giải thích cách **RSRP** được tạo trong data generator của dự án
HandoverOptimDRL, dành cho người không chuyên về viễn thông. Mục tiêu không
phải là học toàn bộ tiêu chuẩn 3GPP, mà là hiểu được:

- RSRP là gì và khác SINR ở đâu;
- vì sao RSRP tăng hoặc giảm khi UAV bay;
- mỗi yếu tố trong mô hình có vai trò gì;
- yếu tố nào thay đổi chậm, thay đổi nhanh, hoặc gây bước nhảy lớn;
- vì sao những yếu tố này quan trọng đối với bài toán handover và PPO.

## 2. Hình dung đơn giản: nghe một người nói giữa thành phố

Hãy xem BS là một người đang nói bằng loa, còn UAV là người nghe khi đang di
chuyển trong thành phố.

- **PTX** là âm lượng loa phát.
- **Antenna gain** là loa đang hướng thẳng vào người nghe hay quay lệch đi.
- **FSPL** là âm thanh yếu dần chỉ vì người nghe đi xa.
- **LoS/NLoS** là giữa hai người có nhìn thẳng thấy nhau hay có tòa nhà chắn.
- **Shadow fading** là người nghe đi qua từng khu vực bị chắn mạnh/yếu khác nhau.
- **Fast fading** là tiếng dội từ tường, nhà và mặt đất đôi khi cộng nhau, đôi
  khi triệt nhau.

RSRP là mức công suất tín hiệu tham chiếu mà UAV đo được từ từng sector. Nó trả
lời câu hỏi: **“Ở vị trí này, tín hiệu của sector A/B/C mạnh đến mức nào?”**

## 3. RSRP là gì?

RSRP là viết tắt của *Reference Signal Received Power*: công suất của tín hiệu
tham chiếu mà thiết bị thu được từ một cell/sector. Đơn vị là **dBm**.

dBm là thang logarithm. Giá trị càng gần 0 thì tín hiệu càng mạnh:

| RSRP | Cách hiểu gần đúng |
|---:|---|
| −70 dBm | Mạnh |
| −90 dBm | Khá tốt/trung bình |
| −110 dBm | Yếu |
| −130 dBm | Rất yếu |

Các ngưỡng trên chỉ để trực giác; chất lượng sử dụng thực tế còn phụ thuộc vào
nhiễu, bandwidth, dịch vụ và cấu hình mạng.

Trong generator, trước hết code tính công suất thu toàn băng `Pr`, rồi quy đổi
về RSRP:

```text
RSRP = Pr − 10log10(51 × 12) + reference-signal offset
```

Ở cấu hình hiện tại, offset bằng 0 dB. Phép trừ này là bước quy đổi từ công suất
trên toàn channel sang công suất quy về một resource element tham chiếu. Nó là
hằng số trong một dataset, nên **không làm RSRP rung giữa các timestep**.

## 4. RSRP khác SINR thế nào?

RSRP chỉ nhìn mức tín hiệu của một sector. Nó không hỏi các sector khác đang
gây nhiễu mạnh hay yếu.

SINR trả lời câu hỏi khó hơn: **“Tín hiệu mình muốn nghe có mạnh hơn nhiễu và
noise không?”**

```text
SINR = signal / (interference + noise)
```

Ví dụ:

- Cell A có RSRP cao, nhưng các cell khác cũng phát mạnh trên cùng channel:
  SINR vẫn có thể thấp.
- Cell A có RSRP không quá cao, nhưng ít cell khác gây nhiễu: SINR có thể tốt.

RSRP thường hữu ích để nhận biết cell mạnh/yếu và kích hoạt handover; SINR hữu
ích để đánh giá chất lượng liên kết sau khi đã tính nhiễu.

## 5. Công thức trực giác của RSRP trong generator

Mỗi sector `i` tại thời điểm `t` có công suất thu:

```text
Pr_i(t) = PTX_i + Gtx_i(t) + GRX
          − FSPL_i(t) − LoS/NLoS loss_i(t)
          − Shadow_i(t) + FastFading_i(t)
```

Sau đó `Pr_i` được đổi sang `RSRP_i`. Dấu `+` nghĩa là giúp tín hiệu mạnh hơn;
dấu `−` nghĩa là tạo suy hao.

| Thành phần | Vai trò đơn giản |
|---|---|
| `PTX` | Sector phát mạnh đến đâu. |
| `Gtx` | Antenna BS đang chiếu tín hiệu tốt đến hướng UAV hay không. |
| `GRX` | Gain anten thu của UAV. |
| FSPL | Suy hao do khoảng cách. |
| LoS/NLoS loss | Suy hao tăng thêm vì có/không có đường nhìn thẳng. |
| Shadow fading | Che khuất thay đổi chậm theo khu vực. |
| Fast fading | Dao động nhanh do sóng phản xạ cộng/triệt nhau. |

## 6. Những thành phần cố định

Các giá trị sau không đổi trong một lần sinh route. Chúng quyết định “mặt bằng”
của dataset nhưng không tạo dao động theo từng sample:

| Thành phần hiện tại | Giá trị | Vai trò |
|---|---:|---|
| `PTX` | 35 dBm | Công suất phát toàn băng của mỗi sector. |
| `GRX` | 2 dB | Gain thu UAV. |
| Carrier | 2.1 GHz | Ảnh hưởng path loss và Doppler. |
| Bandwidth | 10 MHz | Dùng chủ yếu khi tính noise/SINR. |
| BS height | 25 m | Một phần của khoảng cách 3D và góc antenna. |
| UAV altitude | 100 m mặc định | Có thể override theo route. |
| RB normalization | 51 × 12 | Hằng số quy đổi `Pr` sang RSRP. |

Nếu tăng `PTX` 3 dB, RSRP của sector đó tăng xấp xỉ 3 dB. Tuy nhiên, trong mô
hình đồng kênh, sector cũng gây nhiễu mạnh hơn cho các cell khác; SINR không
nhất thiết tăng tương ứng ở mọi nơi.

## 7. Antenna 3D: yếu tố quyết định sector nào mạnh

Mỗi site có ba sector hướng về các bearing 0°, 120° và 240°. Các sector dùng
panel 3D 8×4 với electrical downtilt 10°.

Antenna không phát đều mọi hướng. Gain `Gtx` được tính từ vị trí UAV so với
sector:

```text
Vị trí UAV và BS
→ khoảng cách + azimuth + elevation
→ đổi sang hệ local của sector theo bearing/tilt
→ element pattern + array factor 8×4
→ Gtx theo hướng UAV
```

Nếu UAV nằm trong main lobe của sector, `Gtx` cao. Nếu nằm lệch hướng, nằm trong
sidelobe hoặc phía sau panel, `Gtx` thấp. Vì vậy, ba sector cùng một site có thể
có RSRP rất khác nhau dù khoảng cách tới UAV bằng nhau.

`Gtx` thường thay đổi **mượt giữa hai timestep**, vì UAV chỉ dịch chuyển ngắn
mỗi 10 ms. Tuy vậy, trên toàn route nó có thể thay đổi hàng chục dB khi UAV đi
từ vùng main lobe sang vùng sidelobe hoặc phía sau sector.

## 8. FSPL: suy hao chỉ vì khoảng cách

FSPL là *Free-Space Path Loss*, tức suy hao khi sóng truyền trong không gian
trống. Nó dùng khoảng cách 3D giữa UAV và BS:

```text
d3 = sqrt((xUAV − xBS)^2 + (yUAV − yBS)^2 + (hUAV − hBS)^2)

FSPL = 32.4 + 20log10(fc_GHz) + 20log10(d3_m)
```

Khi khoảng cách tăng gấp đôi, FSPL tăng khoảng 6 dB, khiến RSRP giảm khoảng
6 dB trước khi xét mọi hiệu ứng khác.

FSPL tạo xu hướng dài hạn của route, nhưng gần như không gây nhấp nháy từng
sample. Ở tốc độ 240 km/h, UAV chỉ đi khoảng 0.667 m mỗi 10 ms; với chênh cao
UAV–BS tối thiểu 75 m, thay đổi FSPL cực đại trong một step chỉ khoảng 0.08 dB.

## 9. LoS và NLoS: có đường nhìn thẳng hay bị chắn

**LoS** (*Line of Sight*) nghĩa là có đường nhìn thẳng giữa UAV và BS.
**NLoS** (*Non-Line of Sight*) nghĩa là đường truyền bị che bởi môi trường.

Generator không dùng bản đồ nhà thật; nó dùng mô hình thống kê urban dựa trên:

- tỷ lệ diện tích xây dựng;
- mật độ building;
- phân bố chiều cao building;
- khoảng cách và altitude của UAV.

Từ đó nó tính xác suất `P(LoS)`, rồi tạo một state có tương quan không gian.
State này không được tung đồng xu mới hoàn toàn ở từng timestep.

| State | Loss cộng thêm vào FSPL |
|---|---:|
| LoS | 1 dB |
| NLoS | 20 dB |

Do đó chuyển LoS → NLoS có thể làm RSRP giảm gần 19 dB ngay lập tức. Đây là
một trong các sự kiện có ảnh hưởng lớn nhất trong mô hình.

Hiện `LOS_STATE_CORRELATION_DISTANCE_M = 50 m`: UAV đi ít thì state thường giữ
nguyên; UAV đi xa thì khả năng state đổi cao hơn. Ba sector cùng site dùng chung
LoS/NLoS state vì cùng nhìn qua gần như cùng đường che khuất đến vị trí site.

## 10. Shadow fading: che khuất thay đổi chậm

Shadow fading là suy hao do cụm vật cản lớn quanh khu vực UAV đang đi qua, ví
dụ một dãy nhà hoặc vùng đô thị dày hơn. Đây không phải suy hao do “xa”, mà là
độ chênh so với mức path loss trung bình tại cùng khoảng cách.

| State | Standard deviation shadow |
|---|---:|
| LoS | 4.6 dB |
| NLoS | 10 dB |

Nếu `Shadow = +6 dB`, RSRP giảm thêm 6 dB. Nếu `Shadow = −3 dB`, RSRP tăng
thêm 3 dB so với mức trung bình.

Shadow được cập nhật có tương quan theo quãng đường. Với P0 canonical:

```text
shadow correlation distance = max(20 m, 0.25 × 100 m) = 25 m
```

Vì vậy shadow thường thay đổi chậm theo hàng chục mét. Trong generator, ba
sector cùng site cũng dùng chung shadow state.

## 11. Fast fading: dao động nhanh do phản xạ

Fast fading là dao động nhanh do UAV nhận nhiều bản sao của cùng tín hiệu qua
đường trực tiếp và các đường phản xạ từ nhà, mặt đất hoặc vật thể khác.

- Sóng cùng pha: cộng tăng cường, RSRP tăng.
- Sóng ngược pha: triệt nhau, RSRP giảm.

LoS dùng Rician fading: có thành phần đường trực tiếp mạnh cùng các sóng phản
xạ. NLoS dùng Rayleigh fading: chủ yếu là các sóng phản xạ nên dễ deep fade hơn.

Generator dùng 16 sóng diffuse, cập nhật theo Doppler do vận tốc/hướng bay và
carrier 2.1 GHz. Nó tính fading bên trong mỗi 1 ms, sau đó trung bình power
trong một measurement 10 ms. Điều này làm RSRP vẫn biến động nhanh, nhưng không
quá nhấp nháy chỉ vì một pha tức thời.

Fast fading thường là nguyên nhân chính làm RSRP thay đổi giữa hai sample liên
tiếp. Tốc độ UAV cao hơn hoặc carrier cao hơn làm Doppler lớn hơn, vì vậy fading
thay đổi nhanh hơn.

## 12. Thành phần nào quan trọng ở thang thời gian nào?

| Thang thời gian | Yếu tố thường quan trọng nhất | Dấu hiệu quan sát |
|---|---|---|
| Mỗi 10 ms | Fast fading | RSRP rung vài dB dù UAV gần như chưa đi xa. |
| Vài mét đến vài chục mét | Shadow fading; thay đổi gain antenna | RSRP dịch chuyển dần theo đoạn đường. |
| Sự kiện hiếm nhưng mạnh | LoS ↔ NLoS | Có thể nhảy gần 19 dB. |
| Toàn route | FSPL và antenna 3D | Quyết định site/sector nào mạnh ở từng vùng. |
| Toàn dataset | PTX, GRX, carrier, geometry, profile urban | Quyết định mặt bằng và độ khó chung của dữ liệu. |

## 13. Vì sao RSRP quan trọng cho handover?

Handover là quá trình chuyển kết nối từ serving cell hiện tại sang cell khác.
Nếu RSRP/quality của cell hiện tại giảm và cell khác tốt hơn đủ lâu, logic
handover có thể chuyển cell.

Data generator không thực hiện quyết định này. Nó chỉ tạo “thế giới radio” để
handover environment quan sát. Vì vậy:

- antenna 3D tạo ra vùng mạnh/yếu khác nhau giữa sector;
- FSPL tạo xu hướng khi UAV tiến gần/rời xa site;
- LoS/NLoS và shadow tạo các thay đổi môi trường;
- fast fading tạo nhiễu nhanh mà policy không nên phản ứng thái quá.

Một policy tốt cần phân biệt giữa một xu hướng suy giảm thật sự và một fade ngắn
hạn. Nếu phản ứng với mọi dao động fast fading, policy có thể handover quá nhiều;
nếu phản ứng quá chậm với LoS → NLoS hoặc rời main lobe, policy có thể giữ cell
yếu quá lâu.

## 14. Cách đọc kết quả audit P0

Tool `audit-rsrp-factors` tái lập route từ seed và manifest, sau đó báo cáo năm
thành phần:

| Tên audit | Ý nghĩa |
|---|---|
| `antenna_gain_db` | `Gtx`: hướng/gain antenna 3D. |
| `fspl_db` | Suy hao do khoảng cách. |
| `excess_loss_db` | 1 dB LoS hoặc 20 dB NLoS. |
| `shadow_loss_db` | Che khuất chậm theo không gian. |
| `fast_fading_gain_db` | Multipath/Doppler nhanh. |

Ba metric chính:

- **Route range:** biên độ biến thiên của một thành phần trên cả route.
- **Mean absolute delta per step:** mức thay đổi trung bình mỗi 10 ms.
- **Dominant per-step share:** tỷ lệ thời điểm thành phần đó là nguyên nhân có
  độ thay đổi lớn nhất.

Không nên kết luận chỉ từ route range. Một antenna gain có range lớn có thể do
sector rất yếu ở xa hoặc gần null, nhưng không có nghĩa nó là nguyên nhân làm
RSRP rung liên tục. Cần đọc cả ba metric.

## 15. Tóm tắt

RSRP trong generator không đến từ một nguyên nhân duy nhất. Nó là tổng hợp của:

```text
Công suất phát
+ hướng antenna
− khoảng cách
− che khuất LoS/NLoS
− shadow fading
+/− fast fading
```

Nếu chỉ nhớ bốn ý:

1. **Antenna 3D** quyết định sector nào có lợi thế theo vị trí UAV.
2. **FSPL** quyết định xu hướng suy hao khi UAV xa BS hơn.
3. **LoS/NLoS và shadow** tạo biến động môi trường ở thang mét đến hàng chục mét.
4. **Fast fading** tạo rung nhanh từng measurement, không nên bị nhầm với xu
   hướng suy giảm dài hạn.
