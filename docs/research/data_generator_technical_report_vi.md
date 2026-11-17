# Báo cáo kỹ thuật Data Generator UAV

## 1. Giới thiệu tổng quan

`data_generator` là bộ mô phỏng lớp vô tuyến của HandoverOptimDRL. Nó sinh chuỗi RSRP và SINR theo thời gian cho UAV-UE di chuyển giữa các BS mặt đất. Các trace này là đầu vào cho phần handover và PPO tại `src/ho_optim_drl/`.

Generator không chọn serving cell, không chạy A3/TTT, HO preparation/execution, RLF, ping-pong, train hoặc evaluate policy. Phân tách này phù hợp: dữ liệu vô tuyến được cố định trước, còn logic handover thuộc môi trường RL. Vì vậy một thay đổi config generator có thể làm kết quả mới không còn so sánh trực tiếp được với dataset/train run cũ.

Về trực giác, mỗi route là một “chuyến bay” mô phỏng. UAV đi trên road grid trong một khu vực chữ nhật; ở mỗi thời điểm, generator tính chất lượng tín hiệu từ mọi sector quanh UAV rồi ghi hai vector: RSRP (mức công suất tham chiếu) và SINR (chất lượng tín hiệu sau khi tính nhiễu và noise). Handover environment nhận các vector này ở bước sau để quyết định nên ở lại cell hiện tại, chuyển cell, hoặc đánh giá sự cố liên quan đến chất lượng liên kết.

Đầu vào quyết định bốn lớp hành vi của trace:

1. **Topology và antenna:** vị trí site/sector, định hướng beam và gain theo hướng UAV.
2. **Mobility:** đường đi, tốc độ, altitude và độ phân giải road grid.
3. **Propagation:** FSPL, LoS/NLoS, shadow fading và small-scale fading.
4. **Measurement/output:** carrier, bandwidth, khoảng lấy mẫu, cách quy đổi RSRP/SINR, định dạng và provenance CSV.

Nhờ seed determinisitic và manifest, cùng speed/route ID, source và config sẽ cho lại cùng trace. Ngược lại, không nên coi dataset từ config khác nhau là cùng một phân phối dữ liệu, kể cả khi chỉ thay một tham số như grid spacing hoặc LoS correlation distance.

## 2. Ý nghĩa chi tiết các tham số `config.py`

Các giá trị dưới đây là cấu hình module-level trong `data_generator/src/uav_simulation/config.py`. Với profile P1–P4, một số tham số urban/channel được override tạm thời khi generation; giá trị thực tế luôn được đóng băng trong `simulation_manifest.json`.

### 2.1 Công suất, gain và carrier

| Tham số | Giá trị | Ý nghĩa và tác động |
|---|---:|---|
| `PTX` | 35 dBm | Tổng công suất phát của một BS sector. Tăng 3 dB làm received power của sector đó tăng xấp xỉ 3 dB, đồng thời cũng làm nó gây nhiễu mạnh hơn cho các UE đang phục vụ bởi sector khác. |
| `GTX` | 2 dB | Gain phát fallback khi tắt antenna 3D. Khi `USE_3GPP_3D_ANTENNA=True`, gain thực tế đến từ pattern antenna và `GTX` không được cộng thêm. |
| `GRX` | 2 dB | Gain anten thu UAV-UE; cộng trực tiếp vào received power của mọi sector. Do cộng đồng đều cho signal và interference, thường ảnh hưởng SINR ít hơn RSRP trong miền interference-limited. |
| `FC` | 2.1 GHz | Carrier tính theo GHz trong generator, lấy từ `DEFAULT_CARRIER_HZ`. Nó ảnh hưởng FSPL và Doppler; carrier cao hơn làm FSPL và Doppler tăng. |
| `SPEED_OF_LIGHT_M_S` | 299,792,458 m/s | Hằng số dùng chuyển đổi carrier/vận tốc thành Doppler. Không nên thay đổi. |

`PTX` là công suất toàn băng trong mô hình hiện tại, không phải công suất của một RE/PRB. Generator quy đổi received power toàn băng sang RSRP ở bước cuối.

### 2.2 Antenna 3D và sectorization

| Tham số | Giá trị | Ý nghĩa và tác động |
|---|---:|---|
| `USE_3GPP_3D_ANTENNA` | `True` | Bật gain direction-dependent của panel 3D. Nếu tắt, mỗi site chỉ có bearing 0° và dùng `GTX`; distribution trace thay đổi đáng kể. |
| `BS_SECTOR_BEARINGS_DEG` | 0°, 120°, 240° | Hướng boresight của ba sector mỗi site. Thứ tự này chi phối thứ tự các cột `bs_i`. |
| `BS_ANTENNA_NUM_VERTICAL` | 8 | Số phần tử theo chiều dọc của array. Nhiều phần tử hơn thường tạo beam vertical hẹp/sắc hơn. |
| `BS_ANTENNA_NUM_HORIZONTAL` | 4 | Số phần tử theo chiều ngang. Tăng giá trị này làm pattern azimuth có độ phân giải/directivity cao hơn. |
| `BS_ANTENNA_SPACING_VERTICAL_LAMBDA` | 0.5 | Khoảng cách phần tử dọc theo đơn vị bước sóng; 0.5λ là lựa chọn chuẩn để hạn chế grating lobes. |
| `BS_ANTENNA_SPACING_HORIZONTAL_LAMBDA` | 0.5 | Khoảng cách phần tử ngang theo bước sóng. |
| `BS_ANTENNA_MECHANICAL_DOWNTILT_DEG` | 0° | Nghiêng cơ học toàn panel. Giá trị lớn hơn hướng beam xuống mặt đất nhiều hơn, có thể làm UAV ở cao nhận tín hiệu yếu hơn. |
| `BS_ANTENNA_ELECTRICAL_DOWNTILT_DEG` | 10° | Nghiêng beam bằng pha phần tử. Đây là tham số rất nhạy với UAV 100 m: downtilt quá lớn có thể tạo sidelobe coverage và biến động RSRP nhanh. |
| `BS_ANTENNA_MECHANICAL_SLANT_DEG` | 0° | Xoay cơ học/elevation reference của panel. |
| `BS_ANTENNA_POLARIZATION_SLANT_DEG` | 45° | Góc phân cực phần tử ±45°, cách biểu diễn phổ biến cho cross-polarized panel. |
| `BS_ANTENNA_POLARIZATION_MODEL` | 2 | Chọn polarization model trong hiện thực antenna. Cần giữ cố định để so sánh dataset. |
| `BS_ANTENNA_MAX_ELEMENT_GAIN_DBI` | 8 dBi | Peak gain của một element, trước array factor. |
| `BS_ANTENNA_THETA_3DB_DEG` | 65° | Vertical 3 dB beamwidth của element: nhỏ hơn nghĩa là beam vertical hẹp hơn. |
| `BS_ANTENNA_PHI_3DB_DEG` | 65° | Horizontal 3 dB beamwidth của element. |
| `BS_ANTENNA_VERTICAL_SIDE_LOBE_ATTENUATION_DB` | 30 dB | Hạn chế attenuation/gain của vertical sidelobes. |
| `BS_ANTENNA_FRONT_BACK_ATTENUATION_DB` | 30 dB | Front-to-back attenuation: lớn hơn làm tín hiệu phía sau panel thấp hơn. |

Các tham số antenna không chỉ thay mức RSRP trung bình. Chúng có thể thay đổi thứ tự cell tốt nhất, vùng overlap giữa sector và do đó thay đổi trực tiếp số occasion handover mà PPO nhìn thấy.

### 2.3 Thời gian, độ cao và map

| Tham số | Giá trị | Ý nghĩa và tác động |
|---|---:|---|
| `DELTA_T_MS` | 10 ms | Chu kỳ measurement/logging. Nó cũng phải khớp với `Config.delta_t_ms` của handover environment; mọi timer A3/TTT/RLF quy đổi thành số steps dựa trên giá trị này. |
| `H_BS` | 25 m | Độ cao BS mặt đất. Kết hợp với UAV altitude để xác định elevation angle, 3D distance và antenna gain. |
| `H_UT` | 100 m | Default UAV altitude. Batch có thể override bằng `--altitude`; khi đó giá trị route có ưu tiên hơn `H_UT`. |

Map width/height, grid spacing, số step và UAV speed là tham số CLI/route chứ không nằm trong `config.py`. Canonical profile dùng map 2,000×1,667 m, road grid 100 m, 6,000 step. Với `DELTA_T_MS=10`, một route kéo dài chính xác 60 s.

### 2.4 Urban building model và LoS/NLoS

| Tham số | Giá trị baseline | Ý nghĩa và tác động |
|---|---:|---|
| `BUILDING_AREA_RATIO` | 0.3 | Tỷ lệ diện tích xây dựng `ζ` trong mô hình statistical urban. Tăng tỷ lệ này thường giảm LoS probability. |
| `BUILDING_DENSITY_KM2` | 500 | Mật độ building `ξ`. Cùng với `ζ`, nó quyết định số building crossings kỳ vọng theo khoảng cách. |
| `BUILDING_HEIGHT_RAYLEIGH_M` | 15 m | Rayleigh scale `Ω` của độ cao nhà. Giá trị cao hơn làm khả năng che đường truyền UAV–BS tăng. |
| `PATHLOSS_MODE` | `correlated_state` | `correlated_state` chọn state LoS/NLoS theo không gian; `average` trả về weighted pathloss theo LoS probability. Default phù hợp hơn với trace động. |
| `LOS_STATE_CORRELATION_DISTANCE_M` | 50 m | Khoảng cách correlation của latent LoS state. Lớn hơn tạo đoạn LoS/NLoS dài và mượt; nhỏ hơn tạo channel state đổi nhanh hơn. |
| `LOS_STATE_SHARED_PER_SITE` | `True` | Ba sector tại cùng site chia sẻ blockage state. Nếu tắt, mỗi sector có state riêng, có thể tạo diversity không thực tế giữa sectors đồng vị trí. |
| `EXCESS_LOSS_LOS_DB` | 1 dB | Loss cộng thêm FSPL khi link LoS. |
| `EXCESS_LOSS_NLOS_DB` | 20 dB | Loss cộng thêm FSPL khi NLoS. Chênh 19 dB với LoS tạo các quality transition mạnh. |

P1/P4 thay urban parameters thành `ζ=.5`, `ξ=300`, `Ω=20 m`. Đây không chỉ có nghĩa “nhiều nhà hơn”: tỷ lệ diện tích, mật độ và chiều cao tác động đồng thời lên xác suất clear path; cần đọc cả ba tham số thay vì suy ra từ tên `dense_urban`.

### 2.5 Shadow fading

| Tham số | Giá trị | Ý nghĩa và tác động |
|---|---:|---|
| `SF_SIGMA['LOS']` | 4.6 dB | Standard deviation shadow fading trong state LoS. |
| `SF_SIGMA['NLOS']` | 10 dB | Standard deviation shadow fading trong state NLoS; lớn hơn nên NLoS có spread RSRP mạnh hơn. |

Generator tạo shadow sample Gaussian, cắt tại ±3σ, sau đó cập nhật correlated theo quãng đường. Khoảng decorrelation không khai báo trực tiếp: `sf_decorr=max(20 m, 0.25×grid_spacing)`. Vì vậy thay `grid_spacing` cũng thay động học shadow fading, một lý do grid spacing phải được xem là phần của data-distribution contract.

### 2.6 Small-scale fading và Doppler

| Tham số | Giá trị | Ý nghĩa và tác động |
|---|---:|---|
| `ENABLE_SMALL_SCALE_FADING` | `True` | Bật fading nhanh. Nếu tắt, trace chỉ còn large-scale pathloss và shadow fading. |
| `RICIAN_K_FACTOR_DB` | 10 dB | Tỷ lệ công suất direct/diffuse cho LoS. K cao hơn làm LoS ổn định hơn; profile P3/P4 dùng 6 dB để LoS biến động mạnh hơn. |
| `FADING_NUM_SINUSOIDS` | 16 | Số diffuse waves của sum-of-sinusoids. Giá trị lớn hơn xấp xỉ process Rayleigh/Jakes mượt hơn nhưng tốn tính toán hơn. Code yêu cầu tối thiểu 4. |
| `FADING_SUBSTEP_MS` | 1 ms | Time resolution nội bộ để average fading trong một logged step. Giá trị phải dương; 1 ms với logging 10 ms tạo 10 mẫu. |

Doppler cực đại tỷ lệ với `v×fc/c`; do đó tăng tốc độ hoặc carrier làm fading theo thời gian nhanh hơn. Đây là lý do P2 (speed cao) và P3 (correlation ngắn/K thấp) kiểm tra hai loại channel difficulty khác nhau.

### 2.7 Bandwidth, resource grid, noise và measurement convention

| Tham số | Giá trị | Ý nghĩa và tác động |
|---|---:|---|
| `BANDWIDTH_MHZ` | 10 MHz | Bandwidth tính thermal noise và full-band SINR. Nó lấy từ shared project default. |
| `NUM_RESOURCE_BLOCKS` | 51 | Số RB dùng quy đổi full-channel received power thành per-RE RSRP. |
| `SUBCARRIERS_PER_RB` | 12 | Số subcarrier/RB; tổng normalization là 612 RE. |
| `THERMAL_NOISE_DENSITY_DBM_HZ` | −174 dBm/Hz | Noise density thermal chuẩn. |
| `NOISE_FIGURE_DB` | 7 dB | Receiver implementation noise cộng vào thermal noise. |
| `REFERENCE_SIGNAL_POWER_OFFSET_DB` | 0 dB | Offset công suất RE tham chiếu so với data RE; 0 nghĩa không có bù riêng. |

RSRP giảm theo `10log10(51×12)` so với received power toàn băng. SINR không chia qua 612 RE: signal, interference và noise đều ở miền full-band. Vì vậy không nên so sánh trực tiếp một RSRP dBm với SINR dB mà không hiểu measurement convention này.

`NUM_RESOURCE_BLOCKS=51` cùng `BANDWIDTH_MHZ=10` cần được diễn giải rõ hơn: nếu dùng SCS 15 kHz, 51×12×15 kHz=9.18 MHz trong khi noise dùng 10 MHz. Cấu hình chưa có `SUBCARRIER_SPACING_HZ`, nên nên thêm tham số đó hoặc mô tả 51 RB là normalization thực nghiệm. Sai khác này không làm generator lỗi, nhưng cần giữ bất biến khi so sánh các thí nghiệm.

## 3. Kiến trúc

| Thành phần | Nhiệm vụ |
|---|---|
| `config.py` | Hằng số radio, antenna, đô thị, fading và noise. |
| `simulation.py` | Điều phối topology, mobility, measurement loop và output. |
| `radio_models.py` | Path loss, LoS/NLoS, shadow fading, Doppler, RSRP và SINR. |
| `mobility.py` | Chuyển động Manhattan trên road grid. |
| `hexagons.py` | Hexagonal site cover. |
| `antenna_model.py` | 3GPP TR 38.901 single-panel sector array. |
| `logger.py` | Xuất CSV train-ready/legacy. |
| `manifest.py` | Provenance manifest, source hash và metadata file. |
| `batch/generate_dataset.py` | Generator tổng quát, profile P0–P4, parallel workers. |
| `validation/channel_statistics.py` | Kiểm định thống kê channel. |

Luồng xử lý:

```text
CLI/profile → validate → manifest → seed(speed, route)
→ topology + Manhattan route → RSRP/SINR mỗi 10 ms
→ cặp CSV → schema/row validation → metadata SHA-256
→ handover environment / PPO
```

Train-ready output yêu cầu đúng một UE. Mỗi route tạo `rsrp_{speed}kmh_{route}.csv` và `sinr_{speed}kmh_{route}.csv`, với schema `sample,time_ms,bs_0,...`. `bs_i` là một sector/cell, không nhất thiết là một site vật lý.

## 4. Tóm tắt cấu hình cơ sở hiện tại

Generator lấy carrier, bandwidth và sampling interval từ `ho_optim_drl.config`, tránh khai báo radio defaults tách rời khỏi handover pipeline.

| Nhóm | Giá trị | Ý nghĩa |
|---|---:|---|
| Carrier | 2.1 GHz | Shared project default. |
| Bandwidth | 10 MHz | Dùng cho noise toàn băng. |
| Measurement interval | 10 ms | 6,000 samples = 60 s/route. |
| BS transmit power | 35 dBm | Tổng công suất phát trong mô hình. |
| BS/UAV-UE height | 25 m / 100 m | Altitude UAV truyền qua route. |
| Site grid | Hex, scale 500 m | Mỗi site có 3 sector 0°, 120°, 240°. |
| Antenna | 8×4 panel, 0.5λ | Electrical downtilt 10°, element max gain 8 dBi. |
| Pathloss mode | `correlated_state` | Realization LoS/NLoS theo không gian. |
| LoS correlation | 50 m | Profile fast-channel dùng 25 m. |
| Shadow fading | 4.6 dB LoS / 10 dB NLoS | Tương quan không gian, bị chặn ±3σ. |
| Small-scale fading | Bật | Rician LoS K=10 dB, Rayleigh NLoS, 16 sinusoids. |
| Fading substep | 1 ms | 10 sub-samples cho mỗi measurement. |
| Noise figure | 7 dB | Thermal density −174 dBm/Hz. |
| Excess loss | 1 dB LoS / 20 dB NLoS | Cộng vào FSPL. |

Môi trường urban baseline là built-up ratio `ζ=0.3`, density `ξ=500 buildings/km²`, Rayleigh building-height scale `Ω=15 m`.

## 5. Mô hình vô tuyến

### 5.1 Topology và antenna

`build_hex_cover` đặt site theo lưới hex và phủ thêm buffer biên. Mỗi site được triển khai thành ba sector cùng tọa độ, khác bearing. Khi 3D antenna bật, gain lấy từ 3GPP panel pattern theo azimuth/elevation UAV–BS; `GTX=2 dB` chỉ là fallback khi tắt 3D antenna, không cộng chồng lên pattern.

### 5.2 Large-scale propagation

Khoảng cách 3D:

```text
d3 = sqrt((xUE − xBS)² + (yUE − yBS)² + (hUE − hBS)²)
```

Pathloss free-space là `32.4 + 20log10(fc_GHz) + 20log10(d3_m)`. LoS probability dùng mô hình statistical urban buildings: expected building crossings phụ thuộc khoảng cách ngang và `sqrt(ζ×ξ)`; xác suất không che tia phụ thuộc phân bố Rayleigh chiều cao nhà.

Mode mặc định `correlated_state` dùng latent Gaussian, cập nhật với `ρ=exp(−Δd/dcorr)`, rồi chọn hẳn nhánh LoS hay NLoS. Với `LOS_STATE_SHARED_PER_SITE=True`, ba sector của một site cùng chia sẻ trạng thái LoS/NLoS. Điều này tạo channel transition phù hợp hơn với bài toán handover so với weighted-average pathloss.

### 5.3 Shadow, Doppler và fast fading

Shadow cache theo site; decorrelation distance là `max(20 m, 0.25×grid_spacing)`, tức 25 m khi grid 100 m. Khi LoS/NLoS đổi, shadow state được khởi tạo lại theo σ của nhánh mới.

Fast fading chỉ chạy khi có state realization. Doppler lấy từ hướng/vận tốc tức thời và carrier. LoS là Rician (direct + diffuse), NLoS là Rayleigh (diffuse). Gain power được trung bình trong 10 ms nhằm tránh logging biến thiên quá nhanh.

### 5.4 RSRP và SINR

```text
Pr = Ptx + Gantenna + Grx − pathloss − shadow + fast fading
RSRP = Pr − 10log10(NUM_RB × 12) + reference-signal offset
SINR = S / (sum(I) + N)
N[dBm] = −174 + 10log10(10 MHz) + 7
```

SINR là full-band co-channel SINR: all non-serving sectors là interference và noise tính trên toàn 10 MHz. Đây là model nhất quán để tạo feature RL, nhưng không mô phỏng scheduler/load, per-PRB allocation, beam management, MIMO layers hay PDCCH BLER.

## 6. Mobility và route

UAV bắt đầu tại road-grid intersection, di chuyển Đông/Bắc/Tây/Nam và chọn hướng hợp lệ ở mỗi giao lộ để không vượt biên. Dịch chuyển mỗi sample:

```text
distance = speed_kmh × 1000 / 3600 × 0.010 m
```

Trong 60 s, quãng đường danh nghĩa là 500 m (30 km/h), 833 m (50), 1,167 m (70), 1,500 m (90), 2,000 m (120) và 2,500 m (150). Tốc độ cao do đó đi qua nhiều sector và có nhiều opportunity handover hơn.

## 7. Profile P0–P4

Geometry/radio canonical khóa: 6,000 steps, altitude 100 m, grid 100 m, 2,000×1,667 m, 2.1 GHz, 10 MHz, Δt 10 ms. `generate_dataset.py` dùng context manager để áp override trong thời gian generation và ghi effective config vào manifest.

| Profile | Urban | Channel | Tốc độ khóa |
|---|---|---|---|
| `p0_canonical_v1` | ζ=.3, ξ=500, Ω=15 m | dcorr=50 m; K=10 dB | Không khóa; thuận tiện pilot/smoke. |
| `p1_dense_urban_v1` | ζ=.5, ξ=300, Ω=20 m | dcorr=50 m; K=10 dB | Không khóa. |
| `p2_high_mobility_v1` | Baseline | dcorr=50 m; K=10 dB | 90/120/150/180/210/240 km/h. |
| `p3_fast_channel_v1` | Baseline | dcorr=25 m; K=6 dB | 30/50/70/90/120/150 km/h. |
| `p4_combined_harsh_v1` | Dense urban | dcorr=25 m; K=6 dB | 90/120/150/180/210/240 km/h. |

P1 cô lập urban khắc nghiệt; P2 cô lập mobility; P3 cô lập channel nhanh; P4 kết hợp hai stressor. Manifest ghi cả profile parameters lẫn `simulation_config`, vì vậy audit không phải suy luận từ tên thư mục.

## 8. Reproducibility và integrity

Seed mỗi trajectory là:

```text
seed = speed_kmh × 1000 + route_id
```

`simulation_manifest.json` lưu entrypoint, distribution version, SHA-256 source tree, profile/parameters, radio, toàn bộ uppercase config, seed rule, schema và split unit. Resume chỉ được phép khi manifest khớp tuyệt đối. Folder có trace nhưng thiếu manifest bị từ chối để tránh provenance không kiểm chứng được.

Generator chỉ bỏ qua route đã đủ hai file, đúng schema và đúng số rows. Khi batch xong, `dataset_metadata.json` xác nhận full trajectory set và lưu SHA-256 cho từng CSV. Split phải theo **whole trajectory**, không được tách row/sample của cùng route vào train/test vì sẽ gây temporal/spatial data leakage.

## 9. Vận hành

Production nên dùng batch headless và một output directory riêng cho mỗi config:

```powershell
.\.venv\Scripts\python.exe run.py generate_data `
  --output-dir data/uav_p0_canonical_v1_pilot `
  --profile p0_canonical_v1 `
  --speeds 30 50 70 90 120 150 `
  --routes-per-speed 20 `
  --steps 6000 --altitude 100 --grid-spacing 100 `
  --width 2000 --height 1667 --workers 4
```

Ưu tiên `scripts/run_p0_canonical_pilot.ps1` đến `scripts/run_p4_combined_harsh_pilot.ps1`, vì chúng đã truyền đúng contract. `data_generator/main.py` là launcher GUI/interactive, không phải entry point phù hợp cho dataset train-ready quy mô lớn.

## 10. Phát hiện và khuyến nghị

### 10.1 Default CLI và profile locked

Default `--grid-spacing` của generic generator là 500 m, nhưng mọi P0–P4 khóa grid 100 m. Vì vậy `generate --profile p0_canonical_v1` mà không truyền geometry sẽ bị từ chối. P2/P4 cũng từ chối default speed list vì cần high-mobility speeds. Pilot scripts không bị ảnh hưởng nhưng đây là rủi ro UX.

Khuyến nghị: hoặc đặt default CLI theo P0 canonical, hoặc giữ `profile=custom` nhưng hiển thị full command/preset chính xác trong help. Không nên âm thầm tự sửa tham số của named profile vì sẽ che mất sai khác experiment.

### 10.2 Bandwidth, RB và SCS

Noise dùng 10 MHz, còn RSRP chia qua `51×12=612` RE. Nếu ngầm giả định SCS 15 kHz thì occupied bandwidth là 9.18 MHz, khác 10 MHz khoảng 0.37 dB. Hiện config không khai báo SCS, nên 51 RB chưa có diễn giải chuẩn rõ ràng.

Khuyến nghị thêm `SUBCARRIER_SPACING_HZ` và validation `num_RB×12×SCS <= bandwidth`, hoặc ghi rõ `NUM_RESOURCE_BLOCKS` là RE normalization chỉ phục vụ RSRP. Cần giữ bất biến điều này khi so sánh train/evaluation.

### 10.3 Module-level config mutable

Context manager hiện restore config trong `finally`, và process worker có memory state riêng nên cách làm hoạt động an toàn cho batch hiện tại. Về lâu dài, object config bất biến truyền vào simulator sẽ dễ test hơn và tránh nhầm lẫn khi chạy nhiều jobs khác profile trong cùng process/thread.

### 10.4 Giới hạn mô hình

Đây là simulator nghiên cứu với physical assumptions có kiểm soát, không phải full 3GPP system-level network. Báo cáo kết quả phải nêu rõ full-band interference, BS power tĩnh, shared site LoS, mobility 2D theo grid, và việc thiếu scheduler/load/blocker-map.

## 11. Checklist trước khi train/công bố

1. Xác nhận `simulation_manifest.json`: profile, map, altitude, speeds, radio và source hash.
2. Xác nhận `dataset_metadata.json`: đủ pair RSRP/SINR, đúng trajectory count và hashes.
3. Chia train/validation/test theo trajectory.
4. Đối chiếu metadata radio với `Config(fc, bw, delta_t_ms)` của handover environment.
5. Dùng output directories khác nhau cho train, validation và zero-shot; không resume manifest khác.
6. Chạy channel validation/audit trước train lớn.
7. Lưu profile, source hash, seed rule và speed distribution vào experiment report.

## 12. Kết luận

Data generator có nền tảng tốt cho workflow RL: radio trace tách khỏi handover logic, profile stress rõ ràng, seed deterministic, manifest source hash, resume an toàn và metadata từng file. Cấu hình canonical hiện tại đủ nhất quán cho baseline nghiên cứu. Hai ưu tiên cải tiến là làm default CLI dễ dùng với named profile hơn và định nghĩa rõ quan hệ bandwidth–RB–SCS.
