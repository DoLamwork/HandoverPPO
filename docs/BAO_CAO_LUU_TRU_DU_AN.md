# Báo cáo lưu trữ dự án HandoverOptimDRL

**Ngày kiểm tra gần nhất:** 26/09/2026<br>
**Thư mục dự án:** `C:\Project\HandoverOptimDRL`<br>
**Nhánh Git hiện tại:** `codex/a3-multicandidate-ppo`<br>
**Commit nền trước đợt chỉnh sửa:** `b476a4d48b547b2013995aae31d08bfe38ad2ab3` (`result UAV data`, 31/07/2026)<br>
**Dataset chính theo xác nhận của tác giả:** `data/uav_100m_grid100m_dev_processed`<br>
**PPO run lịch sử:** `results/models/uav_100m_grid100m_ppo_30_90`<br>
**Benchmark lịch sử:** `results/benchmark/uav_100m_grid100m_all_speeds_percent`<br>
**Trạng thái test cuối:** 132/132 test qua ngày 26/09/2026.

> Báo cáo này mô tả trạng thái dự án sau đợt tích hợp, không chỉ commit nền
> `b476a4d`. Phần source của `data_generator/` và các file tích hợp được quản lý
> bởi Git; dataset dung lượng lớn được giữ cục bộ và loại khỏi Git thường.
> Muốn tái tạo đầy đủ thí nghiệm phải lưu dataset riêng cùng metadata/hash của
> training run; chỉ giữ source-code commit là chưa đủ.

> **Cập nhật multi-candidate 26/09/2026:** code hiện tại dùng 11 action
> `STAY + HO_C1..HO_C10` và observation 109 chiều. Các bảng kết quả model cũ
> trong báo cáo này là kết quả lịch sử của policy 19-feature/2-action, không
> phải kết quả của thiết kế mới. Xem `docs/a3_multicandidate_actions.md`.

## 1. Tóm tắt điều hành

HandoverOptimDRL là một framework nghiên cứu handover dựa trên trace. UAV di
chuyển trong mô phỏng và tạo ra chuỗi RSRP/SINR của mọi cell. Phần lõi
`src/ho_optim_drl` đọc các chuỗi này, tự chọn serving cell ban đầu, chạy A3/TTT,
RLM, handover, HOF, RLF, tái kết nối, tạo SMDP và huấn luyện PPO.

Ranh giới trách nhiệm đã được thống nhất và thể hiện trong code hiện tại:

- `data_generator` **chỉ sinh RSRP/SINR**; không có serving cell, kết nối, A3,
  TTT, handover, HOF hoặc RLF.
- `src/ho_optim_drl` là nơi duy nhất xử lý toàn bộ trạng thái kết nối và handover.
- A3+TTT vẫn trigger decision, nhưng PPO hiện chọn `STAY` hoặc handover tới một
  trong 10 candidate sector. `C1` là target A3; các slot còn lại được xây từ
  sector cùng site, site vật lý lân cận và RSRP hiện tại.

Dataset chính có 1.060 quỹ đạo, mỗi quỹ đạo gồm một cặp RSRP/SINR, 6.000 mẫu,
21 cell và bước mẫu 10 ms. PPO được train trên các tốc độ 30, 50, 70 và 90 km/h;
120 và 150 km/h chỉ được dùng để kiểm tra zero-shot.

Các kết quả benchmark lưu sẵn trong báo cáo này được tạo bởi model lịch sử có
hai action `STAY/HANDOVER`, không tương thích với không gian action hiện tại.
Kết quả lịch sử đó cho
thấy PPO đạt mục tiêu tối ưu reward tốt hơn baseline A3 Always-HO. Trên 260
trace, PPO có mean controlled return `-7,584`, so với
`-25,819` của A3 Always-HO. Đổi lại, PPO hy sinh
một phần estimated rate so với Always-HO: `20,890` so với `22,179 Mbps`. PPO
giảm HOF từ `12,182%` xuống `8,643%`, RLF recovery time từ `1,649%` xuống
`1,116%`, và ping-pong từ `37,670%` xuống `13,518%`.

Các kết quả này nên được xem là **kết quả sơ bộ của một checkpoint đơn seed**,
chưa phải kết luận cuối cùng. Code hiện tại đã sửa lỗi collector và thống nhất
mặc định radio, nhưng artifact lịch sử chưa được retrain/regenerate:

1. Run lịch sử khai báo 600 train trace nhưng log chứng minh chỉ 162 trace thực sự
   đóng góp macro-transition. Nguyên nhân là validation làm mất trạng thái collector
   và seed lại train environment. Lỗi này đã được sửa trong source hiện tại; các
   checkpoint cũ không tự thay đổi và vẫn phải retrain.
2. Dataset chính không có `simulation_manifest.json`; checkpoint cũ cũng không
   lưu protocol config. Compact metadata `dataset_provenance.json` hiện đã ghi
   topology 7 site/21 sector và cấu hình lịch sử được phục dựng. Runtime vẫn cố
   ý dùng 2,1 GHz/10 MHz/10 ms; file provenance không tự override runtime.

## 2. Phạm vi và cách kiểm tra

Báo cáo được lập theo năm nhóm kiểm tra độc lập:

1. **Đọc code theo luồng thực thi:** CLI, generator, converter, loader, split,
   protocol engine, SMDP, PPO, evaluate và benchmark.
2. **Kiểm kê dataset thực tế:** số file, dung lượng, naming, số mẫu, thời gian,
   số cell và phân bố theo vận tốc.
3. **Đối chiếu artifact huấn luyện:** `training_report.json`, log JSONL, ba
   checkpoint và dataset split manifest.
4. **Tính lại kết quả benchmark:** tổng hợp `per_trajectory.csv` thành tập test
   cùng miền, tập zero-shot và toàn bộ 260 trace; đối chiếu với `aggregate.csv`,
   `by_speed.csv` và `paired_comparison.csv`.
5. **Kiểm thử:** chạy toàn bộ `unittest` và `git diff --check`.

Đã kiểm tra lại SHA-256 của toàn bộ 2.000 CSV thuộc manifest train/validation/test:
`checked=2000`, `bad=0`. Hash của manifest là
`6e445e7cbb78a65fe41d780479e65f21a32d71851d42b9525edf4ebad94a7890`.

Không retrain PPO và không chạy lại toàn bộ benchmark 260 trace trong lần lập báo
cáo này. Mọi số liệu mô hình được đọc từ artifact đã lưu; việc tính lại chỉ là
phép tổng hợp trên kết quả từng trajectory.

## 3. Bài toán mà project thực sự giải quyết

### 3.1. Input, output và biến điều khiển

Input của environment là hai ma trận theo thời gian:

- RSRP của mọi cell, dùng để tạo measurement event A3 sau lọc L3.
- SINR của mọi cell, dùng làm proxy RLM và ước lượng rate.

Mỗi trajectory giữ nguyên trong một episode. Agent không điều khiển đường bay,
tốc độ, độ cao, công suất phát, antenna beam, resource allocation, scheduler hay
cell load. Agent hiện trực tiếp chọn target trong shortlist 10 cell được chốt
khi A3 hoàn tất TTT. `C1` là target A3; `C2..C10` cho phép thử các sector/site
lân cận dựa trên topology và tín hiệu causal, không dùng mẫu tương lai.

Output nghiên cứu gồm:

- policy PPO quyết định chấp nhận/từ chối cơ hội handover;
- estimated rate theo Shannon;
- số lần handover, HOF, RLF, ping-pong và thời gian kết nối;
- reward/return dùng để train và chọn checkpoint;
- bảng so sánh ghép cặp PPO và A3 Always-HO.

### 3.2. Đây là trace-driven on-policy RL

Các trace radio đã được sinh trước, nhưng transition không phải một dataset RL cố
định. Serving cell, A3, target, RLF và quyết định tiếp theo phụ thuộc vào chuỗi hành
động của policy hiện tại. Trong training, PPO vẫn thu rollout mới từ simulator
dưới policy hiện tại. Vì vậy mô tả chính xác là **trace-driven on-policy RL**, không
phải offline RL theo nghĩa học từ một bảng transition cố định.

### 3.3. Thuật ngữ

| Thuật ngữ | Ý nghĩa trong project |
|---|---|
| RSRP | Công suất reference signal của từng cell; sau lọc L3 được dùng cho A3 |
| SINR | Chất lượng tín hiệu trên nhiễu; dùng cho rate và làm proxy Qin/Qout |
| A3 | Neighbor trở nên tốt hơn serving cell theo offset/hysteresis |
| TTT | Thời gian A3 phải duy trì trước khi tạo cơ hội quyết định |
| HOF | Handover execution đã bắt đầu nhưng target không đạt điều kiện phù hợp |
| RLF | Radio link failure; có thể do T310 hết hạn hoặc do HOF |
| PP | Ping-pong A→B→A trong cửa sổ thời gian cấu hình |
| SMDP | Decision chỉ xuất hiện tại event A3; thời gian giữa các decision thay đổi |
| Macro-step | Khoảng từ một decision đến decision kế tiếp hoặc cuối trace |

## 4. Kiến trúc và luồng end-to-end

```text
data_generator/
  mobility + topology + antenna + channel
                   |
                   v
data/<dataset>/{rsrp|sinr}_{speed}kmh_{id}.csv
  schema: sample,time_ms,bs_0,... + manifest + metadata
                   |
                   | loader: validate + lọc L3 RSRP
                   v
RadioTrace [RSRP_A3, SINR_rate, SINR_RLM]
                   |
                   v
HandoverEngine: serving/A3/TTT/HO/RLM/RLF/reconnection
                   |
                   v
SMDP: observation 109 chiều -> STAY/HO_C1..HO_C10 -> macro-transition
                   |
          +--------+---------+
          |                  |
          v                  v
       PPO train       evaluate / benchmark
          |                  |
          v                  v
results/models/*       results/benchmark/*
```

Các lớp chính:

| Lớp | Trách nhiệm | Code chính |
|---|---|---|
| Generator | Mobility, topology, antenna, channel, CSV RSRP/SINR | `data_generator/src/uav_simulation/` |
| Data | Chuẩn hóa, validate, split và tạo `RadioTrace` | `src/ho_optim_drl/data/` |
| Protocol | A3, timer/counter, HO, RLF và serving state | `src/ho_optim_drl/protocol/` |
| SMDP | Action, observation, reward, macro-transition | `src/ho_optim_drl/smdp/` |
| Gym environment | API reset/step, chọn trajectory và episode accounting | `src/ho_optim_drl/envs/handover.py` |
| PPO | Actor-critic, rollout, GAE, update, checkpoint | `src/ho_optim_drl/agents/ppo/` |
| Evaluation | Chạy cùng trace và tổng hợp metric ghép cặp | `src/ho_optim_drl/evaluation/benchmark.py` |
| Commands | Ghép các thành phần thành workflow CLI | `src/ho_optim_drl/commands/` |

Điểm thiết kế tốt là protocol engine dùng chung cho policy học và policy baseline.
`AlwaysHandoverPolicy` đã được test tick-by-tick với reference simulator, giảm nguy
cơ mỗi policy dùng một mô hình handover khác nhau.

## 5. Data generator

### 5.1. Ranh giới sau khi dọn code

Class trung tâm hiện có tên `CellularNetworkReceivedPower` và docstring ghi rõ
“generate per-cell RSRP/SINR traces without connection or handover state”. Vòng
lặp mỗi frame chỉ làm ba việc:

1. cập nhật vị trí UAV;
2. tính RSRP/SINR tới mọi cell;
3. ghi các giá trị vào CSV.

Generator hiện không còn các biến `serving_bs`, `previous_serving_bs`, HOM,
sensitivity, range/candidate phục vụ handover, không vẽ serving link và không gọi
bất kỳ state machine kết nối/handover nào. Đây là ranh giới đúng theo yêu cầu của
dự án.

### 5.2. Cấu hình hiện tại của generator

| Nhóm | Giá trị hiện tại |
|---|---|
| Công suất mỗi sector | 35 dBm |
| Carrier | 2,1 GHz, lấy từ shared default của handover environment |
| Bandwidth | 10 MHz, lấy từ shared default của handover environment |
| Ground BS height | 25 m |
| UAV height cho batch chính | 100 m |
| Antenna | panel 3GPP 8×4, 3 sector/site, bearing 0°/120°/240° |
| Electrical downtilt | 10° |
| Element max gain | 8 dBi |
| Map của batch grid100 | 2.000 × 1.667 m |
| Road-grid spacing | 100 m |
| Hex site scale | 500 m |
| Samples/trace | 6.000 |
| Radio sample interval | 10 ms |
| Duration quy ước | 60 s/trace |
| Pathloss mode | spatially correlated LoS/NLoS state |
| LoS correlation distance | 50 m, dùng chung state theo site |
| Shadowing sigma | 4,6 dB LoS; 10 dB NLoS |
| Small-scale fading | Rician K=10 dB khi LoS; Rayleigh khi NLoS |
| Fading synthesis | 16 sinusoids, substep 1 ms, lấy trung bình công suất trong 10 ms |
| Noise | -174 dBm/Hz + 10log10(BW) + NF 7 dB |
| Seed rule | `speed_kmh * 1000 + route_id` |

Với map batch grid100, code tạo 7 site và 3 sector mỗi site, tương ứng đúng 21
cột cell trong dataset. Route đi theo Manhattan grid. Ở mỗi giao điểm, hướng mới
được chọn ngẫu nhiên trong các hướng vẫn nằm trong biên; quay ngược lại là một
khả năng hợp lệ của mô hình hiện tại.

### 5.3. Mô hình radio đang dùng

Khoảng cách là khoảng cách 3D giữa UAV và BS. Free-space pathloss cơ sở:

```text
PL_FS(dB) = 32.4 + 20 log10(fc_GHz) + 20 log10(d_3D,m)
```

Nhánh LoS/NLoS cộng excess loss tương ứng 1 dB hoặc 20 dB. Trạng thái nhánh được
lấy mẫu với xác suất LoS từ mô hình thống kê công trình đô thị và có tương quan
không gian. Shadowing và small-scale fading sau đó được đưa vào received power.

```text
P_rx,total(dBm) = P_tx + G_tx(theta,phi) + G_rx
                  - pathloss - shadow_loss + fading_gain
```

RSRP được quy đổi từ full-channel received power sang một reference resource
element:

```text
RSRP(dBm) = P_rx,total - 10 log10(N_RB * 12) + RS_power_offset
```

Hiện `N_RB=51`, `12` subcarrier/RB và power offset bằng 0. SINR cell `j` dùng tất
cả cell/sector còn lại làm nhiễu đồng kênh:

```text
SINR_j = P_j / (sum(P_i, i != j) + noise)
```

Trong mô hình đồng kênh này, tại cùng một thời điểm, thứ tự cell theo full-channel
received power và SINR là đơn điệu. Khác biệt trong quyết định vẫn có thể xuất hiện
vì A3 dùng RSRP đã lọc L3, còn rate/RLM dùng SINR raw và serving cell là trạng thái
giao thức, không tự đổi sang argmax ở mỗi mẫu.

### 5.4. Generator ghi dữ liệu ở đâu?

Tất cả đường dẫn tương đối được tính từ project root.

| Lệnh/chế độ | Output mặc định |
|---|---|
| `run.py generate_data` hoặc `uav-sim generate` | `data/uav_generated/` |
| GUI/single run | `data/UAV_single_runs/` |
| `uav-sim batch` | `data/uav_100m_generated/` |
| `uav-sim batch-300m` | `data/uav_300m_generated/` |
| `uav-sim batch-grid100` | `data/uav_100m_grid100m_v3/` |
| Antenna/system figures | `figures/uav_simulation/` |
| Channel validation | `results/channel_validation/` |

`--output-dir` của `generate_data` chọn thư mục dataset. Các lệnh batch hiện ghi
thẳng một cặp train-ready cho mỗi route:

```text
rsrp_30kmh_0.csv
sinr_30kmh_0.csv
```

Schema mới là `sample,time_ms,bs_0,bs_1,...`; số radio được làm tròn hai chữ số
thập phân. Generic generator và các preset `batch`, `batch-300m`, `batch-grid100`
đều dùng contract `handover_ppo_train_ready_v1`. Trước khi chạy chúng ghi
`simulation_manifest.json`; sau khi đủ toàn bộ cặp, đúng header và số dòng, chúng
ghi `dataset_metadata.json` cùng SHA-256 từng CSV. Output hoàn tất có thể đưa thẳng
vào `train_ppo`. GUI/single-run nhiều UE vẫn giữ schema legacy để khảo sát.

## 6. Chuẩn hóa và nạp dữ liệu

### 6.1. Khi nào còn cần `prepare_data`?

Generator mới không cần `prepare_data`. Command này chỉ là adapter cho dataset
legacy dạng `ue0_*`, `Step,bs0,...`; nó không sinh lại kênh và không thay đổi
RSRP/SINR. Khi dùng legacy, nó:

- ghép đúng cặp raw RSRP/SINR theo UE, speed và route ID;
- kiểm tra Step, shape và danh sách BS của hai file khớp nhau;
- đổi tên thành `rsrp_30kmh_0.csv` và `sinr_30kmh_0.csv`;
- thêm `sample,time_ms` và đổi `bs0` thành `bs_0`;
- giữ nguyên route ID trong pipeline tích hợp;
- tính SHA-256 raw/output;
- ghi `dataset_metadata.json` nếu dùng command hiện tại.

Schema chuẩn mà loader chấp nhận:

```text
sample,time_ms,bs_0,bs_1,...,bs_20
```

Mỗi file phải bắt đầu ở 0 ms, sample liên tục từ 0, time đồng đều và đúng
`delta_t_ms`; toàn bộ giá trị phải hữu hạn. Các chữ số dài như
`-138.03999999999999` trong file processed là biểu diễn nhị phân sau khi giữ giá
trị hai chữ số của raw CSV, không phải generator có độ chính xác đo cao hơn.

### 6.2. Biến đổi trong loader

Loader tạo ba stream `(base_stations, timesteps)`:

- `rsrp_a3_db`: RSRP qua lọc L3;
- `sinr_rate_db`: SINR raw;
- `sinr_rlm_db`: hiện là bản sao độc lập của SINR raw.

Lọc L3 là exponential smoothing:

```text
F[n] = (1-w) F[n-1] + w M[n]
w = 1 / 2^(K/4) = 0.0625 khi K=16
```

SINR không được resample và không qua L3. `RadioTrace` sao chép, kiểm tra shape và
đặt các mảng thành read-only để tránh policy vô tình sửa input.

### 6.3. Vấn đề metadata của dataset chính lịch sử

`data/uav_100m_grid100m_dev_processed` không có:

```text
simulation_manifest.json
dataset_metadata.json
```

Khi thiếu metadata, `load_protocol_config()` dùng `Config()` lịch sử:

```text
fc = 2.1 GHz
bw = 10 MHz
delta_t = 10 ms
```

Carrier của environment không được dùng để tính lại RSRP/SINR, nên trường `fc`
không tự biến một trace cũ sang 2,1 GHz. `bw` lại được dùng trực tiếp trong
estimated rate và reward. Vì vậy nếu CSV cũ thực sự được sinh với bandwidth khác
10 MHz thì run lịch sử vẫn không nhất quán; không thể sửa hoàn toàn bằng cách đổi
metadata hoặc nhân cột Mbps sau khi train.

`prepare_data` hiện lấy bộ mặc định dùng chung 2,1 GHz/10 MHz/10 ms nếu raw dataset
không có generation manifest. Đây vẫn chỉ là fallback, không phải bằng chứng suy
ra từ CSV. Với legacy data, cần truyền tham số tường minh sau khi xác nhận cấu hình
nguồn; tốt nhất là regenerate để generator tự ghi manifest.

## 7. Cơ chế serving, handover và RLF trong `src/ho_optim_drl`

### 7.1. Khởi tạo serving cell

Ở mẫu đầu, engine tìm cell có SINR cao nhất. Chỉ khi SINR cell này lớn hơn proxy
`Qin=-6 dB` mới kết nối. Serving cell sau đó là trạng thái RRC; nó không tự chuyển
sang cell argmax ở mỗi frame.

### 7.2. Measurement và A3

Khi đang kết nối, RRC chọn neighbor có RSRP L3 lớn nhất khác serving cell. Với các
offset object/cell bằng 0, điều kiện vào A3 hiện tại là:

```text
M_neighbor - Hys > M_serving + Off
```

Với `Hys=1 dB`, `Off=1 dB`, neighbor phải lớn hơn serving trên 2 dB. Điều kiện
rời A3 là:

```text
M_neighbor + Hys < M_serving + Off
```

tức chênh lệch nhỏ hơn 0 dB. Trong deadband 0–2 dB, trạng thái A3 trước đó được
giữ nguyên. Nếu target đổi trong lúc TTT đang chạy, TTT được reset. TTT hiện là
40 ms, tương ứng 4 tick.

### 7.3. Decision và 11 action

Khi A3 duy trì đủ TTT, engine dừng tại `DECISION` và chốt:

- serving cell;
- target cell;
- tick A3 bắt đầu;
- A3 margin hiện tại.

Không thể tiến thời gian tiếp nếu chưa gọi một trong 11 action:

- `STAY`: từ chối event hiện tại; cùng cặp serving-target bị suppress cho đến khi
  A3 rời điều kiện hoặc target thay đổi. Cơ chế này tránh việc agent bị hỏi cùng
  một quyết định ở mọi tick.
- `HO_C1`: handover tới target A3 đã chốt.
- `HO_C2..HO_C10`: handover tới các candidate sector được chốt cùng observation.

### 7.4. Handover execution

Các giá trị mặc định:

| Tham số | Giá trị | Ý nghĩa |
|---|---:|---|
| HO preparation | 50 ms | Giả định mô phỏng, serving link vẫn truyền data |
| HO execution | 40 ms | UE disconnect; delivered rate bằng 0 |
| Target success | target SINR > Qin | Proxy cho random access/T304, không phải full NR |
| Ping-pong window | 1.000 ms | Đếm successful HO A→B→A |

Target được latch xuyên suốt preparation/execution. Khi execution bắt đầu, T310
được dừng, serving cũ được ghi lại và UE disconnect. Sau execution, nếu target
không đạt Qin thì HOF dẫn vào RLF recovery; nếu đạt thì UE reconnect sang target.

Lưu ý: docstring reward cũ nói cả preparation và execution đều mất rate, nhưng
hành vi code thực tế chỉ mất kết nối từ lúc execution. Báo cáo và bài viết phải
theo hành vi code.

### 7.5. Radio link monitoring và RLF

Dataset không có hypothetical PDCCH BLER nên project dùng SINR threshold làm
proxy:

- `Qout=-8 dB`: out-of-sync;
- `Qin=-6 dB`: in-sync;
- khoảng giữa hai ngưỡng: không có indication mới.

N310=10 out-of-sync indications làm chạy T310=1.000 ms. N311=3 in-sync
indications dừng T310. Khi T310 hết hạn, hoặc handover execution thất bại, engine
disconnect và bắt đầu RLF recovery cố định 200 ms. Hết recovery, engine tìm cell
có SINR cao nhất vượt Qin rồi reconnect.

Đây là A3-based/3GPP-inspired reference có các transition được đối chiếu với TS
38.331, không phải một NR protocol stack đầy đủ. Qin/Qout ở đây là proxy SINR,
50/40/200 ms là giả định mô phỏng, và chưa có PDCCH BLER, T304/T311 đầy đủ, random
access, HARQ hay scheduler.

### 7.6. Estimated rate

Khi kết nối:

```text
rate_Mbps = bw_Hz * log2(1 + 10^(SINR_dB/10)) / 1e6
```

Khi disconnect, HO execution hoặc RLF recovery, delivered rate bằng 0. Đây là
Shannon upper-bound estimate, không phải throughput đo được. Không có MCS ceiling,
protocol overhead, resource sharing, load hay retransmission.

## 8. Định nghĩa SMDP

### 8.1. Hai thang thời gian

- **Radio tick:** 10 ms; protocol cập nhật measurement, counter, timer và rate.
- **Decision tick:** chỉ xuất hiện khi A3 hoàn tất TTT; khoảng thời gian tới
  decision tiếp theo phụ thuộc trace và action.

`reset()` tiêu thụ phần đầu trace trước decision đầu tiên. Phần này không chịu ảnh
hưởng của action nên reward được lưu riêng là `uncontrolled_reward`. Training
return và `controlled_return` bắt đầu từ decision đầu tiên; protocol metric vẫn
bao phủ toàn trace.

### 8.2. Observation 109 chiều

Observation chỉ dùng hiện tại và lịch sử đã quan sát, không đọc sample tương lai.
19 feature đầu giữ nguyên cho serving và A3 target `C1`:

| Nhóm | Feature |
|---|---|
| Link utility | serving spectral efficiency, target spectral efficiency, delta |
| A3 history | margin hiện tại, mean, min, slope, standard deviation |
| SINR trend | serving slope, target slope |
| RLM risk | serving–Qout gap, target–Qin gap, N310 progress, T310 active/progress |
| Mobility history gián tiếp | serving residence time, time since last HO, target là cell trước hay không, recent HO count |

Mỗi candidate `C2..C10` thêm 10 feature: spectral efficiency, delta so với
serving, A3 margin hiện tại/mean/min/slope/std, SINR slope, Qin gap và cờ previous
cell. Tổng cộng `19 + 9 × 10 = 109` feature.

SINR dùng trong ba spectral-efficiency feature bị clip vào `[-10, 10] dB`; clip
này chỉ thuộc observation. Rate/reward vẫn dùng SINR raw. Observation không chứa
speed, vị trí, hướng bay hoặc độ cao một cách tường minh. Vì vậy có thể gọi nó là
causal finite-history observation/approximate state; chưa có cơ sở khẳng định
vector 109 chiều là trạng thái Markov đầy đủ.

### 8.3. Reward

Reward ở mỗi radio tick:

```text
r[n] = delta_t_seconds * delivered_rate_Mbps / 100
       - 5 * I_RLF_started
       - 1 * I_ping_pong_detected
```

Ý nghĩa scale:

- truyền 100 Mbps trong 1 giây cho tổng rate reward +1;
- một RLF bị trừ 5, tương đương 5 giây truyền ở rate tham chiếu về utility, không
  có nghĩa RLF thật kéo dài 5 giây;
- một ping-pong bị trừ 1;
- HOF không có penalty riêng: HOF gây mất rate và khởi động RLF nên chịu tác động
  gián tiếp;
- không có penalty cố định cho mọi handover.

Macro reward hiện là tổng **không chiết khấu bên trong macro-step**:

```text
R[k] = sum r[n] trong khoảng decision k -> decision k+1/end
tau[k] = số tick * 0.01 giây
Gamma[k] = gamma_per_second ^ tau[k]
```

Đây là một lựa chọn utility hợp lệ nếu được công bố đúng. Nó không hoàn toàn tương
đương với tổng primitive reward được discount ở từng radio tick; muốn tương đương
chặt, từng `r[n]` bên trong macro cũng phải có trọng số theo thời gian tương đối.

### 8.4. Duration-aware GAE

PPO dùng:

```text
delta[k] = R[k] + Gamma[k] * (1-terminated) * V(s[k+1]) - V(s[k])
Lambda[k] = lambda_per_second ^ tau[k]
A[k] = delta[k] + Gamma[k] * Lambda[k]
       * (1-episode_end) * A[k+1]
```

`gamma_per_second=0.99`, `lambda_per_second=0.95`. Termination không bootstrap;
truncation có thể bootstrap residual của chính transition nhưng không truyền
advantage qua episode reset. Trong environment hiện tại macro-step không tự
truncate; trường này tồn tại để giữ contract tổng quát.

## 9. PPO: cấu trúc và cách train

### 9.1. Network

```text
input 109
  -> Linear(109,64) -> Tanh
  -> Linear(64,64) -> Tanh
  -> policy head: 11 logits [STAY, HO_C1..HO_C10]
  -> value head: 1 scalar
```

Hai head dùng chung encoder. Linear layer trong encoder được orthogonal
initialization; policy head được khởi tạo đúng bằng 0 để 11 action hợp lệ có
xác suất ban đầu bằng nhau, value head dùng gain 1,0.
Training sample từ categorical distribution; evaluate/benchmark chọn argmax
deterministic.

Checkpoint 19-feature/2-action hoặc các checkpoint thử nghiệm 3-action không
tương thích với policy head 11 logits hiện tại. Sau thay đổi này cần train model
109-feature/11-action mới từ đầu.

### 9.2. Hyperparameter của run chính

| Tham số | Giá trị |
|---|---:|
| Hidden sizes | 64, 64 |
| Activation | Tanh |
| Learning rate | 3e-4 |
| Rollout | 512 macro-step |
| Minibatch | 64 |
| Update epochs | 10 |
| PPO clip | 0,2 |
| Value coefficient | 0,5 |
| Entropy coefficient | 0,01 |
| Max gradient norm | 0,5 |
| Gamma/second | 0,99 |
| Lambda/second | 0,95 |
| Target KL | 0,03 |
| Advantage normalization | Có |
| Seed PPO | 17 |
| Split seed | 17 |
| Device đã ghi | CUDA |
| Training budget | 300.000 macro-step |

PPO update dùng clipped policy-ratio objective, value MSE, entropy bonus,
gradient clipping và early stop khi approximate KL vượt ngưỡng quy ước của
trainer.

### 9.3. Split, checkpoint và tiêu chí chọn model

Split được thực hiện trên **toàn trajectory**, phân tầng độc lập theo tốc độ, rồi
sắp xếp ổn định. Không cắt một trajectory thành train/test nên tránh leakage trực
tiếp theo thời gian trong cùng route.

Trong run chính:

- train/validation/test = 60/20/20%;
- validation chạy mỗi khoảng 5.000 macro-step, thực tế tại boundary sau rollout;
- `best` được chọn duy nhất theo validation mean controlled return;
- `latest` là trạng thái cuối sau 300.000 macro-step;
- `final` được tạo từ `best`, sau đó gắn final validation và held-out test metric;
- test không tham gia chọn model.

`dataset_split_manifest.json` lưu filename, size và SHA-256 của từng RSRP/SINR.
Evaluate/benchmark dùng manifest cạnh checkpoint để phục hồi đúng held-out split
và từ chối file bị thay đổi.

## 10. Dataset chính và split thực tế

### 10.1. Kiểm kê

| Speed (km/h) | Trace pairs | CSV files | Route IDs | Quãng đường/trace | Tổng quãng đường xấp xỉ |
|---:|---:|---:|---:|---:|---:|
| 30 | 420 | 840 | 0–419 | 500 m | 210,0 km |
| 50 | 255 | 510 | 0–254 | 833,3 m | 212,5 km |
| 70 | 180 | 360 | 0–179 | 1.166,7 m | 210,0 km |
| 90 | 145 | 290 | 0–144 | 1.500 m | 217,5 km |
| 120 | 30 | 60 | 0–29 | 2.000 m | 60,0 km |
| 150 | 30 | 60 | 0–29 | 2.500 m | 75,0 km |
| **Tổng** | **1.060** | **2.120** | — | — | **985,0 km** |

Tổng dung lượng CSV: `4.468.749.214 byte`, xấp xỉ `4,162 GiB`. Mỗi file có
6.000 dòng và 21 cell; ví dụ dòng cuối là sample 5.999 tại 59.990 ms. Tổng thời
lượng quy ước là 63.600 giây, tương đương 17 giờ 40 phút.

Số route 30–90 giảm khi tốc độ tăng không phải cân bằng số trajectory, nhưng làm
tổng quãng đường mỗi speed gần 210–217,5 km. Tuy vậy, sampler huấn luyện chọn theo
trajectory/episode và macro-event, nên cân bằng quãng đường nguồn không tự bảo đảm
các speed đóng góp bằng nhau vào gradient.

### 10.2. Split của checkpoint

| Speed | Tổng development pool | Train split | Validation | Test cùng miền | Train thực sự đã dùng |
|---:|---:|---:|---:|---:|---:|
| 30 | 420 | 252 | 84 | 84 | 67 |
| 50 | 255 | 153 | 51 | 51 | 49 |
| 70 | 180 | 108 | 36 | 36 | 30 |
| 90 | 145 | 87 | 29 | 29 | 16 |
| **Tổng 30–90** | **1.000** | **600** | **200** | **200** | **162** |
| 120 | 30 | 0 | 0 | 0 | 0 |
| 150 | 30 | 0 | 0 | 0 | 0 |

Toàn bộ 60 trace 120/150 km/h được giữ ngoài train, validation và checkpoint
selection. Benchmark gọi đây là zero-shot theo **vận tốc**. Điều đó không chứng
minh zero-shot theo topology/channel distribution vì các trace vẫn thuộc cùng họ
mô phỏng.

### 10.3. Coverage thật trong training

Log ghi 300.000 macro-step và 10.352 completed episode nhưng chỉ có 162 trace ID
khác nhau:

| Speed | Unique train trace đã dùng | Macro-step | Tỉ lệ macro-step | Completed episodes |
|---:|---:|---:|---:|---:|
| 30 | 67 | 92.063 | 30,69% | 4.100 |
| 50 | 49 | 89.966 | 29,99% | 3.119 |
| 70 | 30 | 61.157 | 20,39% | 1.935 |
| 90 | 16 | 56.814 | 18,94% | 1.198 |

438/600 train trace không có macro-transition nào trong log. Đây không phải chỉ
là báo cáo thiếu: lịch sử segment cho thấy sau validation, chuỗi lấy mẫu bắt đầu
lại từ cùng trace `30kmh_210`.

Nguyên nhân code:

1. `collect_rollout()` seed environment bằng seed 17 khi collector state trống.
2. `evaluate()` luôn xóa `_collector_env_id` và `_collector_observation`, dù nó
   vừa đánh giá một environment khác.
3. Lần training collection tiếp theo thấy collector trống và seed lại train env
   bằng đúng 17.
4. Validation định kỳ trước khi permutation ngẫu nhiên đi đủ xa làm phần đầu chuỗi
   train bị lặp.

Trạng thái sửa hiện tại: `evaluate()` chỉ vô hiệu hóa collector khi đánh giá đúng
chính environment mà collector đang dùng. Validation environment riêng không còn
xóa observation/RNG của train. Hai regression test khóa cả hai trường hợp. Các số
162/600 ở trên vẫn mô tả run cũ và chỉ biến mất sau khi train lại.

## 11. Kết quả của PPO run chính

### 11.1. Diễn biến train

| Mốc | Giá trị |
|---|---:|
| Tổng macro-step | 300.000 |
| PPO updates | 586 |
| Completed training episodes | 10.352 |
| Số lần validation | 60 |
| Mean train episode return ở update 1 | -10,289 |
| Mean train episode return ở update cuối | +2,602 |
| Validation return lần đầu (5.120 step) | -4,888 |
| Validation return tốt nhất | -3,192 |
| Best checkpoint | 190.464 step, update 372 |
| Validation return ở update cuối | -4,143 |

Train return tăng không đồng nghĩa model cuối tốt hơn trên validation. Run đã chọn
đúng best checkpoint thay vì latest, vì validation tốt nhất xuất hiện trước khi
training kết thúc.

### 11.2. Final validation và held-out test

| Metric | Validation 200 trace | Test 200 trace |
|---|---:|---:|
| Mean controlled return | -3,192 | -4,996 |
| Std return | 11,715 | 12,800 |
| Mean rate reward | 12,183 | 11,814 |
| Mean RLF penalty | -13,500 | -14,825 |
| Mean ping-pong penalty | -1,875 | -1,985 |
| Mean macro-step/trace | 29,20 | 30,11 |
| HANDOVER action fraction | 57,67% | 57,22% |
| Mean estimated rate | 22,226 Mbps | 21,423 Mbps |
| Total RLF | 540 | 593 |
| Total ping-pong | 375 | 397 |
| Trace không có decision | 0 | 0 |

Các metric test trong `training_report.json` khớp với hàng PPO của benchmark trên
đúng 200 test trace 30–90 km/h.

## 12. Benchmark chính

### 12.1. Cách benchmark

Hai policy chạy deterministic trên cùng trajectory, cùng preprocessing, cùng
protocol engine và cùng reward:

- `3gpp_always_ho`: luôn chấp nhận mọi A3 decision;
- `ppo`: argmax action từ selected checkpoint.

Metric được định nghĩa:

```text
Relative rate (%) = 100 * achieved estimated rate / per-trace best-cell rate
HOF (%)           = 100 * total failed HO executions / total HO attempts
RLF time (%)      = 100 * total RLF-recovery time / total trace time
PP (%)            = 100 * total A-B-A returns / total successful HO
```

### 12.2. Kết quả gộp 260 trace

Tập này gồm 200 held-out test trace 30–90 km/h và 60 trace zero-shot 120/150
km/h, tổng thời gian 15.600 giây.

| Policy | Return ↑ | Rate (Mbps) ↑ | Relative rate ↑ | Connected ↑ | HO action | HOF ↓ | RLF time ↓ | PP ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A3 Always-HO | -25,819 | 22,179 | 89,823% | 95,638% | 100% | 12,182% | 1,649% | 37,670% |
| **PPO** | **-7,584** | 20,890 | 83,765% | 97,552% | 58,331% | **8,643%** | **1,116%** | **13,518%** |

So sánh ghép cặp PPO với A3 Always-HO trên đúng 260 trace:

| Đại lượng | PPO − A3 |
|---|---:|
| Mean estimated rate | -1,289 Mbps |
| Mean controlled return | +18,234 |
| Tổng RLF | -417 sự kiện |
| Tổng ping-pong | -2.857 sự kiện |
| Tỉ lệ trace PPO thắng về rate | 5,00% |
| Tỉ lệ trace PPO thắng về return | 86,92% |
| Tỉ lệ trace PPO có ít RLF hơn | 59,23% |
| Tỉ lệ trace PPO có ít ping-pong hơn | 98,08% |

Kết luận đúng từ bảng này là PPO học được một **trade-off rate–reliability phù
hợp với reward đã đặt**. Không nên viết PPO vượt A3 trên mọi tiêu chí: rate của
PPO thấp hơn A3 trên trung bình và chỉ thắng rate ở 5% cặp trace.

### 12.3. Test cùng miền và zero-shot

| Tập | Policy | Return | Relative rate | HOF | RLF time | PP |
|---|---|---:|---:|---:|---:|---:|
| Test 30–90, 200 trace | A3 Always-HO | -19,370 | 91,131% | 10,648% | 1,294% | 38,145% |
| Test 30–90, 200 trace | **PPO** | **-4,996** | 85,120% | **7,594%** | **0,987%** | **12,512%** |
| Zero-shot 120–150, 60 trace | A3 Always-HO | -47,313 | 85,463% | 15,620% | 2,830% | 36,542% |
| Zero-shot 120–150, 60 trace | **PPO** | **-16,211** | 79,248% | **10,694%** | **1,547%** | **15,551%** |

Trên test cùng miền, PPO mất 6,012 điểm phần trăm relative rate so với A3, đổi
lại HOF giảm 28,68% tương đối, RLF recovery time giảm 23,72% tương đối và PP giảm
67,20% tương đối. Trên zero-shot, PPO vẫn giữ hướng trade-off tương tự nhưng mọi
policy đều xấu đi khi tốc độ tăng.

### 12.4. Theo từng tốc độ

| Speed | Policy | Return | Relative rate | HOF | RLF time | PP |
|---:|---|---:|---:|---:|---:|---:|
| 30 | A3 | -11,396 | 92,794% | 9,290% | 0,924% | 38,275% |
| 30 | PPO | **-1,322** | 86,847% | **6,377%** | **0,813%** | **10,365%** |
| 50 | A3 | -17,436 | 92,241% | 10,316% | 1,212% | 38,837% |
| 50 | PPO | **-2,464** | 87,112% | **6,909%** | **0,884%** | **10,807%** |
| 70 | A3 | -28,153 | 88,219% | 10,887% | 1,648% | 36,082% |
| 70 | PPO | **-10,619** | 80,668% | **8,158%** | **1,187%** | **14,324%** |
| 90 | A3 | -34,965 | 87,980% | 13,285% | 2,074% | 39,478% |
| 90 | PPO | **-13,110** | 82,139% | **9,570%** | **1,425%** | **15,798%** |
| 120 | A3 | -42,922 | 86,161% | 14,547% | 2,567% | 35,841% |
| 120 | PPO | **-14,983** | 79,522% | **9,485%** | **1,489%** | **13,713%** |
| 150 | A3 | -51,704 | 84,765% | 16,637% | 3,093% | 37,223% |
| 150 | PPO | **-17,440** | 78,975% | **11,836%** | **1,604%** | **17,337%** |

Trong artifact hiện tại, PPO có return và ba metric reliability tốt hơn A3 ở cả
sáu speed, nhưng relative rate thấp hơn ở cả sáu. Handover action fraction của
PPO tăng từ khoảng 53,9% ở 30 km/h lên quanh 60% ở các speed cao; policy không
biến thành chiến lược luôn từ chối handover.

## 13. Xem output model PPO ở đâu?

### 13.1. Model và log huấn luyện

Thư mục chính:

```text
results/models/uav_100m_grid100m_ppo_30_90/
```

| File | Cách hiểu |
|---|---|
| `smdp_ppo_best.pt` | Checkpoint có validation mean return tốt nhất |
| `smdp_ppo_latest.pt` | Trạng thái ở cuối 300.000 macro-step; không nhất thiết tốt nhất |
| `smdp_ppo_final.pt` | Bản model nên dùng để inference/benchmark; được copy từ best và gắn metric final |
| `training_report.json` | Báo cáo tổng: config PPO, split, history, validation, test, đường dẫn checkpoint |
| `training_metrics.jsonl` | Log theo sự kiện/update, phù hợp để vẽ learning curve hoặc debug |
| `dataset_split_manifest.json` | Split chính xác và SHA-256 từng file |
| `dataset_split_manifest.csv` | Cùng manifest ở dạng bảng dễ mở |

Nếu chỉ cần biết kết quả cuối, đọc `training_report.json`, hai key `validation` và
`test`. Nếu cần dùng model, chọn `smdp_ppo_final.pt`, không mặc định chọn
`latest.pt`.

> Ghi chú phiên bản: bảng trên mô tả artifact huấn luyện cũ. Pipeline mới lưu
> thêm `smdp_ppo_best_return.pt`, `smdp_ppo_best_qos.pt`,
> `smdp_ppo_best_rlf.pt`, toàn bộ checkpoint validation trong
> `checkpoints/all/`, và Pareto front trong `checkpoints/pareto/`. Model `final`
> mới ưu tiên `best_qos` đạt sàn rate so với 3GPP Always-HO; nếu không có model
> đạt sàn mới fallback về `best_return`. Chi tiết lựa chọn nằm trong
> `checkpoints/validation_metrics.csv` và `checkpoints/pareto_front.json`.

### 13.2. Benchmark và hình

Thư mục chính:

```text
results/benchmark/uav_100m_grid100m_all_speeds_percent/
```

| File | Cách hiểu |
|---|---|
| `uav_100m_grid100m_benchmark_by_speed_percent.png` | Hình xem nhanh kết quả theo speed |
| `uav_100m_grid100m_benchmark_by_speed_percent.svg` | Bản vector để đưa vào paper |
| `aggregate.csv` | Tổng hợp cả 260 trace theo policy |
| `by_speed.csv` | Metric chi tiết theo policy và speed |
| `paired_comparison.csv` | Delta PPO/Stay so với A3 trên trace ghép cặp |
| `per_trajectory.csv` | Mức chi tiết cao nhất, một hàng/policy/trajectory |
| `summary.json` | Metadata checkpoint, selection và toàn bộ bảng ở dạng JSON |

Thứ tự đọc khuyến nghị: hình PNG → `aggregate.csv` → `by_speed.csv` →
`paired_comparison.csv` → `per_trajectory.csv` khi cần truy nguyên outlier.

Các thư mục model/benchmark khác trong `results/` là run cũ hoặc thí nghiệm khác
(300 m, 60 route/speed, locked-test, terrestrial hoặc smoke integration). Không
trộn số liệu giữa các thư mục nếu chưa ghi rõ dataset, checkpoint, split và seed.

## 14. Những điểm chưa hợp lý hoặc chưa đủ bằng chứng

### 14.1. Mức bắt buộc sửa trước khi công bố kết quả cuối

| Vấn đề | Bằng chứng | Ảnh hưởng | Cách xử lý |
|---|---|---|---|
| Run cũ chỉ dùng 162/600 train trace | `training_trace_usage`; collector từng seed lại sau validation | Coverage thấp hơn mô tả, lặp dữ liệu, có thể bias model | **Source đã sửa và có regression test**; còn phải retrain để có artifact mới |
| Dataset chính thiếu generator và dataset metadata | Không có hai file JSON cạnh CSV | Không chứng minh được carrier/bandwidth/dt và mã sinh trace | Xác minh nguồn; prepare lại vào thư mục mới với tham số explicit hoặc regenerate có manifest |
| Checkpoint/result cũ không lưu protocol config | `protocol_config` vắng/null trong artifact | Không biết chắc rule/radio config thực tế của run | Không sửa artifact cũ; tạo run mới lưu config đầy đủ |
| Code và artifact không được gắn version | Artifact 14/08, benchmark 27/08; working tree hiện dirty | Khó tái tạo chính xác implementation đã sinh số liệu | Commit/tag code, lưu commit hash và CLI/env trong report của mỗi run |

Không nên retrain đè lên thư mục cũ. Giữ nguyên artifact hiện tại để đối chiếu và
tạo tên dataset/model version mới.

### 14.2. Cấu hình và contract cần làm chặt hơn

1. **`prepare_data` với legacy trace:** hiện dùng shared default khi không có
   manifest. Nên bắt người dùng truyền carrier/bandwidth/dt hoặc ghi rõ provenance
   là `assumed_current_defaults`, không phải `declared`.
2. **Đã sửa cho dataset mới:** mọi batch generator ghi thẳng CSV train-ready, lưu
   entrypoint/version, toàn bộ config, seed rule, hash cây source và hash từng
   output CSV. Không còn bước `prepare_data` sau generation. Dataset lịch sử không
   được backfill vì không thể chứng minh provenance.
3. **Reward/observation config chưa lưu:** checkpoint mới lưu `protocol_config` và
   PPO config, nhưng `HandoverRewardConfig` và `A3ObservationConfig` vẫn dựa vào
   default lúc evaluate. Nếu default thay đổi, nghĩa model cũ thay đổi theo.
4. **`l3_k` và `l3_filter_w` có thể lệch:** weight được tính tại thời điểm class
   definition. Gọi `Config.update({'l3_k': ...})` không tự tính lại weight. Nên chỉ
   lưu K và tạo weight bằng property, hoặc kiểm tra tính nhất quán.
5. **`clip_sinr` không có tác dụng:** observation luôn clip SINR bất kể flag này.
   Cần bỏ flag hoặc thực thi đúng ý nghĩa.
6. **TTT và bước mẫu:** validator chấp nhận các giá trị TS 38.331 như 64/128 ms,
   sau đó lại từ chối vì không chia hết cho dt=10 ms. Đây là constraint của
   simulator, nhưng error/config API cần diễn đạt rõ hoặc hỗ trợ timer theo thời
   gian thay vì integer tick.
7. **Tài liệu reward lệch code:** preparation vẫn có data, execution mới outage.
   Docstring/tài liệu phải cập nhật để tránh viết sai mô hình.

### 14.3. Giới hạn khoa học cần công bố

- Baseline là A3-based/3GPP-inspired, không phải full compliant NR stack.
- Qin/Qout là SINR proxy; không có BLER/PDCCH evaluation period chuẩn.
- Rate là Shannon estimate, không phải throughput thực tế.
- Observation là xấp xỉ trạng thái; không có speed/position/direction explicit.
- Chỉ có một training seed; chưa có confidence interval giữa các seed.
- Benchmark hiện chỉ dùng A3 Always-HO; chưa có tuned A3/TTT, heuristic gate hoặc
  controller tối ưu có cùng độ phức tạp.
- Chưa có ablation cho duration-aware discount, nhóm RLM feature và HO-history.
- Zero-shot chỉ theo speed, chưa theo map, altitude, carrier, cell density hoặc
  channel realization family.
- HOF giảm một phần vì PPO thực hiện ít HO hơn. Cần báo đồng thời HO attempts,
  HOF ratio, tổng RLF và RLF time, không chỉ một metric.
- Macro reward không discount primitive reward bên trong khoảng; checkpoint lại
  được chọn bằng tổng controlled return không discount. Cần mô tả đúng hoặc đồng
  bộ objective nếu muốn claim tương đương theo thời gian liên tục.

## 15. Trạng thái kiểm thử

Lệnh đã chạy:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Kết quả: **122 test qua, 0 fail, 0 error**, thời gian 26,375 giây. Test bao phủ:

- cấu trúc/layer dependency;
- CSV loader, L3 và split;
- reference A3/RLM/RLF;
- action-controlled engine, target latch và event suppression;
- observation, reward, macro-transition và Gym environment;
- duration-aware GAE, PPO update, save/load checkpoint và bảo toàn collector train
  khi validation dùng environment riêng;
- benchmark aggregation và paired comparison;
- converter/integration pipeline;
- antenna, LoS state, shadowing và small-scale fading của generator;
- contract generator chỉ xuất RSRP/SINR, shared radio defaults 2,1 GHz/10 MHz/10 ms,
  manifest có source hash và chặn gắn provenance lên CSV lịch sử chưa kiểm chứng.

`git diff --check` không phát hiện whitespace error; có cảnh báo line-ending
LF→CRLF của working tree Windows.

Test qua chứng minh các contract code đang được kiểm tra, không tự chứng minh mô
hình kênh đúng với một deployment thực, hyperparameter tối ưu, hoặc kết quả có ý
nghĩa thống kê trên nhiều seed.

## 16. Cách chạy pipeline sạch cho thí nghiệm tiếp theo

### 16.1. Cài đặt

```powershell
cd C:\Project\HandoverOptimDRL
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe run.py --help
```

Yêu cầu hiện tại: Python ≥3.12, Gymnasium, NumPy, SciPy, PyTorch, pandas và
Matplotlib theo `requirements.txt`.

### 16.2. Sinh dataset train-ready mới

Ví dụ generic pipeline có manifest:

```powershell
.\.venv\Scripts\python.exe run.py generate_data `
  --speeds 30 50 70 90 120 150 `
  --routes-per-speed 60 `
  --grid-spacing 100 `
  --altitude 100 `
  --steps 6000 `
  --workers 4 `
  --output-dir data/uav_grid100_v3
```

Nếu cần đúng số route không đều 420/255/180/145/30/30, dùng preset:

```powershell
.\.venv\Scripts\python.exe -m uav_simulation batch-grid100 `
  --workers 4 `
  --output-dir data/uav_100m_grid100m_v3
```

Sau khi lệnh báo `Batch complete` và in đường dẫn `Training metadata`, thư mục này
đã dùng trực tiếp được cho PPO.

### 16.3. Chuyển dữ liệu legacy (không dùng cho output generator mới)

Với raw legacy chỉ khi đã xác nhận chính xác cấu hình nguồn:

```powershell
.\.venv\Scripts\python.exe run.py prepare_data `
  --input-dir data/UAV_data_100m_grid100m `
  --output-dir data/uav_grid100_legacy_declared_v2 `
  --carrier-ghz 2.1 `
  --bandwidth-mhz 10 `
  --delta-t-ms 10
```

Tham số trên chỉ **khai báo provenance**, không biến đổi lại giá trị radio. Không
gán bất kỳ bộ tham số nào cho legacy data nếu chưa xác nhận cấu hình nguồn.

### 16.4. Train

Sau khi sửa lỗi collector:

```powershell
.\.venv\Scripts\python.exe run.py train_ppo `
  --data-dir data/uav_100m_grid100m_v3 `
  --speeds 30 50 70 90 `
  --train-fraction 0.60 `
  --validation-fraction 0.20 `
  --test-fraction 0.20 `
  --split-seed 17 `
  --seed 17 `
  --total-macro-steps 300000 `
  --device cuda `
  --output-dir results/models/uav_grid100_v2_ppo_30_90_seed17
```

Các tham số PPO không xuất hiện trong lệnh sẽ lấy default đúng bảng ở mục 9.2.
Mỗi seed phải dùng output directory riêng.

Lệnh đã tạo run lịch sử không được lưu nguyên văn. Lệnh trên là bản dựng lại từ
metadata artifact và default code, không phải bằng chứng exact command của ngày
14/08/2026.

### 16.5. Evaluate và benchmark

```powershell
.\.venv\Scripts\python.exe run.py evaluate_ppo `
  --data-dir data/uav_100m_grid100m_v3 `
  --checkpoint results/models/uav_grid100_v2_ppo_30_90_seed17/smdp_ppo_final.pt `
  --split test `
  --output results/evaluation/uav_grid100_v2_seed17_test.json
```

```powershell
.\.venv\Scripts\python.exe run.py benchmark `
  --data-dir data/uav_100m_grid100m_v3 `
  --checkpoint results/models/uav_grid100_v2_ppo_30_90_seed17/smdp_ppo_final.pt `
  --split test `
  --speeds 30 50 70 90 `
  --zero-shot-speeds 120 150 `
  --device cuda `
  --output-dir results/benchmark/uav_grid100_v2_all_speeds_seed17
```

Benchmark sẽ dùng exact test manifest của checkpoint cho 30–90 và toàn bộ trace
120/150 cho zero-shot. Cần bổ sung hash manifest riêng cho zero-shot trong phiên
bản sau vì checkpoint manifest hiện chỉ fingerprint pool 30–90.

## 17. Kế hoạch sửa và tái chạy đề xuất

Thứ tự nên thực hiện:

1. **Đã làm trong source hiện tại:** `SMDPPPO.evaluate()` không xóa collector train
   khi dùng validation environment riêng; regression test kiểm tra cả validation
   riêng và evaluate trực tiếp trên train environment.
2. Bổ sung assertion/report: unique train traces, coverage %, macro-step/speed và
   cảnh báo nếu coverage chưa đạt ngưỡng.
3. Xác nhận cấu hình thật của raw grid100. Nếu không thể xác nhận, regenerate vào
   dataset v2 có manifest thay vì gán ngược config hiện tại.
4. Lưu đầy đủ protocol, reward, observation, generator config, version package,
   Python/PyTorch/CUDA và Git commit vào checkpoint/report.
5. Chạy lại tối thiểu nhiều seed độc lập; báo mean, standard deviation hoặc
   confidence interval trên metric trajectory-level và seed-level.
6. Thêm tuned A3/TTT baseline, heuristic threshold/gate và ablation:
   duration-aware so với fixed discount, bỏ RLM features, bỏ HO-history.
7. Giữ nguyên artifact lịch sử, đặt tên v2 rõ ràng và chỉ thay hình/bảng paper sau
   khi benchmark mới hoàn tất.

## 18. Bộ tối thiểu cần archive

Để người khác chỉ mở archive là hiểu và có thể kiểm chứng, cần giữ cùng nhau:

```text
docs/BAO_CAO_LUU_TRU_DU_AN.md
source snapshot hoặc Git bundle chứa toàn bộ working tree đã commit
requirements.txt + pyproject.toml + setup.py
data/uav_100m_grid100m_v3/                  # CSV train-ready + hai file metadata
results/models/uav_100m_grid100m_ppo_30_90/
results/benchmark/uav_100m_grid100m_all_speeds_percent/
```

Nên bổ sung vào archive mới:

- `simulation_manifest.json` và `dataset_metadata.json` cạnh CSV train-ready;
- exact CLI/stdout của generation, legacy conversion (nếu có), training và benchmark;
- `git rev-parse HEAD`, trạng thái dirty và patch nếu chưa thể commit;
- `python --version`, `pip freeze`, PyTorch/CUDA/GPU information;
- hash của zero-shot files và toàn bộ figure/table xuất bản.

Nếu không thể archive 4,162 GiB dataset, tối thiểu phải giữ split manifest, hash
toàn dataset, nơi lưu bản gốc có version bất biến và script tải/khôi phục. Chỉ giữ
checkpoint không đủ vì environment cần trace để evaluate.

## 19. Kết luận cuối

Kiến trúc hiện tại đã có ranh giới đúng: generator tạo radio trace, còn mọi cơ chế
kết nối/handover nằm trong HandoverOptimDRL. SMDP event-triggered, observation
nhân quả, reward phân rã và benchmark chung engine là nền tảng tốt để nghiên cứu
trade-off rate–reliability.

Artifact hiện tại cho thấy PPO giảm rõ HOF, RLF time và ping-pong, đồng thời chấp
nhận giảm estimated rate so với A3 Always-HO. Kết quả này phù hợp với reward và
giữ được xu hướng ở 120/150 km/h. Tuy nhiên, artifact cũ vẫn mang lỗi coverage
162/600 và dataset–checkpoint vẫn thiếu provenance/config; vì vậy không nên coi
các con số hiện tại là kết quả cuối để công bố. Source đã sửa collector và thống
nhất default, nhưng phải regenerate/prepare có metadata, retrain và chạy nhiều seed
thì project mới có chuỗi bằng chứng tái lập đủ mạnh.

## 20. Các file nguồn quan trọng để truy nguyên

- Generator config: `data_generator/src/uav_simulation/config.py`
- Generator loop: `data_generator/src/uav_simulation/simulation.py`
- Radio formulas: `data_generator/src/uav_simulation/radio_models.py`
- Raw CSV writer: `data_generator/src/uav_simulation/logger.py`
- Grid100 batch: `data_generator/src/uav_simulation/batch/generate_batch_grid100_h100.py`
- Generic manifest generator: `data_generator/src/uav_simulation/batch/generate_dataset.py`
- Prepare/converter: `src/ho_optim_drl/commands/prepare_uav_data.py`,
  `src/ho_optim_drl/commands/convert_uav_csv.py`
- Runtime config: `src/ho_optim_drl/config.py`
- Loader/split: `src/ho_optim_drl/data/loader.py`, `src/ho_optim_drl/data/split.py`
- Protocol engine/reference: `src/ho_optim_drl/protocol/engine.py`,
  `src/ho_optim_drl/protocol/reference.py`
- Observation/reward/transition: `src/ho_optim_drl/smdp/`
- Gym environment: `src/ho_optim_drl/envs/handover.py`
- PPO: `src/ho_optim_drl/agents/ppo/`
- Training command: `src/ho_optim_drl/commands/train_smdp_ppo.py`
- Benchmark command/aggregation: `src/ho_optim_drl/commands/benchmark_policies.py`,
  `src/ho_optim_drl/evaluation/benchmark.py`
