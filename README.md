# HandoverOptimDRL

Project chính để train và validate PPO/SMDP cho tối ưu handover, so sánh với
baseline 3GPP A3. **UAV simulation là thành phần tạo dữ liệu**, đặt trong
`data_generator/` của project này.

`data_generator/` chỉ sinh RSRP/SINR của mọi cell theo thời gian. Nó không chọn
serving cell và không xử lý kết nối, A3/TTT, handover hay RLF; các cơ chế đó chỉ
nằm trong `src/ho_optim_drl/`.

## Cấu trúc

```text
HandoverOptimDRL/
|-- run.py                           # Lệnh chạy chính
|-- setup.py / pyproject.toml
|-- src/ho_optim_drl/                 # Protocol, environment, PPO, evaluation
|-- scripts/                         # Tương thích các công cụ cũ
|-- data_generator/
|   |-- main.py                      # Mở simulator
|   |-- src/uav_simulation/          # Mobility, anten, radio, CSV, batch, biểu đồ
|   |-- tests/
|   `-- docs/
|-- data/                            # Dữ liệu dùng chung
|   |-- UAV_data_100m/               # CSV thô UAV hiện có
|   |-- UAV_data_300m/
|   |-- UAV_data_100m_grid100m/
|   `-- uav_generated/               # CSV chuẩn hóa và metadata cho luồng mới
|-- results/
|   |-- models/                      # Checkpoint PPO
|   |-- benchmark/                   # So sánh PPO / 3GPP Always-HO
|   `-- 3gpp/                        # Kết quả A3 reference
|-- tests/                           # Test handover và tích hợp
`-- docs/
```

Không có bản sao PPO bên trong `data_generator/`. Một lần cài đặt tại thư mục
HandoverOptimDRL cài cả package chính và thành phần tạo dữ liệu.

## Cài đặt

Python 3.12 trở lên. Tất cả lệnh sau chạy từ `C:\Project\HandoverOptimDRL`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe run.py --help
```

Nếu `.venv` đã hoạt động, dùng lại môi trường đó. `handover-ppo` tương đương
`python run.py` sau khi kích hoạt môi trường. `uav-sim` chỉ cung cấp công cụ
simulator. Trên macOS/Linux dùng `.venv/bin/python`.

## Tạo dữ liệu dùng trực tiếp để train

Dữ liệu thô có sẵn nằm trong `data/UAV_data_100m`, `data/UAV_data_300m` và
`data/UAV_data_100m_grid100m`. Bộ 360 quỹ đạo đã chuẩn hóa từ bước tích hợp trước
được đặt tại `data/uav_generated`; nếu thư mục này đã có dữ liệu, có thể train ngay.

Để tạo một thí nghiệm mới:

```powershell
.\.venv\Scripts\python.exe run.py generate_data --speeds 30 50 70 90 --routes-per-speed 60 --grid-spacing 100 --workers 4 --output-dir data/uav_new
.\.venv\Scripts\python.exe run.py train_ppo --data-dir data/uav_new --speeds 30 50 70 90 --output-dir results/models/uav_new_ppo
```

Để tái tạo đúng profile grid100 của dataset chính (1.060 quỹ đạo với số route
420/255/180/145/30/30 theo vận tốc), sinh thẳng vào thư mục dataset mới:

```powershell
.\.venv\Scripts\python.exe -m uav_simulation batch-grid100 --workers 4 --output-dir data/uav_100m_grid100m_v3
.\.venv\Scripts\python.exe run.py train_ppo --data-dir data/uav_100m_grid100m_v3 --speeds 30 50 70 90 --output-dir results/models/uav_grid100_v3_ppo
```

Hoặc chuẩn hóa dữ liệu thô cũ vào một thư mục mới, trống:

```powershell
.\.venv\Scripts\python.exe run.py prepare_data --input-dir data/UAV_data_100m --output-dir data/uav_generated
```

Generator batch mặc định tạo 6000 mẫu/quỹ đạo, mỗi mẫu cách nhau 10 ms và một UAV
trên mỗi route. Nó ghi luôn tên file `{rsrp|sinr}_{speed}kmh_{route}.csv`, schema
`sample,time_ms,bs_0,...`, `simulation_manifest.json` và `dataset_metadata.json`
kèm hash từng CSV. Vì vậy output mới đưa thẳng vào `train_ppo`, không còn bước
`prepare_data`. Command `prepare_data` chỉ giữ lại để chuyển các bộ legacy có tên
`ue0_*` và schema `Step,bs0,...`; các lệnh không ghi đè dữ liệu cũ.

## Train PPO và validate với 3GPP

```powershell
.\.venv\Scripts\python.exe run.py train_ppo --data-dir data/uav_generated --speeds 30 50 70 90 --output-dir results/models/uav_generated_ppo
.\.venv\Scripts\python.exe run.py evaluate_ppo --data-dir data/uav_generated --checkpoint results/models/uav_generated_ppo/smdp_ppo_final.pt --split validation --output results/evaluation/uav_validation.json
.\.venv\Scripts\python.exe run.py benchmark --data-dir data/uav_generated --checkpoint results/models/uav_generated_ppo/smdp_ppo_final.pt --split test --output-dir results/benchmark/uav_generated_ppo
```

Train mặc định 100000 macro-step; chia toàn bộ quỹ đạo thành train/validation/test
70/15/15, phân tầng theo tốc độ. Trước khi train, pipeline chạy 3GPP Always-HO
trên validation split để làm mốc QoS. Mỗi lần validation đều được lưu trong
`checkpoints/all`; `checkpoints/pareto` giữ các model chưa bị model khác vượt đồng
thời về rate, RLF time, HOF, ping-pong và HO attempts. `smdp_ppo_final.pt` ưu tiên
`best_qos` đạt tối thiểu 98% rate 3GPP ở toàn tập và từng tốc độ, rồi mới fallback
sang `best_return`. Có thể đổi hai sàn bằng `--qos-rate-ratio` và
`--qos-rate-ratio-per-speed`. Mỗi lần train chọn một thư mục model mới. Giữ
`dataset_split_manifest.json` cạnh checkpoint:
evaluate/benchmark sẽ dùng đúng tập đã lưu và kiểm tra hash thay vì chia lại dữ liệu.

Chạy riêng baseline 3GPP:

```powershell
.\.venv\Scripts\python.exe run.py validate_3gpp --data-dir data/uav_generated --output-dir results/3gpp/uav_generated
```

PPO và baseline dùng cùng cấu hình radio từ metadata dataset. Mặc định dùng chung
của generator và handover environment là 2,1 GHz, 10 MHz, 10 ms. Dữ liệu và model
cũ trong `data/` và `results/` được giữ nguyên; cấu hình thật của dataset cũ không
có metadata vẫn phải được xác minh từ nguồn sinh dữ liệu.

## Kiểm thử và tài liệu

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Lệnh này chạy cả test của HandoverOptimDRL, test tích hợp và test simulator trong
`data_generator/tests/`.

- [Thành phần tạo dữ liệu UAV](data_generator/README.md)
- [Chi tiết tích hợp vào project chính](docs/uav_data_generator.md)
- [PPO](docs/ppo.md), [benchmark](docs/benchmark.md), [kiến trúc](docs/architecture.md)
- [Phạm vi và giả thiết baseline 3GPP](docs/3gpp_baseline_audit.md)

Baseline gồm các quy tắc A3/TTT/RLM/RLF được kiểm tra theo TS 38.331 cùng những giả
thiết được ghi rõ; đây không phải triển khai đầy đủ mọi thủ tục NR. Anten simulator
sử dụng các công thức TR 38.901.
