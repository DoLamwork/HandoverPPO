# UAV data generator

Thành phần tạo dữ liệu thuộc **HandoverOptimDRL**. Mã tại
`src/uav_simulation/` chỉ mô phỏng UAV, topology, anten, kênh truyền và xuất CSV.
Train/evaluate PPO nằm ở `../src/ho_optim_drl/` và được chạy bởi `../run.py`.

Data generator không chọn serving cell, không duy trì trạng thái kết nối và
không thực hiện A3/TTT/handover/RLF. Đầu ra duy nhất phục vụ pipeline là ma trận
RSRP và SINR của mọi cell tại mỗi mẫu; toàn bộ cơ chế kết nối và handover thuộc
`../src/ho_optim_drl/`.

Cài đặt một lần từ thư mục HandoverOptimDRL bằng `python -m pip install -e .`.
Thư mục này không có Git repository hay virtual environment riêng.

Chạy từ thư mục HandoverOptimDRL:

```powershell
.\.venv\Scripts\python.exe run.py generate_data --help
.\.venv\Scripts\python.exe data_generator/main.py
.\.venv\Scripts\python.exe -m uav_simulation heatmap --data-dir data/UAV_data_100m --speed 30
.\.venv\Scripts\python.exe -m uav_simulation plot-antenna
```

CSV batch train-ready được lưu dưới `../data/` khi chạy từ project chính. Biểu đồ mặc định nằm
trong `figures/uav_simulation/`, còn thống kê kênh nằm trong
`results/channel_validation/`. Mọi đường dẫn tương đối tính từ working directory.

`python -m uav_simulation` vẫn có simulate, batch, batch-300m, batch-grid100,
route, heatmap, plot-antenna, plot-system và validate. Mọi lệnh batch ghi schema
`sample,time_ms,bs_0,...` và tên `{rsrp|sinr}_{speed}kmh_{route}.csv` mà PPO đọc
trực tiếp. Chế độ GUI/single-run vẫn giữ output legacy nhiều UE để phục vụ khảo sát.

Profile dùng cho dataset grid100 chính có generator riêng:

```powershell
.\.venv\Scripts\python.exe -m uav_simulation batch-grid100 --workers 4 --output-dir data/uav_100m_grid100m_v3
.\.venv\Scripts\python.exe run.py train_ppo --data-dir data/uav_100m_grid100m_v3 --speeds 30 50 70 90 --output-dir results/models/uav_grid100_v3_ppo
```

Trước khi ghi CSV, generator tạo `simulation_manifest.json` chứa cấu hình đầy đủ,
entrypoint, seed rule và SHA-256 source. Sau khi kiểm tra đủ mọi cặp file/schema/số
dòng, nó ghi `dataset_metadata.json` với radio config và SHA-256 từng CSV. Nếu thư
mục đã có CSV nhưng không có manifest, lệnh dừng thay vì gắn nhãn nguồn sinh không
thể kiểm chứng. `prepare_data` giờ chỉ cần cho CSV legacy `ue0_*`, `Step,bs0,...`.

Tài liệu mô hình: [anten](docs/antenna_3gpp_tr38901.md),
[LoS/NLoS](docs/spatial_los_nlos.md), [small-scale fading](docs/small_scale_fading.md).
