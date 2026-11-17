# T6A: PPO được quyết định ngay khi A3 vừa vào

## Mục tiêu

T6A kiểm tra riêng giả thuyết: cho PPO chọn `HO_C1` hoặc `STAY` ngay tại tick
đầu tiên thỏa điều kiện vào A3 chuẩn có tốt hơn việc chỉ quyết định sau khi TTT
40 ms hoàn tất hay không.

T6A **không hạ ngưỡng A3**. Với cấu hình hiện tại, điều kiện vào vẫn là:

```text
RSRP_target - RSRP_serving > a3_offset + a3_hysteresis = 2 dB
```

Phép so sánh vì vậy chỉ thay đổi thời điểm mở action, không đồng thời thay đổi
ngưỡng đo, observation, reward, action target hoặc kiến trúc PPO.

## Ngữ nghĩa protocol

- Trigger PPO mới: `a3_entry_redecision_v1`.
- Khi A3 vừa vào: mở ngay hai action `STAY` và `HO_C1`.
- `STAY`: không hủy episode A3; nếu điều kiện còn hợp lệ thì mở lại quyết định
  sau 20 ms.
- TTT 40 ms vẫn chạy dưới dạng shadow. Khi shadow TTT hoàn tất, PPO vẫn có thể
  quyết định tại mốc maturity và tiếp tục tái quyết định sau đó.
- Nếu A3 rời vùng hysteresis, episode timing và lịch tái quyết định bị hủy.
- HO preparation/execution, RLM, HOF, RLF recovery, ping-pong và rate accounting
  giữ nguyên.
- Baseline `3gpp_always_ho` trong validation/benchmark vẫn dùng trigger chuẩn
  `completed_a3_ttt`; nó không được chuyển sang trigger sớm.

## Tương thích

`completed_a3_ttt` vẫn là mặc định. Checkpoint cũ không có trường trigger được
hiểu là trigger chuẩn. Checkpoint T6A lưu trigger mới trong objective; benchmark
tự dựng PPO bằng trigger mới nhưng dựng 3GPP bằng trigger chuẩn.

## Pilot khóa yếu tố

Pilot dùng cùng cấu hình T5A: K=1, `ranked_v1`, observation
`t4d1_compact_v1`, reward `trajectory_qos_recovery_reward_v1`, critic
`fixed_scale_v1`, seed 17 và 10.240 macro-step. Yếu tố thay đổi duy nhất là
`decision_trigger`.

Chạy từ PowerShell:

```powershell
.\scripts\run_t6a_a3_entry_timing_pilot.ps1
```

Nếu CUDA trong môi trường PyTorch hoạt động:

```powershell
.\scripts\run_t6a_a3_entry_timing_pilot.ps1 -Device cuda
```

## Gate sau pilot

Không mở 100k chỉ vì return tăng. Trước tiên cần xác nhận:

1. PPO thực sự sử dụng cửa sổ trước TTT: có action tại elapsed 0/20 ms và tỷ lệ
   `STAY`/`HO_C1` không suy biến hoàn toàn.
2. Deterministic validation giữ relative-rate retention tối thiểu 98% tổng thể
   và từng tốc độ.
3. HOF, ping-pong hoặc RLF-time có ít nhất một tín hiệu cải thiện mà không làm
   QoS suy giảm ngoài gate.
4. Kết luận không chỉ dựa vào một stochastic seed; stochastic validation chỉ là
   diagnostic, không chọn checkpoint.

Nếu pilot qua các gate trên mới mở T6A 100k đa seed. Nếu policy quay về HO ngay
hoặc luôn WAIT, cần audit advantage/action probability theo stage trước khi tăng
budget.

`training_metrics.jsonl` và `training_report.json` lưu thêm
`timing_stage_action_counts` theo khóa như `entry:STAY`, `entry:HO_C1`,
`scheduled:STAY`, `maturity:HO_C1`. Đây là bằng chứng trực tiếp để phân biệt PPO
dùng cửa sổ trước TTT với PPO chỉ chờ đến maturity rồi bắt chước 3GPP.
