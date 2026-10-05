# Báo cáo triển khai D3 — đối chứng tối ưu actor/critic

Ngày triển khai: 05/10/2026. Trạng thái: code và unit test hoàn tất; chưa chạy
pilot 10.240, screening 100k, test hoặc zero-shot.

## 1. Lý do mở D3

D1 cho thấy critic gradient lớn hơn actor 37–67 lần trong smoke run, global
gradient clipping chỉ giữ khoảng 2–2,5% norm trước clip và một số checkpoint
có hidden saturation gần 98%. D2 đồng thời cho thấy các reward cố định nhận
diện ít nhất một action hữu ích trong 100% state dương, nhưng policy argmax
không chọn action hữu ích. Vì vậy gate mở nhánh optimizer D3 trước D4 reward.

## 2. D3-A: tách encoder actor và critic

- `network_architecture=shared_mlp_v1` là mặc định tương thích checkpoint cũ.
- `separate_mlp_v1` tạo actor encoder 64×64 và critic encoder 64×64 độc lập,
  cùng activation và initialization như control.
- Policy head vẫn khởi tạo logits bằng nhau; action mask, sampling, PPO loss,
  learning rate và global clipping không đổi.
- Checkpoint lưu architecture trong cả config và `model_spec`; loader tự dựng
  đúng mạng và từ chối resume chéo architecture.
- Experiment manifest ghi tổng số parameter và số parameter thuộc đường actor,
  critic. Với shared encoder, shared parameter được tính trong cả hai đường và
  tổng parameter vẫn chỉ đếm một lần.

## 3. D3-B: fixed value-target scale

- `value_normalization=none` giữ đúng hành vi cũ với scale 1.
- `fixed_scale_v1` dùng critic head dự đoán `z`, còn API công khai trả
  `V_raw = s*z`. Collector, bootstrap, GAE, reward và advantage vẫn dùng đơn vị
  raw ban đầu.
- Critic loss là `0.5*MSE(z, G_raw/s)`. Log ghi cả
  `value_loss_normalized`, alias lịch sử `value_loss`, `raw_value_mse` và EV
  trên đơn vị raw; không so trực tiếp normalized MSE giữa hai scale.
- Nếu không truyền `--value-target-scale`, chương trình chọn cố định tối đa 20
  TRAIN route mỗi speed theo thứ tự xác định, chạy Always-HO-C1 và fit
  `s=max(1,sqrt(mean(G_raw^2)))` từ discounted return-to-go.
- `value_scale_calibration.json` lưu estimator, scale, thống kê return, route,
  hash RSRP/SINR và hash danh sách trace. Validation/test không tham gia fit và
  scale không được cập nhật trong training.
- Có thể truyền scale rõ ràng để tái lập một calibration đã khóa; manifest sẽ
  ghi nguồn là `explicit_cli_value`.

## 4. Tương thích và kiểm thử

- Checkpoint cũ thiếu ba field D3 được hiểu rõ ràng là shared/none/scale=1;
  không suy đoán separate từ tên tensor.
- Probe D1 dùng actor encoder cho activation saturation và vẫn đọc value raw,
  nên dùng được cho cả hai architecture.
- Test kiểm tra gradient actor không đi vào critic encoder và ngược lại,
  checkpoint separate round-trip, legacy checkpoint load, public value raw,
  công thức normalized/raw loss, calibration TRAIN xác định và subset cân bằng
  theo speed.
- Toàn bộ test suite: **178 test pass** trong 25,443 giây.

## 5. Pilot đã preregister nhưng chưa chạy

Script `scripts/run_d3_pilots.ps1` chạy tuần tự, chấp nhận thư mục rỗng, bỏ qua
arm đã hoàn tất hợp lệ và từ chối ghi đè mọi arm dở dang có artifact:

1. `D30_e07_control_pilot_b10240_s17`: shared control + diagnostics.
2. `D31_e07_separate_pilot_b10240_s17`: chỉ đổi sang separate encoder.
3. `D32_e07_vscale_pilot_b10240_s17`: shared + fixed TRAIN-fitted scale.

Tất cả giữ E07: K=3, `ranked_v1`, `qos_full_v1`,
`trajectory_qos_reward_v1`, seed/split 17, rollout 1024, batch 128, 4 epoch,
learning rate 5e-5, budget/evaluation 10.240 và `--skip-final-test`.

```powershell
.\scripts\run_d3_pilots.ps1
```

Pilot chỉ dùng để kiểm tra cơ chế: critic/actor gradient ratio, clipping factor,
EV, actor hidden saturation và useful-action probability trên state bank. Chỉ
mở D3-A×B hoặc screening 100k nếu ít nhất một treatment cải thiện nhất quán các
chỉ dấu cơ chế so với D30. Chưa có kết quả pilot nên hiện tại không có claim về
QoS hoặc vượt 3GPP.
