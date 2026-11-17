# Quyết định chốt T5A 100k multi-seed và mở held-out test

## Quyết định khóa trước khi đọc test

- Khóa ngân sách huấn luyện ở **100.000 macro-step**.
- Khóa ba training seed **17, 29, 43**, split seed **17**.
- Chỉ benchmark checkpoint `smdp_ppo_selected.pt` được chọn bằng deterministic
  validation và QoS gate; không dùng checkpoint cuối để thay thế sau khi xem test.
- Held-out benchmark dùng đúng split `test`, các tốc độ 30/50/70/90 km/h,
  dự kiến 150 route.
- Không chạy zero-shot trong gói này.
- Không thay observation, reward, trigger, action, checkpoint-selection rule hay
  hyperparameter sau khi test được mở.

## Lý do chốt ở 100k

Pilot 10.240 step đã được mở rộng thành ba run độc lập 100.000 step. Cả ba
selected checkpoint đều vượt QoS gate 98% trên validation và đều giảm point
estimate HOF, ping-pong, RLF-time so với 3GPP. Learning curve không đơn điệu;
train lâu hơn không đồng nghĩa checkpoint cuối tốt hơn. Vì vậy 300k không được
thêm sau khi nhìn test.

## Runner tái lập

Chạy:

```powershell
.\scripts\run_t5a_100k_multiseed_final_test.ps1
```

Runner từ chối ghi đè, kiểm tra contract của ba training report, lưu SHA-256
checkpoint/report vào `final_test_lock.json`, rồi chạy tuần tự ba benchmark CPU.

