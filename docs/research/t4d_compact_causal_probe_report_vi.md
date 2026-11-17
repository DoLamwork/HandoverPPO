# Báo cáo T4D compact causal QoS/risk probe

## Mục tiêu

T4D kiểm tra giả thuyết rằng lỗi tổng quát hóa của T4C đến từ observation dư thừa và một nhãn nhị phân quá thô. T4D không thay đổi protocol, trigger, policy PPO hoặc tập dữ liệu. Probe tái sử dụng đúng các state TRAIN đã đóng băng tại T4C.

T4D tách bài toán thành:

- QoS head: hồi quy `WAIT_20 - HO_NOW` theo delivered Mbit tại horizon 1 giây.
- Risk head: phân loại WAIT có làm tăng bất kỳ HOF, PP, RLF, recovery time hoặc interruption time hay không.
- Selector: chỉ WAIT khi QoS gain dự đoán vượt ngưỡng và risk score thấp hơn ngưỡng. Architecture và hai ngưỡng chỉ được chọn trên tuning.

## Observation compact

QoS head dùng 14 feature causal về A3 elapsed, RSRP delta/trend/variance và SINR hiện tại/trend/minimum. Risk head dùng 19 feature: 14 feature QoS cộng N310, N311, T310 active/progress và target-is-previous.

T4D chủ động loại:

- speed, route progress, serving residence, time since last HO, recent HO count và candidate persistence;
- bốn RLM-margin feature là phép biến đổi affine của SINR tương ứng.

Mọi window kết thúc tại decision tick; không có future feature.

## Nhãn và split

- Minimum meaningful gain: `0,05 Mbit`.
- Meaningful WAIT: gain ít nhất `0,05 Mbit` và không có risk xấu hơn.
- Số mẫu fit/tuning/confirmation: `2033/523/514`.
- Meaningful positive: `131/42/37`.
- Risk-positive: `129/29/29`.
- Fit dùng route-balanced loss.
- Tuning dùng cho early stopping, chọn architecture và khóa threshold.
- Confirmation không tham gia bất kỳ bước lựa chọn nào.
- Validation/test/zero-shot rows đọc: `0`.

## Pilot seed 29

Architecture được chọn trên tuning là `linear`. Threshold được khóa là predicted gain `0,5 Mbit` và risk score `0,05`.

Trên confirmation:

- QoS Pearson: `0,1004`; bootstrap CI95% `[0,0345; 0,2278]`.
- QoS Spearman: `0,3235`.
- Meaningful AP: `0,2111`, so với prevalence `0,0720`.
- Selector thực hiện `0/514` WAIT; utility và risk delta đều bằng 0.
- Gate QoS signal: pass.
- Gate meaningful ranking: pass.
- Gate local utility: fail.
- Gate vượt simple rule: fail.
- Kết luận local: `NOT_PASSED`; PPO tiếp tục đóng.

MLP-small cũng không tạo utility: QoS Pearson `0,0922`, Spearman `0,3506`, meaningful AP `0,1139`; selector được khóa về no-op.

## Reference vật lý

Rule `WAIT nếu target SINR < serving SINR` chọn 56/514 state và đạt:

- mean gain `+0,02560 Mbit/state`;
- paired route-cluster bootstrap gain CI95% `[+0,00092; +0,05623]`;
- ping-pong và interruption giảm có ý nghĩa trong bootstrap;
- HOF/RLF/recovery có trung bình giảm nhưng CI vẫn cắt 0.

Rule này tốt hơn selector T4D tại local counterfactual. Đây vẫn chưa phải bằng chứng full-route vì các quyết định WAIT liên tiếp có thể làm thay đổi state distribution.

## Kết luận và gate tiếp theo

Việc thu gọn observation làm phục hồi tín hiệu xếp hạng QoS trên route confirmation, nhưng hồi quy robust hiện tại co dự đoán về gần 0 và risk classifier chưa đủ hiệu chuẩn. Khi khóa theo utility tuning, lựa chọn hợp lý nhất của cả hai model là không WAIT. Do đó chưa được mở PPO.

Bước tiếp theo có giá trị là một T4D.1 diagnostic không đọc confirmation để lựa chọn: phân tích calibration của gain/risk trên fit+tuning, bổ sung meaningful-gain classifier hoặc quantile/ranking loss, rồi đánh giá lại trên một confirmation route split mới. Không được hạ threshold thủ công dựa trên confirmation hiện tại. Reference SINR nên được đưa vào full-route TRAIN audit riêng trước khi coi là policy ứng viên.

Artifact: `results/analysis/early_ho_timing/t4d_compact_probe_v1`.
