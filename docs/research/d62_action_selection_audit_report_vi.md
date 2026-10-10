# D62 — audit cách chọn action

## Phạm vi

Audit dùng nguyên checkpoint D62, không cập nhật trọng số. Nó đọc 3.117 state
TRAIN trong bank đã khóa và chạy lại đúng 150 validation trajectory bằng ba
cách giải mã action:

- `joint_argmax`: argmax trên `STAY, HO_C1, HO_C2, HO_C3` như benchmark hiện tại;
- `hierarchical_argmax`: so `P(HO)` với `P(STAY)`, rồi argmax candidate nếu HO;
- `stochastic_joint`: sample phân phối joint với seed 17, 29 và 43.

Không đọc test hoặc zero-shot. Mọi selector đi qua cùng evaluator và cùng đường
tính rate, RLF, HOF, ping-pong.

## TRAIN state bank

Xác suất trung bình của D62 là `STAY=0,2784`, `C1=0,2657`, `C2=0,2321`,
`C3=0,2239`; tổng `P(HO)=0,7216`. Khoảng `P(HO)` rất hẹp, từ 0,7173 đến
0,7326, và lớn hơn 0,5 ở toàn bộ state.

| Selector | STAY | C1 | C2 | C3 |
|---|---:|---:|---:|---:|
| Joint argmax | 3.050 | 67 | 0 | 0 |
| Hierarchical argmax | 0 | 3.111 | 6 | 0 |
| Sampling seed 17 | 891 | 850 | 697 | 679 |
| Sampling seed 29 | 828 | 852 | 737 | 700 |
| Sampling seed 43 | 870 | 839 | 694 | 714 |

Joint và hierarchical argmax bất đồng ở 97,85% state. Trong 167 state có
C2/C3 tốt hơn cả STAY và C1 ở horizon 1 giây, recall của joint argmax bằng 0%,
hierarchical argmax bằng 1,20%. Sampling đạt 23,95–28,74%, gần với tổng xác
suất 27,39% mà policy đặt vào alternative hữu ích; đây là exploration ngẫu
nhiên, chưa phải khả năng xếp đúng candidate.

## Validation

| Selector | Rate Mbps | Relative rate | RLF-time | HOF | PP | HO | STAY |
|---|---:|---:|---:|---:|---:|---:|---:|
| Joint argmax | 20,448 | 78,769% | 1,308% | 1,887% | 10,470% | 1,968% | 98,032% |
| Hierarchical argmax | 23,029 | 91,289% | 1,297% | 10,773% | 37,058% | 100% | 0% |
| Sampling, mean 3 seed | 22,201 | 87,491% | 6,114% | 58,146% | 14,405% | 71,946% | 28,054% |
| 3GPP Always-C1 | 23,024 | 91,272% | 1,313% | 10,903% | 37,057% | 100% | 0% |

Hierarchical argmax thực hiện 5.444 C1, 8 C2 và không C3. Kết quả gần như
trùng 3GPP Always-C1: nó khôi phục QoS bằng cách luôn handover đến C1, đồng thời
mang HOF và ping-pong trở lại mức 3GPP. Vì vậy không thể xem đây là cải thiện
của PPO.

Sampling chọn C2/C3 thường xuyên nhưng không đúng ngữ cảnh. HOF tăng lên
57,70–58,95%, RLF-time khoảng 6,1% và connected time chỉ còn khoảng 91,8%.
Điều này xác nhận tăng exploration hoặc ép chọn alternative không giải quyết
được bài toán target selection.

## Kết luận nguyên nhân

Hai cách giải mã hiện tại nằm ở hai cực:

- joint argmax chia khối xác suất HO thành ba action, khiến STAY nhỏ nhưng vẫn
  thắng từng candidate riêng lẻ;
- hierarchical argmax thấy tổng `P(HO)>0,5` ở mọi state và biến thành Always-C1.

Candidate head cũng chưa phân biệt target: conditional argmax là C1 ở
3.111/3.117 TRAIN state. Do đó chỉ đổi selector không tạo ra policy tốt hơn.

Head timing hiện nhận 17 feature chung, trong khi block candidate chỉ đi vào
candidate scorer. Quyết định có HO hay không vì thế không trực tiếp sử dụng
rủi ro và xu hướng của từng đích đến. Ngoài ra prior D62 đặt `P(HO)=0,75` để
bốn joint action cùng bằng 0,25; prior này phù hợp action phẳng nhưng không trung
lập khi giải mã theo hai tầng.

## Hướng triển khai tiếp

Không promote D62 và không thay benchmark sang hierarchical argmax.

1. D63-A kiểm tra contract hai tầng nhất quán: khởi tạo timing 50% STAY / 50%
   HO, candidate conditional đều nhau; train vẫn dùng log-probability joint và
   validation dùng hierarchical argmax. Giữ nguyên observation, reward,
   redecision, optimizer, seed và budget 10.240.
2. Gate D63-A yêu cầu `P(HO)` không cùng phía ngưỡng ở hầu hết state, hành vi
   deterministic không rơi vào Always-STAY/Always-HO, relative rate ít nhất
   không thấp hơn D61 và HOF/PP không quay về mức 3GPP.
3. Nếu timing vẫn không phân biệt state, mở D63-B: timing head nhận context cộng
   biểu diễn pooling permutation-invariant của các candidate hợp lệ. Candidate
   scorer và mọi factor khác giữ nguyên.
4. Chỉ nếu timing đạt gate nhưng conditional candidate vẫn luôn C1 mới thêm
   auxiliary candidate loss trên TRAIN counterfactual labels, có đối chứng
   route-holdout và phạt false alternative.

## Artefact

- JSON: `results/analysis/p23_diagnostics/d62_action_selection_audit_v1/action_selection_audit.json`
- Runner: `scripts/run_d62_action_selection_audit.ps1`
- Checkpoint: `results/models/d6_pilots/D62_hierarchical_pilot_b10240_s17/smdp_ppo_selected.pt`
