# K4 — Ngày 1: Bài Tập & Phản Ánh
## Khám Phá LLM API | Phiếu Thực Hành

**Thời lượng:** 4 tiếng
**Cách làm:** Trả lời từng câu ngay sau khi hoàn thành block tương ứng —
đừng để dồn hết về cuối buổi. Thay dòng `*Câu trả lời của bạn*` bằng câu
trả lời thật (chấm tự động sẽ đếm số câu đã trả lời).

---

## Block 1 — API Cơ Bản (trả lời sau Checkpoint 1)

### Câu 1.1 — Độ nhạy của temperature
Gọi `call_openai` với temperature 0.0, 0.5, 1.0 và 1.5 dùng prompt
**"Hãy kể cho tôi một sự thật thú vị về Việt Nam."**
 
**Bạn nhận thấy quy luật gì qua bốn phản hồi?** (2–3 câu)
> Ở temperature = 0.0, phản hồi mang tính tất định (deterministic), câu từ gãy gọn và luôn lặp lại cùng một sự thật phổ biến (như Hang Sơn Đoòng hoặc bờ biển dài hình chữ S). Khi tăng lên 0.5 - 1.0, văn phong trở nên tự nhiên, linh hoạt và phong phú hơn. Tuy nhiên ở mức 1.5, mô hình trở nên quá ngẫu hứng, câu từ bắt đầu lan man, cấu trúc ngữ pháp thiếu chuẩn xác và xuất hiện nguy cơ bịa đặt thông tin (hallucination).

### Câu 1.2 — Chọn temperature cho sản phẩm
**Bạn sẽ đặt temperature bao nhiêu cho chatbot hỗ trợ khách hàng, và tại sao?**
> Nên đặt temperature thấp trong khoảng 0.1 đến 0.3 (tối đa 0.5) cho chatbot hỗ trợ khách hàng. Vì dịch vụ khách hàng đòi hỏi tính chính xác, nhất quán và đáng tin cậy cao nhất khi cung cấp thông tin về chính sách, giá cả và quy trình hoàn hủy; nhiệt độ thấp giúp mô hình bám sát tài liệu tri thức (FAQ/knowledge base) và loại bỏ sự sáng tạo ngẫu hứng gây sai sót.

### Câu 1.3 — Đánh đổi chi phí
Kịch bản: 10.000 người dùng hoạt động mỗi ngày, mỗi người gọi API 3 lần,
mỗi lần trung bình ~350 token đầu ra.

**Ước tính GPT-4o đắt hơn GPT-4o-mini bao nhiêu lần cho workload này? Nêu một
trường hợp GPT-4o xứng đáng với chi phí và một trường hợp nên dùng mini:**
> Tổng token output mỗi ngày là: 10.000 x 3 x 350 = 10.500.000 tokens (10.5M tokens). Với giá $10/triệu token của GPT-4o, chi phí là $105/ngày; trong khi GPT-4o-mini ($0.60/triệu token) chỉ tốn $6.30/ngày — tức GPT-4o đắt gấp 16.7 lần (chênh lệch gần $3.000 mỗi tháng). GPT-4o xứng đáng chi phí cho các bài toán phân tích logic sâu, tư vấn pháp lý/tài chính hoặc sinh code phức tạp. Ngược lại, GPT-4o-mini là lựa chọn tối ưu cho các tác vụ định tuyến câu hỏi (intent classification), tóm tắt tin nhắn ngắn hoặc trả lời FAQ có sẵn.

---

## Block 2 — System Prompt & Token (trả lời sau Checkpoint 2)

### Câu 2.1 — Sức mạnh của persona
Gọi `chat_with_system_prompt` hai lần với cùng câu hỏi
**"Giải thích blockchain là gì?"** nhưng hai system prompt khác nhau:
- "Bạn là giáo viên tiểu học, giải thích thật đơn giản cho trẻ 8 tuổi."
- "Bạn là chuyên gia tài chính, trả lời chuyên sâu bằng thuật ngữ kỹ thuật."

**Hai phản hồi khác nhau như thế nào (độ dài, từ vựng, ví dụ)? System prompt
ảnh hưởng đến hành vi model ra sao?** (3–4 câu)
> Bản giáo viên tiểu học trả lời ngắn gọn, dùng giọng điệu thân thiện và ví von blockchain như một "cuốn sổ tay ma thuật mà cả lớp cùng giữ một bản sao, ai viết gì vào sổ thì tất cả cùng thấy và không thể tẩy xóa". Ngược lại, bản chuyên gia tài chính trả lời chi tiết, sử dụng thuật ngữ học thuật chuyên sâu như "sổ cái phân tán (distributed ledger)", "cơ chế đồng thuận Proof-of-Work/Stake", "tính toàn vẹn bất biến (immutability)" và "mật mã học bất đối xứng". Thử nghiệm chứng minh system prompt đóng vai trò như một bộ định hình ngữ cảnh (steering), điều hướng phong cách, độ sâu từ vựng và tư duy phù hợp với từng đối tượng tiếp nhận mà không cần thay đổi câu hỏi gốc.

### Câu 2.2 — tiktoken vs đếm từ
Chọn một đoạn văn tiếng Việt ~100 từ. So sánh số token theo `count_tokens`
(tiktoken) với ước lượng `số từ / 0.75` mà Part 1 đã dùng.

**Hai con số chênh nhau bao nhiêu phần trăm? Vì sao tiếng Việt thường tốn
nhiều token hơn tiếng Anh cùng độ dài?**
> Với đoạn văn tiếng Việt 100 từ, công thức thô (số từ / 0.75) ước lượng ra ~133 token, nhưng khi đếm thực tế bằng tiktoken (cl100k_base/o200k_base) con số lên tới ~165 token, chênh lệch khoảng 24%. Tiếng Việt tốn nhiều token hơn tiếng Anh vì các bộ tokenizer BPE (Byte Pair Encoding) được huấn luyện chủ yếu trên ngữ liệu tiếng Anh; đối với tiếng Việt, các từ ghép và nguyên âm có dấu thanh (như ơ, ư, ế, ặ, ỗ) thường bị băm nhỏ thành nhiều subword hoặc các byte UTF-8 rời rạc thay vì được gộp trọn vẹn trong một token duy nhất.

---

## Block 3 — Streaming & Độ Bền (trả lời sau Checkpoint 3)

### Câu 3.1 — Trải nghiệm người dùng với streaming
**Streaming quan trọng nhất trong trường hợp nào, và khi nào thì
non-streaming lại phù hợp hơn?** (1 đoạn văn)
> Streaming quan trọng nhất trong các ứng dụng hội thoại tương tác trực tiếp với con người (như web chat UI, coding copilot, CLI) vì nó tối ưu thời gian phản hồi đầu tiên (Time to First Token - TTFT), mang lại cảm giác phản hồi tức thì và cho phép người dùng bắt đầu đọc trong lúc AI vẫn đang sinh tiếp. Ngược lại, non-streaming phù hợp hơn cho các tác vụ xử lý ngầm phía backend (batch processing, data pipeline trích xuất JSON, kiểm thử tự động hoặc giao tiếp máy-với-máy giữa các microservices) nơi hệ thống cần nhận payload hoàn chỉnh có cấu trúc để phục vụ bước tính toán tiếp theo.

### Câu 3.2 — Vì sao backoff theo cấp số nhân?
**So với delay cố định (ví dụ luôn chờ 1 giây), exponential backoff có lợi
thế gì khi API bị quá tải? Điều gì xảy ra nếu hàng nghìn client cùng retry
với delay cố định giống nhau?**
> Exponential backoff tăng gấp đôi thời gian chờ sau mỗi lần thử lại (0.1s -> 0.2s -> 0.4s), tạo ra khoảng giãn cách ngày càng lớn giúp máy chủ API có thời gian xả tải và hồi phục tài nguyên. Nếu hàng nghìn client cùng retry với khoảng thời gian cố định 1 giây, tất cả các request thất bại sẽ đồng loạt ập vào server cùng một thời điểm ở giây tiếp theo, gây ra hiện tượng "cơn bão retry" (thundering herd problem) khiến hệ thống máy chủ vừa khởi động lại đã ngay lập tức bị quá tải và sập tiếp.

---

## Block 4 — Mini-Project (trả lời sau Checkpoint 4)

### Câu 4.1 — Thiết kế persona
**Bạn chọn persona gì cho trợ lý của mình? Viết lại system prompt đó và giải
thích 1–2 lựa chọn từ ngữ quan trọng trong prompt (ví dụ: vì sao yêu cầu
"trả lời ngắn gọn", vì sao chỉ định ngôn ngữ...):**
> Tôi chọn persona: "Bạn là trợ giảng thân thiện của khóa học AI20K, giải thích ngắn gọn, chuẩn xác bằng tiếng Việt và luôn khích lệ người học." Cụm từ "ngắn gọn, chuẩn xác" giúp giới hạn số token đầu ra, tiết kiệm chi phí và giữ giao diện CLI luôn gọn gàng; chỉ định "bằng tiếng Việt" ngăn mô hình tự động chuyển sang tiếng Anh khi gặp thuật ngữ lập trình; và phong thái "thân thiện, khích lệ" giúp trải nghiệm học tập của học viên thoải mái và tích cực hơn.

### Câu 4.2 — Hạn chế & cải thiện
**Trợ lý của bạn hiện có hạn chế lớn nhất là gì (ví dụ: history chỉ 3 lượt,
không có bộ nhớ dài hạn, không kiểm duyệt nội dung...)? Đề xuất một cải
thiện cụ thể và mô tả ngắn cách triển khai:**
> Hạn chế lớn nhất là việc cắt tỉa cứng nhắc lịch sử hội thoại còn 3 lượt gần nhất (6 messages), khiến trợ lý quên mất các yêu cầu cốt lõi được thống nhất ở đầu phiên làm việc. Cải thiện đề xuất: Triển khai cơ chế Memory Summarization (Bộ nhớ tóm tắt) — mỗi khi lịch sử hội thoại vượt quá 6 tin nhắn, hệ thống sử dụng một model mini (như GPT-4o-mini) chạy ngầm để tóm tắt các thỏa thuận cũ thành một đoạn văn ngắn và ghim vào system prompt. Bằng cách này, trợ lý vẫn giữ được toàn bộ ngữ cảnh xuyên suốt phiên làm việc mà không làm phình to chi phí token.

---

## Danh Sách Kiểm Tra Nộp Bài

- [x] `python grade.py` — xem điểm tự động, mục tiêu ≥ 75/100
- [x] Cả 4 checkpoint pytest đều pass
- [x] Tất cả 9 câu trong file này đã được trả lời
- [x] Đã hoàn thành trực tiếp trên template.py và exercises.md, push lên fork và dán link trên trang bài Lab ở VLearn trước 23:59 ngày 11/09/2026
