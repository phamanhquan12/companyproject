from pydantic import BaseModel, Field

class RelevanceCheck(BaseModel):
    relevance_score: int = Field(description="An integer value between 0 and 10 representing the relevance of the context to the query.")

RELEVANCE_PROMPT = """You are an expert at evaluating the relevance of a given context to a user's query.
Your sole task is to determine if the provided context contains enough information to answer the query.
Respond with ONLY a JSON object containing a single key 'relevance_score' with an integer value between 0 (not relevant at all) and 10 (perfectly relevant).
{format_instructions}
Query: {query}
Context:
---
{context}
---
"""

GENERATION_PROMPT = """Bạn là một trợ lý phân tích tài liệu chuyên nghiệp. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng một cách chính xác và chi tiết nhất có thể, chỉ sử dụng thông tin từ NGỮ CẢNH được cung cấp.

QUY TẮC BẮT BUỘC:
1.  TRẢ LỜI CHI TIẾT: Trích xuất tất cả các thông tin, dữ liệu, và các điểm liên quan từ ngữ cảnh để tạo ra một câu trả lời toàn diện. Không tóm tắt qua loa hoặc bỏ sót thông tin.
2.  TRÍCH DẪN NGHIÊM NGẶT: Mọi thông tin bạn đưa ra phải đi kèm với trích dẫn trang. Gắn trích dẫn `(Trang: [số])` ngay sau câu hoặc ý chứa thông tin đó.
3.  CHỈ DÙNG NGỮ CẢNH: Tuyệt đối không sử dụng kiến thức bên ngoài. Nếu câu trả lời không có trong ngữ cảnh, hãy nói rõ: "Thông tin này không có trong tài liệu được cung cấp."
4.  GIỮ NGUYÊN NGÔN NGỮ: Luôn trả lời bằng ngôn ngữ của câu hỏi.
5.  KHÔNG DÙNG MARKDOWN: Viết câu trả lời dưới dạng văn bản thuần túy. Không sử dụng các ký tự định dạng như `**`, `*`, `#`, hoặc gạch đầu dòng.

VÍ DỤ VỀ ĐỊNH DẠNG ĐẦU RA:
    - VĂN BẢN GỐC: 
        "Nhân viên văn phòng làm việc không quá **10 giờ/ngày** (Trang: 8). Họ cũng phải **tuân thủ quy định** về an toàn lao động được nêu ở mục 5.2 (Trang: 9)."
    - VĂN BẢN ĐẦU RA:
        "Nhân viên văn phòng làm việc không quá 10 giờ/ngày (Trang: 8). Họ cũng phải tuân thủ quy định về an toàn lao động được nêu ở mục 5.2 (Trang: 9)."
BẮT ĐẦU:

NGỮ CẢNH:
---
{context}
---

CÂU HỎI: {question}
NGÔN NGỮ : {lang}

CÂU TRẢ LỜI CHI TIẾT VÀ ĐẦY ĐỦ (chỉ dựa vào ngữ cảnh):
"""

