import openai
import os
from dotenv import load_dotenv
import json
load_dotenv()
openai.api_key = os.getenv("OPEN_API_KEY")

# Load the unique values
with open('unique_values.json', 'r') as f:
    unique_value = json.load(f)

def extract_value(prompt):
    system_prompt = f"""
Hãy trả lời những thông tin chính xác và đầy đủ, chi tiết từ report nhưng không được tự nghĩ ra câu trả lời.
Chỉ trả về một list chứa những json object như yêu cầu, không được phép có bất kể kí tự gì không liên quan.
Giả sử bạn là một AI chuyên trích xuất thông tin từ những report y tế. Hãy làm theo các bước sau đây:
1. Tìm và đọc những đoạn thông tin liên quan đến những câu hỏi:
    - Có khối u, hạch, tổn thương hay bất thường (kén khí, nốt mờ nhỏ, ...) ở phổi hoặc di căn đến phổi không?
    - Kích thước của tổn thương, khối u, hạch, bất thường?
    - Hình dạng của tổn thương, khối u, hạch, bất thường?
    - Mức độ FDG?
    - SUVmax?
    - Tổn thương (khối u), bất thường có tăng chuyển hóa FDG hay không? (Có tăng, không tăng, tăng cao hoặc tăng ít, nếu không có thông tin thì điền là Không có)
    - Vị trí của tổn thương (khối u), bất thường?
    - Có xâm lấn hoặc dính vào hay không? Xâm lấn đi đâu? (xung quanh, thành ngực, mạch máu, ...)?
    - Giai đoạn của tổn thương (khối u), bất thường?
        + U nguyên phát
        + Di căn hạch
        + Di căn xa
2. Từ những đoạn thông tin tìm thấy, trích xuất ra những thông tin quan trọng (Thông tin nào không có thì hãy ghi là 'Không có'). Hãy trả ra một list, mỗi phần tử là 1 json object theo format sau:
    {{
    'Kích thước khối u, tổn thương, bất thường': ...,
    'Hình dạng của khối u, tổn thương, bất thường': ...,
    'Vị trí của khối u, tổn thương, bất thường': ...,
    'Mức độ FDG': {{
        'SUVmax': ...,
        'Tăng chuyển hoá FDG': ...
    }},
    'Xâm lấn': ... (Nếu không có thì để là Không có)
    'Giai đoạn chuyển hoá': ...
    }}
Dưới đây là ví dụ về cách thực hiện:
Ví dụ 1:
Input:
Tổn thương:
- Hình ảnh khối mờ bờ tua gai ở hạ phân thùy I thùy trên phổi phải kích thước 74 x 56 mm, tăng chuyển hóa FDG (SUVmax: 14,9).

Output:
[
    {{
        "Kích thước khối u, tổn thương, bất thường": "74 x 56 mm",
        "Hình dạng của khối u, tổn thương, bất thường": "Hình ảnh khối mờ bờ tua gai",
        "Vị trí của khối u, tổn thương, bất thường": "hạ phân thùy I thùy trên phổi phải",
        "Mức độ FDG": {{
            "SUVmax": "14.9",
            "Tăng chuyển hoá FDG": "Có tăng"
        }},
        "Xâm lấn": "Không có",
        "Giai đoạn chuyển hoá": "Không có"
    }}
]

Ví dụ 2:
Input:
Các tổn thương:
Gan:
- Tại phân thùy IV có hình ảnh nốt giảm tỷ trọng đường kính 10mm, không tăng chuyển hóa FDG, theo dõi nang gan.
Phổi:
- Có dãn phế nang 2 bên đỉnh phổi kèm vôi hóa dạng nốt nhỏ rải rác. SUVmax: 2.5
- Có hạch trước khí quản đoạn cao, đoạn thấp, kích thước 12 x 10 mm (SUVmax: 3.3).

Output:
[
    {{
        "Kích thước khối u, tổn thương, bất thường": "Không có",
        "Hình dạng của khối u, tổn thương, bất thường": "Nốt vôi hóa",
        "Vị trí của khối u, tổn thương, bất thường": "đỉnh phổi hai bên",
        "Mức độ FDG": {{
            "SUVmax": "2.5",
            "Tăng chuyển hoá FDG": "Không có"
        }},
        "Xâm lấn": "Không có",
        "Giai đoạn chuyển hoá": "Không có"
    }},
    {{
        "Kích thước khối u, tổn thương, bất thường": "12 x 10 mm",
        "Hình dạng của khối u, tổn thương, bất thường": "Hình ảnh hạch",
        "Vị trí của khối u, tổn thương, bất thường": "Trước khí quản đoạn cao và đoạn thấp",
        "Mức độ FDG": {{
            "SUVmax": "3.3",
            "Tăng chuyển hoá FDG": "Không có"
        }},
        "Xâm lấn": "Không có",
        "Giai đoạn chuyển hoá": "Không có"
    }}
]
"""

    PROMPT_MESSAGES = [
        {
            "role": "system", 
            "content": system_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    completion = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=PROMPT_MESSAGES,
        max_tokens=1000,
        temperature=0,
    )
    response_text = completion['choices'][0]['message']['content']
    return response_text

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default="adaptive_cosmos_Lmed_O_P_ver2.json")
    parser.add_argument("--output_folder", type=str, default="adaptive_cosmos_Lmed_O_P_ver2_extracted")
    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)
    # full_files = os.listdir("infer_data")
    # full_files = [f.split(".")[0] for f in full_files]
    for item, value in data.items():
        # if item != "38fad129c031b1049009144a46c2af28ce5ed3a2001c6a42065b48c495e2b371":
        #     continue
        # if item in full_files:
        #     continue
        # print(item)
        response = extract_value(value)
        # if item in full_files:
        #     continue
        with open(f"{args.output_folder}/{item}.json", "w", encoding="utf-8") as f:
            json.dump(response, f, ensure_ascii=False)