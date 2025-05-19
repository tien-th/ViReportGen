from categories.categories_ver1 import hinh_dang_ton_thuong, vi_tri_ton_thuong, xam_lan_dict, tang_chuyen_hoa
import json
from typing import Dict, Any, Optional

class AbnormalityMapper:
    def __init__(self):
        self.shape_categories = hinh_dang_ton_thuong
        self.location_categories = vi_tri_ton_thuong
        self.invasion_categories = xam_lan_dict
        self.fdg_categories = tang_chuyen_hoa

    def _find_matching_category(self, 
                              description: str, 
                              category_dict: Dict[str, list], 
                              default: str = "không rõ") -> str:
        """
        Tìm category phù hợp cho một mô tả
        
        Args:
            description (str): Mô tả cần phân loại
            category_dict (Dict[str, list]): Dictionary chứa các categories
            default (str): Giá trị mặc định nếu không tìm thấy
            
        Returns:
            str: Category phù hợp
        """
        if not description:
            return default

        description = description.lower()
        for category, values in category_dict.items():
            if any(value.lower() in description for value in values):
                return category
        return default

    def map_shape(self, shape_desc: str) -> str:
        """Map hình dạng tổn thương vào category"""
        return self._find_matching_category(shape_desc, self.shape_categories, "khác")

    def map_location(self, location_desc: str) -> str:
        """Map vị trí tổn thương vào category"""
        return self._find_matching_category(location_desc, self.location_categories)

    def map_invasion(self, invasion_desc: str) -> str:
        """Map xâm lấn vào category"""
        if not invasion_desc:
            return "Không xâm lấn"
        return self._find_matching_category(invasion_desc, self.invasion_categories)

    def map_fdg(self, fdg_info: Dict[str, Any]) -> str:
        """
        Map mức độ tăng chuyển hóa FDG vào category
        
        Args:
            fdg_info (Dict[str, Any]): Dictionary chứa thông tin FDG
            
        Returns:
            str: Category phù hợp
        """
        fdg_desc = fdg_info.get("Tăng chuyển hóa FDG", "")
        return self._find_matching_category(fdg_desc, self.fdg_categories)

    def map_sample(self, sample_data: Dict[str, Any]) -> Dict[str, str]:
        # print(type(sample_data))
        """
        Map toàn bộ thông tin từ sample vào categories
        
        Args:
            sample_data (Dict[str, Any]): Dictionary chứa thông tin sample cần map
            
        Returns:
            Dict[str, str]: Dictionary chứa kết quả mapping
        """
        try:
            return {
                "hinh_dang": self.map_shape(
                    sample_data.get("Hình dạng của khối u, tổn thương, bất thường", "")
                ),
                "vi_tri": self.map_location(
                    sample_data.get("Vị trí của khối u, tổn thương, bất thường", "")
                ),
                "xam_lan": self.map_invasion(
                    sample_data.get("Xâm lấn", "")
                ),
                "tang_chuyen_hoa": self.map_fdg(
                    sample_data.get("Mức độ FDG", {})
                )
            }
        except:
            print(sample_data)

def main():
    # Ví dụ cách sử dụng
    sample = {
        "Hình dạng của khối u, tổn thương, bất thường": "Hình ảnh khối mờ bờ đa cung",
        "Vị trí của khối u, tổn thương, bất thường": "Ngoại vi thùy dưới phổi trái",
        "Mức độ FDG": {
            "SUVmax": 13.9,
            "Tăng chuyển hóa FDG": "Cao"
        },
        "Xâm lấn": "xâm lấn rốn phổi phải, dính vào màng phổi trung thất"
    }
    
    mapper = AbnormalityMapper()
    result = mapper.map_sample(sample)
    
    print("Kết quả mapping:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()