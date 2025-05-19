import json
from typing import Dict, List, Any, Set
from collections import defaultdict
from mapping_abnormality_v2 import AbnormalityMapper
from utils import * 

special_path = './vitri_dacbiet.txt'
special_list = read_set_from_file(special_path)


class PredictionEvaluator:
    def __init__(self, gt_file: str, pred_file: str, fields: List[str]):
        """
        Khởi tạo evaluator với đường dẫn tới file ground truth và predictions
        
        Args:
            gt_file (str): Đường dẫn tới file ground truth
            pred_file (str): Đường dẫn tới file predictions
        """

        # self.fields = [
        #     "hinh_dang",
        #     "vi_tri",
        #     # "xam_lan",
        #     "tang_chuyen_hoa"
        # ]
        self.fields = fields


        self.gt_file = gt_file
        self.pred_file = pred_file
        self.mapper = AbnormalityMapper()
        self.raw_gt_data = self._load_json(gt_file)
        self.raw_pred_data = self._load_json(pred_file)

        self.split_key=None
        # if "vi_tri" in self.fields: 
        self.split_key = 'Vị trí của khối u, tổn thương, bất thường'
        self.split_str_and_duplicate()
        
        # Map dữ liệu về các category chuẩn
        self.gt_data = [self.mapper.map_sample(sample) for sample in self.raw_gt_data]
        self.pred_data = [self.mapper.map_sample(sample) for sample in self.raw_pred_data]
        
        # print(self.gt_data)
        # print(self.pred_data)
        # print("----------------------------------")
    def reload(self): 
        self.raw_gt_data = self._load_json(self.gt_file)
        self.raw_pred_data = self._load_json(self.pred_file)
    
    def remap(self): 
        self.gt_data = [self.mapper.map_sample(sample) for sample in self.raw_gt_data]
        self.pred_data = [self.mapper.map_sample(sample) for sample in self.raw_pred_data]


    def _load_json(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load data từ file JSON và xử lý trường hợp dữ liệu là string
        
        Args:
            file_path (str): Đường dẫn tới file JSON
            
        Returns:
            List[Dict[str, Any]]: List các dictionary sau khi parse
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        # Nếu data là string (JSON string), parse nó
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                # print(data)
                data = [data]  # Nếu không parse được, wrap nó trong list
                
        # Chuyển đổi thành list nếu cần
        if not isinstance(data, list):
            data = [data]
            
        return data
    
    def _is_matching(self, pred_sample: Dict[str, Any], gt_sample: Dict[str, Any]) -> bool:
        """
        Kiểm tra xem hai sample có match với nhau không.
        Lưu ý: Samples đã được map về category chuẩn trước khi so sánh
        
        Args:
            pred_sample: Sample từ predictions (đã được map)
            gt_sample: Sample từ ground truth (đã được map)
            
        Returns:
            bool: True nếu hai sample match với nhau
        """
        fields = self.fields
        
        return all(
            (pred_sample.get(field) == gt_sample.get(field) and pred_sample.get(field) != 'khac' )
            for field in fields
        )
    
    def evaluate(self) -> Dict[str, Any]:
        """
        Đánh giá predictions so với ground truth
        
        Returns:
            Dict chứa các metric đánh giá:
            - total_gt: Tổng số samples trong ground truth
            - total_pred: Tổng số samples trong predictions
            - correct_predictions: Số lượng predictions đúng
            - accuracy: Tỷ lệ dự đoán đúng
            - matched_samples: List các cặp sample matched
        """
        total_gt = len(self.gt_data)
        total_pred = len(self.pred_data)
        matched_samples = []
        used_gt_indices = set()
        
        # Với mỗi prediction, tìm gt sample phù hợp
        for pred_idx, pred_sample in enumerate(self.pred_data):
            for gt_idx, gt_sample in enumerate(self.gt_data):
                if gt_idx in used_gt_indices:
                    continue
                    
                if self._is_matching(pred_sample, gt_sample):
                    matched_samples.append({
                        "pred_idx": pred_idx,
                        "gt_idx": gt_idx,
                        "sample": {
                            "prediction": pred_sample,
                            "ground_truth": gt_sample
                        }
                    })
                    used_gt_indices.add(gt_idx)
                    break
        
        correct_predictions = len(matched_samples)
        accuracy = correct_predictions / total_gt if total_gt > 0 else 0
        
        return {
            "total_gt": total_gt,
            "total_pred": total_pred,
            "correct_predictions": correct_predictions,
            "accuracy": accuracy,
            "matched_samples": matched_samples
        }
    
    def get_unmatched_samples(self) -> Dict[str, List[Dict[str, Any]]]:
        """
        Lấy ra các samples không được match
        
        Returns:
            Dict chứa:
            - unmatched_gt: List ground truth samples không được match
            - unmatched_pred: List prediction samples không match với gt nào
        """
        matched_gt_indices = {match["gt_idx"] for match in self.evaluate()["matched_samples"]}
        matched_pred_indices = {match["pred_idx"] for match in self.evaluate()["matched_samples"]}
        
        unmatched_gt = [
            sample for idx, sample in enumerate(self.gt_data)
            if idx not in matched_gt_indices
        ]
        
        unmatched_pred = [
            sample for idx, sample in enumerate(self.pred_data)
            if idx not in matched_pred_indices
        ]
        
        return {
            "unmatched_gt": unmatched_gt,
            "unmatched_pred": unmatched_pred
        }

    def process_vitri(self, text):
        text = text.lower()
        text = remove_x_at_start(text, 'ở')
        texts = xu_ly_chuoi_vi_tri_nang_cao(text, special_list)
        texts = remove_and_report_single_word_strings_from_list(texts)
        return texts 
    
    def split_str_and_duplicate(self): 

            if self.split_key is not None: 
                try:
                    for obj in self.raw_gt_data : 
                        # try: 
                        obj[self.split_key] = self.process_vitri(obj[self.split_key])
                        # except KeyError: 
                        #     print(obj)
                            
                    for obj in self.raw_pred_data : 
                        obj[self.split_key] = self.process_vitri(obj[self.split_key])
                except: 
                    print('+' * 100) 
                    print(self.pred_file)
                    print('+' * 100) 
                self.raw_gt_data = expand_dictionaries_by_list_key(self.raw_gt_data, target_key=self.split_key)
                self.raw_pred_data = expand_dictionaries_by_list_key(self.raw_pred_data, target_key=self.split_key)

def main():
    # Ví dụ sử dụng
    evaluator = PredictionEvaluator(
        gt_file="data/PETCT_2023_THANG 12_27_patient_1658_REPORT_patient_1658.json",
        pred_file='/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/datajson/adaptive_cosmos_Lmed_O_P_ver2/7b7e9ec1aa516b31551384f3c781f1cf4309a6c6752efd6eb20671c9e92fb379.json'
    )
    
    # Đánh giá kết quả
    results = evaluator.evaluate()
    print("\nKết quả đánh giá:")
    print(f"Tổng số samples trong ground truth: {results['total_gt']}")
    print(f"Tổng số samples trong predictions: {results['total_pred']}")
    print(f"Số lượng predictions đúng: {results['correct_predictions']}")
    print(f"Accuracy: {results['accuracy']:.2%}")
    
    # Lấy các samples không match
    unmatched = evaluator.get_unmatched_samples()
    print(f"\nSố lượng ground truth samples không được match: {len(unmatched['unmatched_gt'])}")
    print(f"Số lượng prediction samples không match với gt nào: {len(unmatched['unmatched_pred'])}")

if __name__ == "__main__":
    main()