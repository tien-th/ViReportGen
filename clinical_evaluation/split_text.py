from utils import * 

directory_paths = ['/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/datajson/adaptive_cosmos_lmed', 
'/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/datajson/adaptive_cosmos_Lmed_O_P_ver2', '/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/datajson/adaptive_cosmos_m3d', '/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/datajson/adaptive_cosmos_m3d_LORA_O_P_ver2', '/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/datajson/ctvit_Lmed_LORA_O_P_ver2', '/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/datajson/ctvit_m3d_O_P_ver2', '/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/datajson/noise_ctvit_m3d_projector', '/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/datajson/noise_ctvit_med']

print (len(directory_paths))

target_key = 'Vị trí của khối u, tổn thương, bất thường'
vitri_texts = set()
for directory_path in directory_paths: 
    vitri_texts.update( extract_unique_values_from_json_files(directory_path, target_key))

gt_path = '/home/jovyan/shared/tienhuu060102/data-petct/clinical_eval/datajson/data'
vitri_texts_gt = extract_unique_values_from_json_files(gt_path, target_key)



splitted_unique_values = []
for text in vitri_texts: 
    text = remove_x_at_start(text, 'ở')
    texts = xu_ly_chuoi_vi_tri_nang_cao(text)
    splitted_unique_values.extend(texts)
    
splitted_unique_values = remove_and_report_single_word_strings(set(splitted_unique_values))

splitted_unique_values_gt = []
for text in vitri_texts_gt: 
    text = remove_x_at_start(text, 'ở')
    texts = xu_ly_chuoi_vi_tri_nang_cao(text)
    splitted_unique_values_gt.extend(texts)
    
splitted_unique_values_gt = remove_and_report_single_word_strings(set(splitted_unique_values_gt))


write_set_to_file(splitted_unique_values, 'vitrri_prediction.txt')
write_set_to_file(splitted_unique_values_gt, 'vitrri_gt.txt')
