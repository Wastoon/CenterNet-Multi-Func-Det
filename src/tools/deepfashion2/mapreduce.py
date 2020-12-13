
import json
import os

"""
test_json_path = '/data1/mry/datasets/deepfashion2_coco/annotations/keypoints_test_information.json'

total_anno = json.loads(open(test_json_path).read())
anno1= total_anno.copy()

anno1["images"] = total_anno['images'][:15657]
json.dump(anno1, open('/data1/mry/datasets/deepfashion2_coco/annotations/testimgpart1.json', 'w'))

anno2= total_anno.copy()
anno2["images"] = total_anno['images'][15657:15657*2]
json.dump(anno2, open('/data1/mry/datasets/deepfashion2_coco/annotations/testimgpart2.json', 'w'))

anno3= total_anno.copy()
anno3["images"] = total_anno['images'][15657*2:15657*3]
json.dump(anno3, open('/data1/mry/datasets/deepfashion2_coco/annotations/testimgpart3.json', 'w'))

anno4= total_anno.copy()
anno4["images"] = total_anno['images'][15657*3:]
json.dump(anno4, open('/data1/mry/datasets/deepfashion2_coco/annotations/testimgpart4.json', 'w'))

"""

save_path = '/data1/mry/datasets/deepfashion2_coco/annotations/submission.json'

part1 = json.loads(open('/data/mry/code/CenterNet/exp/cloth/subbssion320/S1results_part1.json').read())
part2 = json.loads(open('/data/mry/code/CenterNet/exp/cloth/subbssion320/S1results_part2.json').read())
part3 = json.loads(open('/data/mry/code/CenterNet/exp/cloth/subbssion320/S1results_part3.json').read())
part4 = json.loads(open('/data/mry/code/CenterNet/exp/cloth/subbssion320/S1results_part4.json').read())
final_anno = part1
final_anno = final_anno+part2 + part3+ part4


json.dump(final_anno, open(save_path, 'w'))
print(len(final_anno))

"""
resFile = '/data1/mry/datasets/deepfashion2_coco/annotations/submission/test_keypoints.json'
res = json.load(open(resFile))
file = open('/data1/mry/datasets/deepfashion2_coco/annotations/test_keypoints.txt', 'a')
for ann in res:
    file.write(json.dumps(ann))
    file.write('\n')
file.close()
"""