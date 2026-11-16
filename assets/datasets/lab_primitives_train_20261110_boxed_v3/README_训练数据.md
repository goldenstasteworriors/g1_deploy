# 最新训练数据

已按要求移除 RETURN_DROPPER-001、002、003 及对应初始框。

364 段，147675 帧；放回滴管剩余 40 段。309 段有初始框，55 段无需标框。三路视频与数值数据同步保留，EP 编号已重排，review_id 不变。

初始框：meta/initial_ego_boxes.json；索引：meta/segment_provenance.json；检查：meta/removal_validation.json。

原始录像和旧训练版作为归档保留。


## 放回试管框更新

42 段已改为人工完整放置区域框，参考放回后试管范围，不使用 SAM；005、006、012、034–045 为右边界截断区域。

[完整对比图和检查视频](initial_ego_boxes/return_tube_full_regions_20261110/README.md)。训练使用的 meta/initial_ego_boxes.json 已同步更新。


最新训练版：[v4，363 段，删除归还试管 004 并修正五段框](../lab_primitives_train_20261110_boxed_v4/README_训练数据.md)。
