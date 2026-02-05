Milestone 8 手动验证清单

1) 创建世界
   - 输入 world prompt 并点击 Create World。
   - 观察生成成功提示与 session_id。

2) 进入游戏页并验证持久化
   - 查看左侧 Session 与当前位置。
   - 刷新页面后仍能加载相同 session（状态不丢失）。

3) 切换地点
   - 选择地图下拉框切换地点。
   - 刷新页面后位置保持一致。

4) 选择 NPC 并对话
   - 在当前地点选择 NPC。
   - 发送一条消息并查看对话记录是否写入。

5) NPC 移动与拒绝
   - 请求 NPC 移动到其它地点：
     - 若人格拒绝：日志中出现拒绝，NPC 仍在原地。
     - 若人格接受：NPC 出现在新地点。

6) RAG 注入检查
   - 勾选“显示最近一次 RAG IDs”。
   - 确认 always_include_ids 与 retrieved_ids 出现。

7) 日志增长
   - 检查 data/sessions/<session_id>/turns.jsonl 持续追加。
