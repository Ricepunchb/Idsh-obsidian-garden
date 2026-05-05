---
title: 2026 상반기 취준 현황
date: 2026-04-30
publish: true
tags:
  - Career
---

```dataview
TABLE
  title AS "title",
  회사명 AS "회사", 
  지원직무 AS "직무", 
  상태 AS "현재 상태" 
FROM "Job_Hunting/logs" 
SORT 상태 DESC
```

