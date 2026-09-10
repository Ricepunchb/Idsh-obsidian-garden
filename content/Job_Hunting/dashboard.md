---
title: 2026 하반기 취준 현황
date: 2026-09-10
publish: true
tags:
  - Career
---

```dataview
TABLE
  title AS "title",
  회사명 AS "회사", 
  지원직무 AS "직무",
  서류마감 AS "서류마감일",
  상태 AS "현재 상태" 
  
FROM "Job_Hunting/2026하반기/logs" 
SORT 상태 DESC
```

