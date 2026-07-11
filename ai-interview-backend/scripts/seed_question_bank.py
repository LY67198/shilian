"""
批量导入题库初始数据 — Python 后端 / Java 后端 / 前端 三个岗位，各 10 题。

用法:
    docker exec shilian-app python scripts/seed_question_bank.py
"""
import asyncio
from app.db.base import get_session_local
from app.services.backoffice.question_bank_service import question_bank_service

QUESTIONS = [
    # ==================== Python 后端 (python_backend) ====================
    dict(category="self-intro", position_tag="python_backend", difficulty="easy",
         question="请做一个简单的自我介绍，重点突出你的 Python 后端项目经验。",
         reference_answer="面试官你好，我是一名 Python 后端开发，有 X 年经验。最近的项目是...",
         key_points=["清晰表达年限", "突出 Python 技术栈", "提及 1-2 个代表性项目"],
         tags=["自我介绍", "开场"], source="manual"),
    dict(category="technical", position_tag="python_backend", difficulty="easy",
         question="Python 中列表 (list) 和元组 (tuple) 有什么区别？分别适用于什么场景？",
         reference_answer="列表可变、元组不可变。列表适合动态数据集合，元组适合固定数据（如坐标、配置）、字典 key 等。元组性能略优于列表，且不可变性提供安全保证。",
         key_points=["可变 vs 不可变", "性能差异", "适用场景举例", "元组可作为 dict key"],
         tags=["Python 基础", "数据结构"], source="manual"),
    dict(category="technical", position_tag="python_backend", difficulty="medium",
         question="Python 的 GIL 是什么？它对多线程程序有什么影响？如何应对？",
         reference_answer="GIL (全局解释器锁) 确保同一时刻只有一个线程执行 Python 字节码。对 CPU 密集型任务，多线程反而可能更慢。应对方案：1) CPU 密集型用 multiprocessing；2) IO 密集型线程仍然有效（IO 操作会释放 GIL）；3) 用 C 扩展或 asyncio 协程。",
         key_points=["GIL 本质是互斥锁", "CPU 密集 vs IO 密集的影响", "multiprocessing 方案", "协程 asyncio 方案", "C 扩展可释放 GIL"],
         tags=["Python 进阶", "并发", "GIL"], source="manual"),
    dict(category="technical", position_tag="python_backend", difficulty="medium",
         question="装饰器的原理是什么？请写一个记录函数执行时间的装饰器。",
         reference_answer="装饰器本质是一个接受函数、返回新函数的高阶函数。常用 functools.wraps 保留原函数元信息。示例代码涉及 time.time() 记录前后时间差。",
         key_points=["高阶函数概念", "@语法糖", "functools.wraps 保留元信息", "带参数的装饰器需要额外一层"],
         tags=["Python 进阶", "装饰器"], source="manual"),
    dict(category="technical", position_tag="python_backend", difficulty="hard",
         question="asyncio 的事件循环是如何工作的？async/await 和传统的多线程比有什么优势？",
         reference_answer="asyncio 基于事件循环 (Event Loop)，使用协程实现协作式多任务。async def 定义协程，await 挂起当前协程等待 IO 完成。相比多线程：无 GIL 竞争、无锁开销、上下文切换成本极低。适合 IO 密集型高并发场景。",
         key_points=["事件循环单线程模型", "await 挂起不阻塞", "协程 vs 线程开销对比", "适用 IO 密集场景", "CPU 密集仍需要多进程"],
         tags=["Python 进阶", "asyncio", "并发"], source="manual"),
    dict(category="system_design", position_tag="python_backend", difficulty="medium",
         question="设计一个 RESTful API 的限流系统，你会如何实现？",
         reference_answer="常见方案：1) 令牌桶 (Token Bucket) — Redis + Lua 脚本原子操作；2) 滑动窗口 — Sorted Set；3) 漏桶 (Leaky Bucket)。存储选 Redis（分布式）、或内存（单机）。返回 429 Too Many Requests + Retry-After 头。",
         key_points=["令牌桶/滑动窗口算法", "Redis 原子操作", "分布式 vs 单机", "429 状态码 + Retry-After"],
         tags=["系统设计", "API", "限流", "Redis"], source="manual"),
    dict(category="system_design", position_tag="python_backend", difficulty="hard",
         question="设计一个支持千万级 DAU 的短链接系统，谈谈架构和数据存储方案。",
         reference_answer="核心流程：长链 → hash(MD5/SHA256 取前 7 位) → 存 DB → 返回短链。访问：短链 → 查缓存 → 302 跳转。存储选 MySQL 分库分表 + Redis 热数据缓存。Base62 编码优于纯 hash（避免碰撞）。预估 QPS/存储量推动技术选型。",
         key_points=["hash + Base62 生成", "302 临时跳转", "分库分表策略", "Redis 热数据缓存", "布隆过滤器防穿透"],
         tags=["系统设计", "短链接", "高并发"], source="manual"),
    dict(category="technical", position_tag="python_backend", difficulty="medium",
         question="SQLAlchemy 中 session 的生命周期是怎样的？什么是 N+1 查询问题？",
         reference_answer="Session 是工作单元：begin → 操作 → commit/rollback → close。N+1 问题：查 N 条记录后逐条查关联对象，产生 N+1 次 SQL。解决：joinedload/subqueryload 预加载、selectinload。",
         key_points=["Session 工作单元模式", "N+1 的本质", "joinedload 等预加载方式", "lazy 加载策略"],
         tags=["SQLAlchemy", "ORM", "数据库"], source="manual"),
    dict(category="project", position_tag="python_backend", difficulty="medium",
         question="请描述一个你解决过的、最有挑战性的后端性能问题。你是怎么定位和优化的？",
         reference_answer="（引导候选人按 STAR 方法回答：背景 Situation → 任务 Task → 行动 Action → 结果 Result）",
         key_points=["STAR 方法", "具体数据（QPS/延迟前后对比）", "用了什么 profiling 工具", "优化思路（SQL/缓存/架构）"],
         tags=["项目经验", "性能优化", "STAR"], source="manual"),
    dict(category="behavioral", position_tag="python_backend", difficulty="easy",
         question="在团队开发中，你和前端同学在 API 联调时出现过分歧吗？你是怎么处理的？",
         reference_answer="关键在于前期约定好接口文档。实践中用 OpenAPI/Swagger 文档先行，定义清楚请求/响应 schema、错误码，前后端按文档各自治开发，联调只是验证。分歧点通常在于错误码定义和字段命名，提前对齐即可。",
         key_points=["API 文档先行", "沟通技巧", "实际案例", "OpenAPI/Swagger"],
         tags=["行为面试", "团队协作", "API 联调"], source="manual"),

    # ==================== Java 后端 (java_backend) ====================
    dict(category="self-intro", position_tag="java_backend", difficulty="easy",
         question="请做一个简单的自我介绍，重点突出你的 Java 后端开发经验。",
         reference_answer="面试官你好，我是一名 Java 后端开发，熟悉 Spring Boot + MyBatis 技术栈...",
         key_points=["清晰表达年限", "突出 Java 生态", "提及代表性项目"],
         tags=["自我介绍", "开场"], source="manual"),
    dict(category="technical", position_tag="java_backend", difficulty="easy",
         question="String、StringBuilder 和 StringBuffer 的区别是什么？",
         reference_answer="String 不可变；StringBuilder 可变且线程不安全（性能高）；StringBuffer 可变且线程安全（synchronized）。频繁字符串拼接用 StringBuilder，多线程环境用 StringBuffer。",
         key_points=["String 不可变", "StringBuilder 快但非线程安全", "StringBuffer 线程安全但有锁开销"],
         tags=["Java 基础", "字符串"], source="manual"),
    dict(category="technical", position_tag="java_backend", difficulty="medium",
         question="Spring 的 IoC 和 DI 是什么？有哪些注入方式？",
         reference_answer="IoC (控制反转) 将对象创建权交给容器。DI (依赖注入) 是 IoC 的实现方式。Spring 支持：1) 构造器注入（推荐，不可变 + 便于测试）；2) Setter 注入；3) Field 注入（@Autowired，反射，不推荐）。",
         key_points=["IoC 容器概念", "构造器注入 vs Field 注入", "@Autowired/@Resource 区别", "循环依赖问题"],
         tags=["Spring", "IoC", "DI"], source="manual"),
    dict(category="technical", position_tag="java_backend", difficulty="medium",
         question="Java 中 HashMap 的底层原理是什么？JDK 8 做了哪些优化？",
         reference_answer="数组 + 链表 / 红黑树。hash(key) → index → 链表；链表长 >8 且数组长 ≥64 时转红黑树（O(n)→O(log n)）。扩容：容量翻倍，rehash。JDK 8 优化：链表→红黑树、头插→尾插（防并发死循环）、hash 扰动简化。",
         key_points=["数组+链表+红黑树结构", "树化阈值 8/64", "扩容机制", "JDK8 尾插防死循环"],
         tags=["Java 进阶", "HashMap", "数据结构"], source="manual"),
    dict(category="technical", position_tag="java_backend", difficulty="hard",
         question="详细解释 JVM 内存模型和 GC 算法。Full GC 频繁是什么原因？怎么排查？",
         reference_answer="JVM 内存：堆（年轻代 Eden+S0+S1 + 老年代）+ 方法区/元空间 + 栈 + 本地方法栈 + 程序计数器。GC 算法：标记-清除、复制、标记-整理。CMS/G1 等收集器。Full GC 频繁原因：老年代空间不足、System.gc() 显式调用、元空间溢出、大对象直接进老年代。排查：jstat -gc、GC 日志、MAT/JProfiler 堆 dump。",
         key_points=["JVM 内存区域划分", "GC 算法演进", "CMS vs G1 vs ZGC", "Full GC 排查工具"],
         tags=["JVM", "GC", "性能调优"], source="manual"),
    dict(category="system_design", position_tag="java_backend", difficulty="medium",
         question="设计一个分布式 ID 生成器。Snowflake 算法有哪些优缺点？",
         reference_answer="Snowflake：1bit 符号 + 41bit 时间戳 + 10bit 机器 ID + 12bit 序列号。优点：趋势递增、高性能、纯内存。缺点：依赖机器时钟（时钟回拨会重复）、workerId 需手动分配。改进：美团 Leaf（号段模式+Snowflake）、百度 UidGenerator。",
         key_points=["Snowflake 64bit 结构", "时钟回拨问题", "号段模式", "美团 Leaf/百度 UidGenerator"],
         tags=["系统设计", "分布式 ID", "Snowflake"], source="manual"),
    dict(category="technical", position_tag="java_backend", difficulty="medium",
         question="MySQL 索引的底层结构是什么？为什么 B+ 树比 B 树更适合做数据库索引？",
         reference_answer="InnoDB 用 B+ 树。B+ 树优势：1) 数据只存叶子节点，非叶子只存 key（高度更低、IO 更少）；2) 叶子节点形成双向链表，支持范围查询；3) 数据更集中利于预读和缓存。B 树节点存完整数据，IO 次数更多。",
         key_points=["B+ 树 vs B 树结构差异", "叶子节点双向链表", "范围查询优势", "磁盘 IO 次数对比"],
         tags=["MySQL", "索引", "B+树"], source="manual"),
    dict(category="project", position_tag="java_backend", difficulty="medium",
         question="你在项目中用过的设计模式有哪些？举一个具体的业务场景说明。",
         reference_answer="常见：策略模式（不同支付/优惠计算）、模板方法模式（业务流程骨架）、工厂模式（不同渠道适配）、观察者模式（事件驱动、MQ 消息）。最好结合具体业务场景说明，而非单纯罗列。",
         key_points=["举具体场景而非罗列", "策略模式（支付/规则引擎）", "模板方法（流程编排）", "工厂模式（渠道适配）"],
         tags=["设计模式", "项目经验"], source="manual"),
    dict(category="behavioral", position_tag="java_backend", difficulty="easy",
         question="如果上线后出现生产事故，你会怎么应对？",
         reference_answer="标准流程：1) 快速止损（回滚/切流/降级）优先于定位原因；2) 同步 TL/相关方；3) 保留现场日志和 dump；4) 事后复盘（5 Whys）+ 改进措施进 work item。",
         key_points=["止损优先于定位", "及时同步", "保留现场", "事后复盘形成闭环"],
         tags=["行为面试", "故障处理", "线上应急"], source="manual"),

    # ==================== 前端 Vue/React (vue_frontend) ====================
    dict(category="self-intro", position_tag="vue_frontend", difficulty="easy",
         question="请做个自我介绍，重点突出你的前端项目经历和技术栈。",
         reference_answer="面试官你好，我是一名前端开发，主要使用 Vue3 + TypeScript...",
         key_points=["清晰表达年限", "Vue/React 技术栈", "提及代表性项目"],
         tags=["自我介绍", "开场"], source="manual"),
    dict(category="technical", position_tag="vue_frontend", difficulty="easy",
         question="Vue 3 Composition API 和 Options API 有什么区别？你更推荐哪个？",
         reference_answer="Options API 按选项分类 (data/methods/computed/watch)，适合简单组件。Composition API 按逻辑关注点组织 (setup)，更适合：1) 复杂组件逻辑拆分；2) 逻辑复用 (composables)；3) 更好的 TypeScript 支持。推荐新项目用 Composition API。",
         key_points=["Options 按选项分类", "Composition 按逻辑归类", "composables 逻辑复用", "TypeScript 支持更好"],
         tags=["Vue3", "Composition API"], source="manual"),
    dict(category="technical", position_tag="vue_frontend", difficulty="medium",
         question="Vue Router 的路由懒加载是怎么实现的？有什么好处？",
         reference_answer="利用 ES 动态 import() 语法：component: () => import('./views/About.vue')。Vite/Webpack 会为每个懒加载路由生成独立 chunk (code split)，用户访问时才加载。好处：首屏加载快、按需加载减少带宽浪费。结合 prefetch 预加载可进一步优化。",
         key_points=["ES dynamic import()", "code split 独立 chunk", "减少首屏体积", "prefetch 预加载策略"],
         tags=["Vue Router", "性能优化", "懒加载"], source="manual"),
    dict(category="technical", position_tag="vue_frontend", difficulty="medium",
         question="谈谈 virtual DOM 的工作原理，以及 Vue 和 React 在 diff 算法上的差异。",
         reference_answer="virtual DOM 是真实 DOM 的 JS 对象抽象。数据变化 → 生成新的 VDOM → diff 比较新旧 → patch 最小化真实 DOM 操作。Vue3 优化：Block Tree + 静态标记 (PatchFlag)，只 diff 动态节点；React 用 Fiber 架构实现可中断的 Reconciliation。",
         key_points=["VDOM 是 JS 对象抽象", "diff → patch 最小更新", "Vue3 Block Tree + PatchFlag", "React Fiber 可中断"],
         tags=["Virtual DOM", "diff 算法", "Vue3", "React"], source="manual"),
    dict(category="technical", position_tag="vue_frontend", difficulty="hard",
         question="前端性能优化你有哪些实践经验？从网络、渲染、JS 三个层面谈谈。",
         reference_answer="网络：CDN、HTTP2、Gzip/Brotli 压缩、合理的缓存策略、图片 WebP/懒加载。渲染：减少重排重绘、will-change、CSS contain、虚拟列表、requestAnimationFrame。JS：code split、tree shaking、Web Worker 计算密集任务、防抖节流、合理使用 useMemo/useCallback (React) 或 computed (Vue)。",
         key_points=["网络层：CDN/压缩/缓存", "渲染层：减少重排/CSS 优化", "JS 层：code split/防抖节流", "Lighthouse 量化评估"],
         tags=["性能优化", "前端"], source="manual"),
    dict(category="system_design", position_tag="vue_frontend", difficulty="medium",
         question="如果让你设计一个前端监控系统（错误追踪 + 性能监控），你会怎么做？",
         reference_answer="错误监控：window.onerror + unhandledrejection 捕获错误，sourcemap 还原堆栈。性能：Performance API / web-vitals (LCP/FID/CLS)。采集端 SDK → 上报（避免阻塞，用 sendBeacon/图片打点）→ 后端聚合存储（ClickHouse/ES）→ 可视化大盘（Grafana）。采样策略控制上报量。",
         key_points=["错误：onerror + sourcemap", "性能：web-vitals", "sendBeacon 非阻塞上报", "ClickHouse/ES 聚合存储"],
         tags=["系统设计", "前端监控", "Sentry"], source="manual"),
    dict(category="technical", position_tag="vue_frontend", difficulty="medium",
         question="跨域是什么？解决跨域的常用方案有哪些？",
         reference_answer="跨域 (CORS) 由浏览器同源策略限制。方案：1) 服务端 CORS 头（Access-Control-Allow-Origin）；2) 开发代理（Vite proxy）；3) JSONP（仅 GET）；4) Nginx 反向代理；5) WebSocket（不受同源限制）；6) postMessage（iframe 通信）。生产环境优先用 Nginx 反向代理。",
         key_points=["同源策略限制", "CORS 头配置", "开发代理 vs 生产 Nginx", "JSONP/WebSocket 适用场景"],
         tags=["CORS", "跨域", "前端基础"], source="manual"),
    dict(category="project", position_tag="vue_frontend", difficulty="medium",
         question="描述一个你做过的最复杂的前端交互需求，你是怎么实现的？",
         reference_answer="（引导候选人描述复杂交互，如拖拽排序、富文本编辑器、实时协作、复杂表单联动、大屏可视化等）",
         key_points=["需求复杂度说明", "技术选型理由", "实现关键点", "遇到问题和解决方案"],
         tags=["项目经验", "复杂交互"], source="manual"),
    dict(category="behavioral", position_tag="vue_frontend", difficulty="easy",
         question="UI 设计稿还原度一直不达标时，你怎么推动解决？",
         reference_answer="1) 对齐设计规范（间距、颜色、字号变量化）；2) 建立组件库，减少重复造轮子；3) 使用 Figma/Sketch 的 Dev Mode 精确取参数；4) 定期 review 和设计同事同步，发现问题及时沟通而不是积压。",
         key_points=["设计规范 + 变量化", "组件库协同", "定期 review 机制", "主动沟通"],
         tags=["行为面试", "团队协作", "设计还原"], source="manual"),
]


async def main():
    sf = get_session_local()
    async with sf() as db:
        created = 0
        skipped = 0
        for q in QUESTIONS:
            try:
                await question_bank_service.create(db, q)
                created += 1
            except Exception as e:
                print(f"  [SKIP] {q['question'][:30]}... | {e}")
                skipped += 1
        await db.commit()
        print(f"Done: created {created}, skipped {skipped}")


if __name__ == "__main__":
    asyncio.run(main())
