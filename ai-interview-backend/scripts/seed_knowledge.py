"""
批量导入知识库文档 — 创建 markdown 文件 + 触发摄入流水线。

用法:
    docker exec shilian-app python scripts/seed_knowledge.py
"""
import asyncio
from pathlib import Path

from app.core.security import get_password_hash
from app.db.base import get_session_local
from app.models.knowledge import KnowledgeDocument
from app.services.backoffice.knowledge_service import KnowledgeService

UPLOAD_DIR = Path("/app/uploads/knowledge")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# (文件名, 内容, category, description)
DOCUMENTS = [
    ("Python后端面试高频考点.md", """# Python 后端面试高频考点

## 1. Python 语言基础

### GIL 全局解释器锁
- GIL 是 CPython 解释器层面的互斥锁
- 同一时刻只有一个线程执行 Python 字节码
- CPU 密集型任务推荐 multiprocessing，IO 密集型线程依然有效
- Python 3.9+ 的 sub-interpreters 和 nogil 项目是未来方向

### 装饰器
- 本质是接受函数、返回新函数的高阶函数
- 保留元信息使用 `functools.wraps`
- 带参数的装饰器需要额外一层包装函数
- 常见应用：日志、计时、权限检查、缓存、路由注册

### async/await 协程
- 基于事件循环（Event Loop）的协作式多任务
- `async def` 定义协程，`await` 挂起等待 IO 完成
- 相比多线程：无 GIL 竞争、低上下文切换开销
- 适用场景：IO 密集型高并发（Web 服务、爬虫、消息处理）

## 2. Web 框架

### FastAPI 核心特性
- 基于 Starlette + Pydantic
- 自动生成 OpenAPI/Swagger 文档
- 依赖注入系统（Depends）
- async/await 原生支持
- 请求/响应自动校验

### Django vs FastAPI 选型
| 场景 | 推荐 |
|------|------|
| 全栈应用（ORM+模板+Admin） | Django |
| 纯 API / 微服务 | FastAPI |
| WebSocket 密集型 | FastAPI |
| 内容管理系统 | Django |

## 3. 数据库与 ORM

### SQLAlchemy 最佳实践
- Session 管理：每个请求一个 session（FastAPI Depends）
- N+1 问题：使用 joinedload / selectinload 预加载关联数据
- 批量操作：bulk_insert_mappings 而非逐条 add
- 异步：AsyncSession + asyncpg 驱动

### MySQL 索引优化
- 最左前缀原则
- 覆盖索引避免回表查询
- EXPLAIN 分析执行计划
- 索引下推 (ICP) 和 MRR 优化

## 4. 缓存与消息队列

### Redis 应用场景
- 缓存热点数据（string/hash）
- 分布式锁（SET NX EX）
- 计数器/限流（INCR + EXPIRE）
- 消息队列（List/BLPop）
- 排行榜（Sorted Set）

### 消息队列选型
- RabbitMQ：可靠消息投递，适合企业级
- Kafka：高吞吐日志/事件流，大数据生态
- Redis Streams：轻量级，部署简单

## 5. 系统设计

### 微服务架构要点
- 服务拆分原则（DDD 限界上下文）
- 服务间通信（HTTP/gRPC/消息队列）
- 分布式事务（Saga 模式、本地消息表）
- API 网关（鉴权、限流、路由）

### 常见系统设计题思路
1. **短链接系统**：hash+Base62 → DB → 302 跳转 → 缓存热数据
2. **限流系统**：令牌桶/滑动窗口 + Redis Lua 原子操作
3. **分布式 ID 生成**：Snowflake/号段模式
""", "backend", "Python 后端面试知识体系总结"),

    ("微服务架构设计指南.md", """# 微服务架构设计指南

## 服务拆分原则

### 限界上下文（Bounded Context）
- 从业务领域出发，识别核心子域
- 每个微服务对应一个限界上下文
- 服务间通过 API/事件松耦合通信

### 拆分粒度
- **过细**：运维复杂度爆炸，网络开销大
- **过粗**：退化为单体，失去微服务优势
- **经验法则**：一个团队可以独立负责 2-3 个服务

## 服务间通信

### 同步通信（请求-响应）
- RESTful API：通用、简单
- gRPC：高性能、强类型、Protocol Buffers
- GraphQL：灵活查询、前端驱动

### 异步通信（事件驱动）
- 消息队列（Kafka/RabbitMQ）
- 事件总线
- CQRS + Event Sourcing（复杂业务场景）

## 分布式事务

### Saga 模式
- **编排型**：中心编排器协调各服务
- **编排/协同型**：事件驱动，各服务自主响应
- 补偿事务（冲正）

### 最终一致性
- 本地消息表 + 定时任务
- 事务性发件箱（Transactional Outbox）
- CDC + Debezium

## 可观测性

### 三要素
1. **日志**：结构化日志（JSON）、traceId 关联
2. **指标**：Prometheus + Grafana（QPS/延迟/错误率）
3. **链路追踪**：Jaeger/Zipkin + OpenTelemetry

### 告警策略
- 错误率 > 1% → P2
- 延迟 P99 > 1s → P2
- 可用性 < 99.9% → P1
- 磁盘/内存 > 80% → P1

## API 网关

### 核心能力
- 统一鉴权（JWT/OAuth2）
- 限流（令牌桶/漏桶算法）
- 路由转发（基于 path/header）
- 请求/响应转换
- API 版本管理

### 常用方案
- Kong / APISIX（OpenResty 生态）
- Spring Cloud Gateway（Java 生态）
- Envoy（Service Mesh 数据面）
""", "backend", "微服务架构设计核心概念和最佳实践"),

    ("前端性能优化实战.md", """# 前端性能优化实战

## 性能指标体系

### Core Web Vitals
- **LCP (Largest Contentful Paint)**：最大内容绘制，< 2.5s
- **FID (First Input Delay)**：首次输入延迟，< 100ms
- **CLS (Cumulative Layout Shift)**：累积布局偏移，< 0.1

### 补充指标
- **TTFB (Time to First Byte)**：首字节时间，< 800ms
- **FCP (First Contentful Paint)**：首次内容绘制，< 1.8s
- **TTI (Time to Interactive)**：可交互时间，< 3.8s

## 网络层优化

### 资源加载
- HTTP/2 多路复用（单连接并发）
- Gzip/Brotli 压缩（文本资源可减少 60-80%）
- CDN 就近访问 + 缓存
- DNS Prefetch：`<link rel="dns-prefetch">`
- Preload 关键资源：`<link rel="preload">`

### 缓存策略
- 静态资源 hash 命名（强缓存 1 年）
- HTML 协商缓存（ETag/Last-Modified）
- Service Worker 离线缓存（PWA）

### 图片优化
- WebP/AVIF 格式（比 JPEG 小 30-50%）
- 响应式图片（srcset + sizes）
- 懒加载（lazy loading attribute）
- 渐进式 JPEG（先低画质后高清）

## 渲染层优化

### 关键渲染路径
1. HTML → DOM Tree
2. CSS → CSSOM Tree
3. DOM + CSSOM → Render Tree
4. Layout（计算位置）
5. Paint（绘制像素）

### 减少重排 (Reflow) 和重绘 (Repaint)
- 批量修改 DOM（DocumentFragment）
- CSS class 一次性变更，避免逐条改 style
- 读写分离（避免 forced synchronous layout）
- `will-change` 提示 GPU 加速
- 动画使用 `transform` 和 `opacity`（GPU 合成层）

### 虚拟列表
- 长列表只渲染可视区域
- 方案：vue-virtual-scroller、react-window

## JS 层优化

### Code Split 代码分割
- 路由懒加载：`() => import('./Page.vue')`
- 组件按需加载
- Webpack SplitChunksPlugin / Vite rollup 自动分割

### Tree Shaking
- ES Module 静态分析做死代码消除
- 避免副作用导入（`import 'lib'` 可能整个打入）
- Lodash → lodash-es

### 防抖 (Debounce) 与节流 (Throttle)
- 防抖：输入框搜索（最后一次触发后 N ms 执行）
- 节流：滚动事件、resize（固定间隔执行）

### Web Worker
- 将 CPU 密集型计算移到 Worker 线程
- 不阻塞主线程 UI 渲染
- Comlink 库简化 Worker 通信

## 构建优化

### Vite
- 开发环境原生 ESM（秒启）
- 生产环境 Rollup 打包
- esbuild 预构建依赖（快 10-100 倍）

### Webpack
- thread-loader / HappyPack 多线程构建
- cache-loader / hard-source-webpack-plugin 缓存
- externals 分离大型库（可用 CDN 加载）
""", "frontend", "前端性能优化知识体系"),

    ("Java面试核心知识点.md", """# Java 面试核心知识点

## 1. Java 基础

### String 不可变性
- String 底层 final char[]（JDK8）或 final byte[]（JDK9+）
- 字符串常量池（String Pool）复用
- StringBuilder 可变、非线程安全，适合单线程拼接
- StringBuffer 可变、线程安全（synchronized）

### HashMap 原理
- JDK8：数组 + 链表 + 红黑树
- 链表长度 > 8 且数组长度 >= 64 → 红黑树
- 扩容：容量翻倍、重新 hash（JDK8 尾插法防死循环）
- 线程不安全（put 可能导致数据覆盖），并发用 ConcurrentHashMap

## 2. JVM

### 内存模型
- 堆：Edon + S0 + S1（年轻代）+ Old（老年代）
- 方法区/元空间（Metaspace JDK8+）
- 虚拟机栈（栈帧 → 局部变量表、操作数栈）
- 本地方法栈
- 程序计数器（线程私有）

### GC 算法
- 标记-清除：产生碎片
- 复制：年轻代常用（Eden → Survivor）
- 标记-整理：老年代常用
- CMS（JDK8 老年代）：标记-清除，低延迟但易碎片化
- G1（JDK9+ 默认）：Region 分区，可预测停顿
- ZGC（JDK11+）：超低延迟（< 10ms），支持 TB 级堆

### Full GC 排查
- jstat -gcutil 监控 GC 频率和时间
- GC 日志分析（-Xlog:gc*）
- 堆 dump（jmap）+ MAT / JProfiler 分析
- 常见原因：老年代满、System.gc()、元空间溢出、大对象直接晋升

## 3. Spring 框架

### IoC 容器
- Bean 生命周期：实例化 → 属性注入 → Aware → 初始化 → 就绪 → 销毁
- 作用域：singleton / prototype / request / session
- 循环依赖：三级缓存（singletonObjects / earlySingletonObjects / singletonFactories）

### AOP 原理
- JDK 动态代理（接口代理）
- CGLIB 代理（子类代理，final 方法不可代理）
- Spring Boot 2.x 默认 CGLIB

### 事务
- `@Transactional` 原理：AOP 代理
- 传播行为：REQUIRED（默认）/ REQUIRES_NEW / NESTED
- 自调用事务失效（this.method() 不走代理）

## 4. MySQL

### 索引
- 聚簇索引（主键，叶子存完整行数据）
- 非聚簇索引（二级索引，叶子存主键值）
- 回表：二级索引找到主键，再查聚簇索引
- 覆盖索引：查询列全部在索引中，避免回表

### 事务隔离级别
| 级别 | 脏读 | 不可重复读 | 幻读 |
|------|------|------------|------|
| READ UNCOMMITTED | ✓ | ✓ | ✓ |
| READ COMMITTED | ✗ | ✓ | ✓ |
| REPEATABLE READ (默认) | ✗ | ✗ | ✓(部分解决) |
| SERIALIZABLE | ✗ | ✗ | ✗ |

### 锁
- 行锁（InnoDB）：Record Lock / Gap Lock / Next-Key Lock
- 表锁：意向锁（IX/IS）
- 死锁：交叉等待 → 超时回滚或死锁检测

## 5. 并发编程

### 线程池
- corePoolSize → maxPoolSize → 拒绝策略
- 工作队列：ArrayBlockingQueue / LinkedBlockingQueue
- 拒绝策略：Abort（抛异常）/ CallerRuns（主线程执行）/ Discard（丢弃）/ DiscardOldest
- **禁止 Executors 创建**（无界队列 OOM、线程数无上限）
""", "backend", "Java 后端面试核心知识点整理"),
]


async def main():
    ks = KnowledgeService()

    sf = get_session_local()
    async with sf() as db:
        created = 0
        for filename, content, category, description in DOCUMENTS:
            # Write file
            title = filename.rsplit(".", 1)[0]
            ext = filename.rsplit(".", 1)[-1]
            file_bytes = content.encode("utf-8")

            # Create DB record
            doc = KnowledgeDocument(
                title=title,
                file_name=filename,
                file_type=ext,
                file_size=len(file_bytes),
                category=category,
                description=description,
                status="pending",
            )
            db.add(doc)
            await db.commit()
            await db.refresh(doc)

            # Save file to disk
            save_path = UPLOAD_DIR / f"{doc.id}_{filename}"
            save_path.write_bytes(file_bytes)
            doc.file_url = str(save_path)
            await db.commit()

            # Trigger async ingestion
            await ks.ingest_from_path(doc.id, str(save_path), ext)
            created += 1
            print(f"  [{created}/{len(DOCUMENTS)}] {title}")

        print(f"Done: {created} documents ingested")


if __name__ == "__main__":
    asyncio.run(main())
